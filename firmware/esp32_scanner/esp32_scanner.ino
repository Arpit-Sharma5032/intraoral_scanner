// ============================================================
// IOSCAN - ESP32 INTRAORAL SCANNER FIRMWARE  (BNO086 over I2C)
// ESP32 + BNO086 (7Semi Qwiic board) + OV2311 Camera + Laser MOSFET
//
// Serial protocol (115200 baud):
//   @timestamp_us,laser_on,qw,qx,qy,qz,frame_sequence*CRC16
//
// Wiring (see I2C connection sheet):
//   GPIO21 SDA  -> BNO086 SDA          GPIO22 SCL -> BNO086 SCL
//   GPIO4  INT  -> BNO086 INT (wired, not required by the library)
//   GPIO17 RST  -> BNO086 NRST         3V3 / GND
//   GPIO27      -> HW-042 SIG (laser MOSFET gate)
//   GPIO34      <- OV2311 STROBE (input-only pin, 3.3 V logic)
//
// IMPORTANT:
// - Camera STROBE synchronizes laser and frame events.
// - First exposure is OFF, then laser alternates OFF/ON.
// - No free-running fallback is permitted.
// - Missing STROBE forces laser OFF.
// - Invalid IMU startup is rejected; stale samples are warned about without dropping STROBE packets.
//
// Board: classic ESP32 (ESP32 Dev Module).
// ============================================================

#include <Arduino.h>
#include <stdio.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include "soc/soc.h"
#include "soc/gpio_reg.h"

// IMU debug counters (declare after Arduino types are included).
uint32_t dbg_events = 0, dbg_resets = 0;
uint32_t dbg_poll_calls = 0, dbg_empty_polls = 0;
uint8_t  dbg_last_id = 0;
uint8_t  bno_i2c_addr = 0;

// ---------------- PIN CONFIGURATION ----------------

// BNO086 I2C
static const int PIN_I2C_SDA = 21;
static const int PIN_I2C_SCL = 22;
static const int PIN_BNO_INT = 4;    // wired for future use; Adafruit library polls over I2C
static const int PIN_BNO_RST = 17;

// Laser and camera
static const int PIN_LASER_GATE = 27;   // must be < 32
static const int PIN_CAM_STROBE = 34;   // input-only pin, must be >= 32

static const uint32_t LASER_MASK = (1UL << PIN_LASER_GATE);

// Fast register access (safe inside an ISR, no flash-cache dependency).
#define LASER_ON_FAST()   REG_WRITE(GPIO_OUT_W1TS_REG, LASER_MASK)
#define LASER_OFF_FAST()  REG_WRITE(GPIO_OUT_W1TC_REG, LASER_MASK)

#define DEBUG_IMU 0   // production: keep Serial free for synchronized scan packets

// ---------------- TIMING / I2C ----------------

static const uint32_t STROBE_TIMEOUT_US = 100000;
static const uint32_t MAX_IMU_AGE_US    = 50000;
static const uint32_t I2C_CLOCK_HZ      = 400000;   // matches the known-working diagnostic sketch
static const uint8_t  BNO_ADDR_PRIMARY   = 0x4B;    // try the address used first by the known-working sketch
static const uint8_t  BNO_ADDR_SECONDARY = 0x4A;

// ---------------- BNO086 ----------------

Adafruit_BNO08x bno08x(PIN_BNO_RST);
sh2_SensorValue_t sensorValue;

float q_w = 1.0f;
float q_x = 0.0f;
float q_y = 0.0f;
float q_z = 0.0f;

uint32_t last_imu_sample_us = 0;
bool imu_ever_sampled = false;

bool bno_ok = false;

// ---------------- CAMERA ISR STATE ----------------

// Single-producer (ISR) / single-consumer (loop) ring buffer so that two
// strobes arriving before the loop runs do not lose a packet.
static const uint8_t RING_SIZE = 128;           // power of two; 127-event capacity (~2.54 s at 50 FPS)
static const uint8_t RING_MASK = RING_SIZE - 1;

volatile uint32_t ring_ts[RING_SIZE];
volatile uint32_t ring_seq[RING_SIZE];
volatile uint8_t  ring_laser[RING_SIZE];
volatile uint8_t  ring_head = 0;                // written by ISR only
volatile uint8_t  ring_tail = 0;                // written by loop only
volatile uint32_t ring_overflow = 0;

volatile uint32_t isr_frame_seq = 0;
volatile bool next_laser_state = false;         // first exposure is OFF

uint32_t last_strobe_seen_us = 0;
uint32_t last_overflow_reported = 0;
volatile uint32_t isr_rising_edges = 0;
volatile uint32_t isr_falling_edges = 0;

bool strobe_warning_sent = false;
bool bno_error_sent = false;
uint32_t last_imu_stale_warning_us = 0;

// Forward declaration: setup() warms the IMU before enabling the STROBE ISR.
void poll_imu();

// ---------------- CAMERA STROBE INTERRUPT ----------------

void IRAM_ATTR onCameraStrobeChange() {

    // Use the same Arduino GPIO input API as the earlier working firmware.
    // GPIO34 is input-only on the classic ESP32. Keep the edge handling simple
    // and observable while diagnosing the new I2C-IMU firmware.
    const int level = digitalRead(PIN_CAM_STROBE);

    if (level == HIGH) {
        isr_rising_edges++;

        // Rising edge: exposure starts. Set the laser for this exposure.
        const bool laser_on = next_laser_state;

        if (laser_on) {
            LASER_ON_FAST();
        } else {
            LASER_OFF_FAST();
        }

        next_laser_state = !laser_on;

        isr_frame_seq++;

        const uint8_t head = ring_head;
        const uint8_t next = (head + 1) & RING_MASK;

        if (next == ring_tail) {
            // Buffer full: drop this event. The sequence number still advanced,
            // so the PC sees a gap instead of silent misalignment.
            ring_overflow++;
        } else {
            ring_ts[head]    = micros();
            ring_seq[head]   = isr_frame_seq;
            ring_laser[head] = laser_on ? 1 : 0;
            ring_head = next;
        }

    } else {
        isr_falling_edges++;

        // Falling edge: laser OFF outside the camera exposure window.
        LASER_OFF_FAST();
    }
}

// ---------------- BNO086 I2C INITIALIZATION ----------------

static bool init_bno086() {

    // Match the known-working standalone diagnostic: explicitly reset the
    // BNO086 BEFORE probing I2C. Adafruit_BNO08x::begin_I2C() probes the
    // address before its internal _init()/hardwareReset(), so relying only
    // on the library reset can be too late when the sensor is stuck at boot.
    pinMode(PIN_BNO_RST, OUTPUT);
    digitalWrite(PIN_BNO_RST, LOW);
    delay(50);
    digitalWrite(PIN_BNO_RST, HIGH);
    delay(100);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Wire.setClock(I2C_CLOCK_HZ);
    Wire.setTimeOut(50);                        // ms; avoids a hung bus blocking the loop
    delay(250);                                 // allow the BNO086 to finish reset/startup

    Serial.printf("# INFO: I2C started SDA=%d SCL=%d clock=%lu Hz\n",
                  PIN_I2C_SDA, PIN_I2C_SCL, (unsigned long)I2C_CLOCK_HZ);

    // Scan the complete bus like the known-working diagnostic sketch.
    // Keep the BNO address if either expected address acknowledges.
    uint8_t found_bno_addr = 0;
    uint8_t found_devices = 0;
    for (uint8_t address = 1; address < 127; ++address) {
        Wire.beginTransmission(address);
        const uint8_t error = Wire.endTransmission();
        if (error == 0) {
            found_devices++;
            Serial.printf("# I2C: device found at 0x%02X\n", address);
            if (address == BNO_ADDR_PRIMARY || address == BNO_ADDR_SECONDARY) {
                if (found_bno_addr == 0 || address == BNO_ADDR_PRIMARY) {
                    found_bno_addr = address;
                }
            }
        }
    }

    if (found_devices == 0) {
        Serial.println("# ERROR: I2C bus empty after BNO086 reset; check 3V3, GND, SDA(21), SCL(22)");
        return false;
    }
    if (found_bno_addr == 0) {
        Serial.printf("# ERROR: I2C has %u device(s), but BNO086 not found at 0x4B or 0x4A\n",
                      (unsigned)found_devices);
        return false;
    }

    bno_i2c_addr = found_bno_addr;
    Serial.printf("# INFO: BNO086 I2C ACK at 0x%02X\n", bno_i2c_addr);

    // begin_I2C() now sees the already-responsive device. Its own internal
    // reset/SH-2 initialization follows the initial I2C address probe.
    if (!bno08x.begin_I2C(bno_i2c_addr, &Wire)) {
        Serial.println("# ERROR: begin_I2C failed: SH-2 startup/product-ID handshake failed");
        Serial.println("# CHECK: verify RST(GPIO17), I2C mode straps (PS0/PS1), and power-cycle the BNO086");
        return false;
    }
    Serial.println("# INFO: BNO08x SH-2 interface initialized");

    // Request Game Rotation Vector at 100 Hz (10 ms report interval).
    // Adafruit enableReport() takes microseconds; 10000 us = 10 ms = 100 Hz.
    if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000)) {
        Serial.println("# ERROR: enableReport(SH2_GAME_ROTATION_VECTOR) failed");
        return false;
    }

    Serial.println("# INFO: Game Rotation Vector report enabled (100 Hz)");
    return true;
}

// ---------------- INITIALIZATION ----------------

void setup() {

    Serial.begin(115200);

    pinMode(PIN_LASER_GATE, OUTPUT);
    digitalWrite(PIN_LASER_GATE, LOW);

    pinMode(PIN_CAM_STROBE, INPUT);
    pinMode(PIN_BNO_INT, INPUT_PULLUP);
    Serial.printf("# INFO: camera STROBE input GPIO%d (input-only; external 3.3V signal required)\n",
                  PIN_CAM_STROBE);

    delay(20);

    if (init_bno086()) {
        bno_ok = true;
        Serial.println("# INFO: BNO086 I2C initialized; Game Rotation Vector at 100 Hz");
    } else {
        bno_ok = false;
        Serial.println("# ERROR: BNO086 I2C initialization failed; scan packets are disabled");
    }

    // Wait for the first real orientation sample BEFORE accepting camera strobes.
    // Without this warm-up, early rising edges can be discarded while the IMU
    // has not produced its first report, creating an immediate sequence gap.
    if (bno_ok) {
        const uint32_t warmup_start_ms = millis();
        while (!imu_ever_sampled && (millis() - warmup_start_ms) < 750U) {
            poll_imu();
            delay(1);
        }
        if (!imu_ever_sampled) {
            bno_ok = false;
            Serial.println("# ERROR: BNO086 initialized but produced no initial orientation sample within 750 ms");
        } else {
            Serial.println("# INFO: first BNO086 orientation sample received; enabling camera STROBE capture");
        }
    }

    last_strobe_seen_us = micros();

    // Camera synchronization interrupt (attach last, after all state is ready)
    attachInterrupt(
        digitalPinToInterrupt(PIN_CAM_STROBE),
        onCameraStrobeChange,
        CHANGE
    );
}

// ---------------- IMU DATA ACQUISITION ----------------

void poll_imu() {

    if (!bno_ok) {
        return;
    }

    dbg_poll_calls++;

    if (bno08x.wasReset()) {
        dbg_resets++;
        if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000)) {
            Serial.println("# ERROR: failed to re-enable 100 Hz Game Rotation Vector after IMU reset");
        }
    }

    bool received_event = false;
    while (bno08x.getSensorEvent(&sensorValue)) {
        received_event = true;
        dbg_events++;
        dbg_last_id = sensorValue.sensorId;

        if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {

            q_w = sensorValue.un.gameRotationVector.real;
            q_x = sensorValue.un.gameRotationVector.i;
            q_y = sensorValue.un.gameRotationVector.j;
            q_z = sensorValue.un.gameRotationVector.k;

            last_imu_sample_us = micros();
            imu_ever_sampled = true;
        }
    }

    if (!received_event) {
        dbg_empty_polls++;
    }
}

// ---------------- FRAME EVENT PROCESSING ----------------

void process_frame_event(uint32_t ts, uint32_t seq, bool laser_on) {

    // Reject frames if IMU initialization failed. In that case the host must
    // see the explicit error rather than a fabricated identity quaternion.
    if (!bno_ok || !imu_ever_sampled) {
        if (!bno_error_sent) {
            Serial.println("# ERROR: BNO086 unavailable; synchronized scan packets are disabled");
            bno_error_sent = true;
        }
        return;
    }

    // IMPORTANT: Never omit a STROBE packet just because the last IMU report
    // is old. Omitting it increments the physical frame sequence but creates
    // an apparent UART packet-loss gap on the PC. Send the latest known pose,
    // and report stale orientation as a quality warning instead. The host can
    // then preserve camera/laser event alignment and flag pose quality.
    const int32_t imu_age_us = (int32_t)(ts - last_imu_sample_us);
    if (imu_age_us > (int32_t)MAX_IMU_AGE_US) {
        const uint32_t now_us = micros();
        if ((uint32_t)(now_us - last_imu_stale_warning_us) >= 1000000UL) {
            Serial.println("# WARN: IMU orientation is stale; repeating latest valid quaternion to preserve STROBE sequence");
            last_imu_stale_warning_us = now_us;
        }
    }

    // Format a CRC-protected CSV packet. Build the COMPLETE frame in memory
    // and submit it with one Serial.write() call to avoid split writes that can
    // interleave with diagnostics or expose partial frames to the UART driver.
    char payload[112];
    const int payload_len = snprintf(
        payload, sizeof(payload),
        "%lu,%d,%.6f,%.6f,%.6f,%.6f,%lu",
        (unsigned long)ts,
        laser_on ? 1 : 0,
        (double)q_w,
        (double)q_x,
        (double)q_y,
        (double)q_z,
        (unsigned long)seq
    );
    if (payload_len <= 0 || payload_len >= (int)sizeof(payload)) {
        Serial.println("# ERROR: failed to format IMU frame packet");
        return;
    }

    uint16_t crc = 0xFFFF;
    for (int i = 0; i < payload_len; ++i) {
        crc ^= ((uint16_t)(uint8_t)payload[i] << 8);
        for (uint8_t bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
        }
    }

    char framed[128];
    const int framed_len = snprintf(
        framed, sizeof(framed), "@%s*%04X\n", payload, (unsigned int)crc
    );
    if (framed_len <= 0 || framed_len >= (int)sizeof(framed)) {
        Serial.println("# ERROR: failed to frame IMU packet");
        return;
    }
    Serial.write((const uint8_t *)framed, (size_t)framed_len);
}

// ---------------- MAIN LOOP ----------------

void loop() {

    // Drain STROBE events first. I2C polling can take milliseconds; putting it
    // first can let the ISR ring fill during a burst of 50 FPS camera edges.
    while (ring_tail != ring_head) {

        const uint8_t tail = ring_tail;

        const uint32_t ts  = ring_ts[tail];
        const uint32_t seq = ring_seq[tail];
        const bool laser_on = ring_laser[tail] != 0;

        ring_tail = (tail + 1) & RING_MASK;

        last_strobe_seen_us = ts;
        strobe_warning_sent = false;

        process_frame_event(ts, seq, laser_on);
    }

    // Refresh orientation after servicing every queued exposure event. Stale
    // samples are warned about but no longer cause STROBE packets to be omitted.
    poll_imu();

    // Report ring overflow (events dropped because the loop was too slow).
    const uint32_t overflow = ring_overflow;
    if (overflow != last_overflow_reported) {
        Serial.printf("# ERROR: STROBE ring overflow; dropped %lu exposure event(s)\n",
                      (unsigned long)(overflow - last_overflow_reported));
        last_overflow_reported = overflow;
    }

    // No free-running fallback. If camera STROBE disappears, the laser stays OFF.
    const int32_t since_strobe_us = (int32_t)(micros() - last_strobe_seen_us);

    if (since_strobe_us > (int32_t)STROBE_TIMEOUT_US) {

        LASER_OFF_FAST();

        if (!strobe_warning_sent) {
            Serial.println(
                "# WARN: Camera STROBE missing; "
                "laser forced OFF; capture paused"
            );
            strobe_warning_sent = true;
        }
    }

    #if DEBUG_IMU
    static uint32_t last_dbg = 0;
    if (millis() - last_dbg > 200) {
        last_dbg = millis();
        Serial.printf(
            "# DBG: bno_ok=%u addr=0x%02X imu_int=%d polls=%lu empty=%lu "
            "imu_events=%lu resets=%lu last_id=%u strobe_pin=%d rises=%lu falls=%lu "
            "seq=%lu overflow=%lu\n",
            bno_ok ? 1U : 0U,
            bno_i2c_addr,
            digitalRead(PIN_BNO_INT),
            (unsigned long)dbg_poll_calls,
            (unsigned long)dbg_empty_polls,
            (unsigned long)dbg_events,
            (unsigned long)dbg_resets,
            dbg_last_id,
            digitalRead(PIN_CAM_STROBE),
            (unsigned long)isr_rising_edges,
            (unsigned long)isr_falling_edges,
            (unsigned long)isr_frame_seq,
            (unsigned long)ring_overflow
        );
    }
    #endif
}
