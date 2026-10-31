// IOSCAN ESP32 firmware
// ESP32 + BNO086 (hardware VSPI) + laser MOSFET + OV2311 frame STROBE.
// Serial protocol (921600 baud):
//   timestamp_us,laser_on,qw,qx,qy,qz,frame_sequence
// timestamp_us is the uint32 micros() value at the rising camera-STROBE edge.
// The host extends micros() wrap and checks frame_sequence for dropped packets.
//
// IMPORTANT: free-running laser/serial fallback is intentionally disabled for real scans.
// Without camera STROBE, images and laser/IMU packets are not synchronized and must not be
// presented as a valid scan. The laser is forced LOW if the strobe disappears.

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>

// BNO086 hardware VSPI wiring
static const int PIN_SPI_SCK    = 18;
static const int PIN_SPI_MISO   = 19;
static const int PIN_SPI_MOSI   = 23;
static const int PIN_BNO_CS     = 5;
static const int PIN_BNO_INT    = 4;
static const int PIN_BNO_WAKE   = 16;
static const int PIN_BNO_RST    = 17;

// Laser and camera synchronization
static const int PIN_LASER_GATE = 27;
static const int PIN_CAM_STROBE = 34;  // input-only GPIO; 3.3 V logic required
static const uint32_t STROBE_TIMEOUT_US = 100000;  // force laser off after 100 ms without edges
static const uint32_t MAX_IMU_AGE_US = 50000;      // reject stale BNO086 orientation samples

Adafruit_BNO08x bno08x(PIN_BNO_RST);
sh2_SensorValue_t sensorValue;

// Latest BNO086 quaternion [qw, qx, qy, qz].
volatile float q_w = 1.0f, q_x = 0.0f, q_y = 0.0f, q_z = 0.0f;
volatile uint32_t last_imu_sample_us = 0;
bool bno_ok = false;

// ISR state. One event is generated for every rising camera-STROBE edge.
volatile uint32_t isr_frame_ts_us = 0;
volatile uint32_t isr_frame_seq = 0;
volatile bool isr_laser_on = false;
volatile bool isr_new_frame = false;
volatile bool next_laser_state = false; // first exposure OFF, then alternate OFF/ON

uint32_t last_strobe_seen_us = 0;
bool strobe_warning_sent = false;
bool bno_error_sent = false;

void IRAM_ATTR onCameraStrobeChange() {
  const int level = digitalRead(PIN_CAM_STROBE);
  if (level == HIGH) {
    const bool laser_on = next_laser_state;
    if (laser_on) {
      GPIO.out_w1ts = (1UL << PIN_LASER_GATE);
    } else {
      GPIO.out_w1tc = (1UL << PIN_LASER_GATE);
    }
    next_laser_state = !laser_on;
    isr_frame_ts_us = micros();
    isr_laser_on = laser_on;
    isr_frame_seq++;
    isr_new_frame = true;
  } else {
    // Laser is enabled only during the camera exposure window.
    GPIO.out_w1tc = (1UL << PIN_LASER_GATE);
  }
}

void setup() {
  Serial.begin(921600);

  pinMode(PIN_LASER_GATE, OUTPUT);
  digitalWrite(PIN_LASER_GATE, LOW);
  pinMode(PIN_CAM_STROBE, INPUT);

  // Select BNO086 SPI mode before reset release.
  pinMode(PIN_BNO_WAKE, OUTPUT);
  digitalWrite(PIN_BNO_WAKE, HIGH);
  pinMode(PIN_BNO_CS, OUTPUT);
  digitalWrite(PIN_BNO_CS, HIGH);
  delay(20);

  SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI, PIN_BNO_CS);
  if (bno08x.begin_SPI(PIN_BNO_CS, PIN_BNO_INT, &SPI)) {
    bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 4000);  // 250 Hz
    bno_ok = true;
    Serial.println("# INFO: BNO086 SPI initialized; Game Rotation Vector at 250 Hz");
  } else {
    bno_ok = false;
    Serial.println("# ERROR: BNO086 SPI initialization failed; scan packets are disabled");
  }

  attachInterrupt(digitalPinToInterrupt(PIN_CAM_STROBE), onCameraStrobeChange, CHANGE);
  last_strobe_seen_us = micros();
}

void poll_imu() {
  if (!bno_ok) return;

  if (bno08x.wasReset()) {
    bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 4000);
  }

  while (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {
      q_w = sensorValue.un.gameRotationVector.real;
      q_x = sensorValue.un.gameRotationVector.i;
      q_y = sensorValue.un.gameRotationVector.j;
      q_z = sensorValue.un.gameRotationVector.k;
      last_imu_sample_us = micros();
    }
  }
}

void loop() {
  poll_imu();
  const uint32_t now_us = micros();

  if (isr_new_frame) {
    noInterrupts();
    const uint32_t ts = isr_frame_ts_us;
    const uint32_t seq = isr_frame_seq;
    const bool laser_on = isr_laser_on;
    isr_new_frame = false;
    interrupts();

    last_strobe_seen_us = ts;
    strobe_warning_sent = false;

    if (!bno_ok) {
      if (!bno_error_sent) {
        Serial.println("# ERROR: BNO086 unavailable; refusing to send identity/fake orientation");
        bno_error_sent = true;
      }
    } else {
      const uint32_t imu_age_us = now_us - last_imu_sample_us;
      if (last_imu_sample_us == 0 || imu_age_us > MAX_IMU_AGE_US) {
        Serial.println("# WARN: BNO086 orientation sample is stale; this frame is omitted");
      } else {
        Serial.printf("%lu,%d,%.6f,%.6f,%.6f,%.6f,%lu\n",
                      (unsigned long)ts,
                      laser_on ? 1 : 0,
                      (double)q_w, (double)q_x, (double)q_y, (double)q_z,
                      (unsigned long)seq);
      }
    }
  }

  // No asynchronous fallback: without frame STROBE, do not pulse the laser or emit
  // synthetic frame records. This prevents unsynchronized frames from entering the scan.
  if ((uint32_t)(now_us - last_strobe_seen_us) > STROBE_TIMEOUT_US) {
    digitalWrite(PIN_LASER_GATE, LOW);
    if (!strobe_warning_sent) {
      Serial.println("# WARN: OV2311 camera STROBE not detected; laser forced OFF, capture data paused");
      strobe_warning_sent = true;
    }
  }
}
