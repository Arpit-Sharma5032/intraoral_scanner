# ESP32 scanner firmware

This sketch is the hardware reference supplied for IOSCAN, updated for safer capture handshaking.

## Serial protocol

At 115200 baud, one framed CSV packet is emitted for each accepted rising camera-STROBE edge:

```text
@timestamp_us,laser_on,qw,qx,qy,qz,frame_sequence*CRC16
```

- `@`: packet start marker used for resynchronization.
- `timestamp_us`: ESP32 `micros()` at the camera exposure edge (uint32; host extends wrap).
- `laser_on`: laser state assigned to that camera exposure (`0` = off, `1` = on).
- `qw,qx,qy,qz`: normalized by the host; quaternion order is WXYZ.
- `frame_sequence`: increments on every physical camera-STROBE rising edge. Gaps indicate a firmware ring overflow or serial transport loss and cause capture to stop rather than silently mispair frames.
- If the IMU report becomes stale, firmware repeats the latest valid quaternion and emits a throttled `# WARN`; it does not omit a STROBE packet. This preserves camera/laser sequence alignment, but scans with stale-pose warnings must be treated as motion-quality failures.
- `CRC16`: four hexadecimal digits containing CRC-16/CCITT-FALSE over the CSV payload (the bytes between `@` and `*`).
- `# INFO`, `# WARN`, and `# ERROR` lines are diagnostics and are not frame packets.
- The host verifies CRC before parsing and rejects corrupt packets; it never repairs a missing timestamp by guessing.

The first exposure is laser-OFF, then OFF/ON alternates. This ensures there is an ambient reference image before the first laser-ON profile.

## Important behaviour

- The laser is forced LOW if camera STROBE disappears for 100 ms.
- The sketch does not fabricate 50 Hz frame packets when STROBE is missing. A scan without real frame synchronization must fail rather than create misleading geometry.
- If the BNO086 fails to initialize, the sketch reports `# ERROR` and emits no valid frame packets. The host will show the hardware error instead of accepting the default identity quaternion.
- Stale BNO086 samples are flagged, but the latest valid quaternion is repeated so every STROBE retains a corresponding packet. The host records the warning; pose quality is not guaranteed during those intervals.

## Wiring / validation requirements

The sketch currently communicates with the BNO086 over I2C at 400 kHz (SDA GPIO21, SCL GPIO22). It does not use SPI. The sketch assumes the pin mapping in its header. Verify the board revision, voltage levels, laser driver circuit, camera STROBE polarity, and BNO086 I2C wiring against the actual schematic before powering the hardware. In particular, GPIO34 is input-only and expects a 3.3 V-compatible STROBE signal.

Compile/upload with the Arduino ESP32 core and Adafruit BNO08x library versions selected for the actual board. The firmware has not been flashed or electrically tested by this project build.
