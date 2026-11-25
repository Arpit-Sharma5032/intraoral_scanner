# ESP32 strobe-sequence gap fix (2026-11-07)

## Root cause corrected

The previous firmware incremented `frame_sequence` on every physical rising camera-STROBE edge, but `process_frame_event()` returned without transmitting a packet whenever the latest BNO086 report was older than 50 ms. The next valid packet therefore appeared to the host as a missing serial packet, even when UART transport had not lost anything. Startup could also accept STROBE edges before the first IMU orientation sample had arrived.

## Changes

- Warm up the BNO086 and require its first real orientation sample before attaching the camera-STROBE interrupt.
- Do not omit a STROBE packet solely because an IMU sample is stale. Repeat the last valid quaternion and emit a throttled `# WARN` diagnostic. Such intervals preserve camera/laser event alignment but are **not** guaranteed to have accurate pose.
- Increase the ISR-to-loop event ring from 16 to 128 slots (127 usable entries, about 2.54 seconds at 50 rising edges/second).
- Drain queued camera events before I2C polling so slow sensor transactions do not unnecessarily delay event service.
- Construct the complete `@...*CRC\n` frame in one buffer and submit it through one `Serial.write()` call.
- Report ring overflow explicitly. A real ring overflow or UART/USB packet loss still causes a sequence gap and safely aborts capture.
- Update the dashboard diagnosis so a sequence gap is not automatically described as UART-only loss.

## Host compatibility

The wire format is unchanged: `@timestamp_us,laser_on,qw,qx,qy,qz,frame_sequence*CRC16` at **115200 baud**. Existing Python CRC validation and strict sequence-gap checks remain enabled.

## Validation and limitation

Python suite: 71 passed, 6 skipped. Python compilation passed. Arduino CLI / PlatformIO is not installed in this build environment, so this firmware has not been compiled, flashed, or electrically validated here. Compile the sketch with the same ESP32 board core and Adafruit BNO08x library used by your hardware before uploading.
