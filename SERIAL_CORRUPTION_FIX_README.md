# IOSCAN serial corruption / packet resynchronization update

## What this patch changes

1. ESP32 UART is set to **115200 baud**.
2. Python capture uses **115200 baud** by default. The estimated packet load at 50 strobe events/s is only a few kB/s, below the 115200-baud 8-N-1 capacity.
3. The host parser now splits a serial read at every `@` start marker. This can recover a later intact CRC-framed packet when a missing/corrupted newline causes multiple packets to be returned by one `readline()` call.
4. CRC validation remains mandatory. Corrupt packets are rejected.
5. Frame-sequence continuity remains strict. If a packet is lost, capture still stops rather than pairing the wrong camera frame and IMU pose.

## Install

- Upload `firmware/esp32_scanner/esp32_scanner.ino` to the ESP32.
- Set Arduino Serial Monitor to **115200 baud** when viewing diagnostics.
- Close Serial Monitor before launching IOSCAN; only one process can own COM8 at a time.
- The IOSCAN host uses 115200 automatically unless the capture configuration explicitly overrides it.

## What this patch does NOT prove

The supplied log contains invalid bytes inserted into ASCII packets and CRC failures. That is consistent with UART/USB transport corruption or electrical interference; software cannot reconstruct the original values. This patch improves packet resynchronization and reduces UART signalling speed, but physical link integrity must still be verified.

If CRC failures continue at 115200 baud:

1. Test with the laser driver physically powered OFF while leaving the camera and ESP32 powered. If corruption disappears, suspect laser-driver switching noise or shared-ground voltage bounce.
2. Use a short, known-good USB data cable and a direct PC USB port (no hub).
3. Power the laser from its appropriate supply, keep its high-current return away from the ESP32/USB ground path, and use a deliberate common signal reference. Do not disconnect protective ground or exceed ESP32 pin voltage limits.
4. Capture the ESP32 TX waveform with a logic analyser at 115200 baud. If the bytes are already wrong on TX, inspect firmware/board/power; if TX is clean but the PC log is wrong, inspect cable/USB-UART/driver.

Do not relax CRC or sequence checks to make scans proceed. A scan with unverified timestamp/quaternion packets is not safe to reconstruct.
