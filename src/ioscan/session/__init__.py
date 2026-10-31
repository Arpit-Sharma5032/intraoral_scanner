"""Session format: the contract between (simulator | real logger) and the pipeline.

Planned on-disk layout per session directory:
    meta.json          calibration, config snapshot, timestamps base
    frames/            000000.png ...   (mono 8/16-bit)
    frames.csv         frame_idx, t_ns, laser_on
    imu.csv            t_ns, qw, qx, qy, qz
    gt/                (simulator only) true poses, true profiles
"""
