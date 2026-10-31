from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local IOSCAN operator dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; localhost is safest for a scanner workstation.")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Dashboard dependencies missing. Install with: python -m pip install -e '.[dashboard]'") from exc
    from ioscan.dashboard.server import create_app
    uvicorn.run(create_app(ROOT), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
