"""Launch the NEXUS API and built visual knowledge explorer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .api import create_app
from .factory import create_gui_resources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument(
        "--investigations-db", type=Path, default=Path(".nexus/investigations.sqlite")
    )
    parser.add_argument("--runtime-db", type=Path, default=Path(".nexus/runtime.sqlite"))
    parser.add_argument("--frontend-dir", type=Path, default=Path("frontend/dist"))
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.snapshot is not None and not arguments.snapshot.is_file():
        raise SystemExit(f"knowledge snapshot does not exist: {arguments.snapshot}")
    resources = create_gui_resources(
        snapshot=arguments.snapshot,
        investigations_db=arguments.investigations_db,
        runtime_db=arguments.runtime_db,
    )
    application = create_app(resources.service, frontend_dir=arguments.frontend_dir)
    try:
        uvicorn.run(
            application,
            host=arguments.host,
            port=arguments.port,
            log_level=arguments.log_level,
        )
    finally:
        resources.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
