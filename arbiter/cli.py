"""Command line interface for ARBITER.

Usage:
    arbiter --help
    arbiter serve --port 8000 --device cuda:0
    arbiter --version
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="arbiter",
        description="ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show ARBITER version and exit",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")
    subparsers.add_parser(
        "serve",
        help="Launch the ARBITER online inference micro-service",
        add_help=False,
    )

    if not argv:
        parser.print_help()
        return 0

    if argv[0] in ("-v", "--version"):
        from . import __version__
        print(f"arbiter {__version__}")
        return 0

    if argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    if argv[0] == "serve":
        try:
            from .serve.__main__ import main as serve_main
        except ImportError as exc:
            print(
                f"Error: Serving dependencies are not installed ({exc}).\n"
                "Please install them via:\n"
                "    pip install \"spectral-arbiter[serve]\"\n"
                "or install uvicorn and fastapi directly:\n"
                "    pip install uvicorn fastapi pydantic httpx",
                file=sys.stderr,
            )
            return 1
        return serve_main(argv[1:])

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
