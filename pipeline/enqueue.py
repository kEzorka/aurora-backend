"""CLI постановки срока в устойчивую очередь инференса."""

import argparse
import os
from pathlib import Path
from typing import Final

from pipeline.queue import enqueue, open_queue

QUEUE_ENV: Final = "AURORA_QUEUE"
DEFAULT_QUEUE: Final = Path("artifacts/jobs.sqlite")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--init", required=True, help="срок вида 2026-08-01T00:00:00Z")
    cli.add_argument(
        "--queue",
        type=Path,
        default=Path(os.environ.get(QUEUE_ENV, DEFAULT_QUEUE)),
        help=f"SQLite queue (env {QUEUE_ENV})",
    )
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    connection = open_queue(args.queue)
    try:
        job = enqueue(connection, args.init)
    finally:
        connection.close()
    print(f"job={job.id} init_time={job.init_time} status={job.status} attempts={job.attempts}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
