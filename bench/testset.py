"""The fixed set of cases every measurement in this repo runs against.

Kept in its own module, away from the code that measures, so that a number
quoted anywhere — latency, RMSE, read throughput — names a case that can be
looked up rather than whatever the author happened to type that day. Change a
case and every chart has to be regenerated; that is the point.

The inits are not four arbitrary dates. Each one is here because it exercises
something the others do not:

* `2026-04-05T00` — an ordinary init in the interior of a month, the case with
  nothing special about it. Everything else is measured against this one.
* `2026-05-01T00` — the month boundary. Its `init - 6 h` lives in the previous
  month, which under the raw NetCDF layout means a second file. Any indexing
  bug that treats a month as self-contained shows up here and nowhere else.
* `2026-06-01T00` — the same boundary in a different season, so that a result
  which depends on the atmosphere rather than on the code can be told apart
  from one that does not.
* `2026-06-15T12` — a midday init. Every other case starts at 00 UTC, and a
  forecast system that has only ever been run at 00 UTC has an untested
  assumption in it.

Rollout lengths separate the fixed cost from the marginal one. One step is
almost entirely startup; forty steps is almost entirely rollout. Four is what a
user actually asks for.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# The archive on this box: 2026-04-01T00 .. 2026-06-30T18, 6-hourly.
ARCHIVE_START = dt.datetime(2026, 4, 1, 0)
ARCHIVE_END = dt.datetime(2026, 6, 30, 18)


@dataclass(frozen=True)
class Init:
    time: str
    tag: str
    why: str


INITS: tuple[Init, ...] = (
    Init("2026-04-05T00", "interior", "ordinary init, nothing special"),
    Init("2026-05-01T00", "boundary-spring", "init - 6 h is in the previous month"),
    Init("2026-06-01T00", "boundary-summer", "same boundary, different season"),
    Init("2026-06-15T12", "midday", "12 UTC rather than 00 UTC"),
)

# Short rollouts, run across every init. 40 steps is handled separately: it
# costs six minutes in fp32 and only two inits are needed to show that the
# precision result is not a property of one date.
SHORT_STEPS: tuple[int, ...] = (1, 4)
LONG_STEPS: int = 40
LONG_INITS: tuple[str, ...] = ("2026-05-01T00", "2026-06-01T00")

# The variants under test. "fp32" is the reference every other number is
# compared against; it is not a candidate.
VARIANTS: tuple[str, ...] = ("fp32", "fp16", "fp16+compile")

# How many forecasts to ask for at once. The box has four V100s, so 4 is the
# point where every GPU is busy and 8 is the point where they are oversubscribed
# and requests start queueing behind each other.
CONCURRENCY: tuple[int, ...] = (1, 2, 4, 8)


def variant_env(variant: str) -> dict[str, str]:
    """Environment for a variant, so the runner never open-codes the mapping."""
    return {
        "fp32": {"AURORA_AUTOCAST": "off", "AURORA_COMPILE": "0"},
        "fp16": {"AURORA_AUTOCAST": "fp16", "AURORA_COMPILE": "0"},
        "fp16+compile": {"AURORA_AUTOCAST": "fp16", "AURORA_COMPILE": "1"},
    }[variant]


# Read shapes, phrased as the question a user is really asking. The point of
# the set is the spread between them: this store is chunked one (timestamp,
# level) map per chunk, so a query that wants one number still pays for every
# chunk that number lives in.
READ_QUERIES: tuple[tuple[str, str], ...] = (
    ("one map", "2t over the whole globe at one moment"),
    ("one day", "2t over the whole globe, 4 moments"),
    ("one month", "2t over the whole globe, 120 moments"),
    ("whole archive", "2t over the whole globe, 364 moments"),
    ("one point, whole archive", "2t at a single grid point, 364 moments"),
    ("one level, one month", "t at 500 hPa, 120 moments"),
    ("all levels, one moment", "t on all 13 levels, one moment"),
    ("full input", "all 69 channels at one moment — what a forecast reads"),
)


def has_truth(init: str, steps: int) -> bool:
    """Whether the archive can score a rollout of this length from this init."""
    end = dt.datetime.fromisoformat(init) + dt.timedelta(hours=6 * steps)
    return end <= ARCHIVE_END
