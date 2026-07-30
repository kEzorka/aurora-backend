"""Exercise the registry without a GPU, an archive or a model.

    python -m scripts.check_registry

Everything here is disk and SQLite, so it runs on a laptop. The three things
worth checking are the three the dict never did: a lookup by content, a total
that survives being reopened, and an eviction that frees the right forecasts.
"""

from __future__ import annotations

import datetime as dt
import tempfile
import uuid
from pathlib import Path

from app.registry import Job, Registry, store_size

GB = 1024**3


def fake_forecast(root: Path, name: str, mb: int) -> Path:
    """A directory of files, because that is what a zarr store is."""
    path = root / name
    path.mkdir(parents=True)
    for i in range(4):
        (path / f"chunk.{i}").write_bytes(b"\0" * (mb * 1024**2 // 4))
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reg = Registry(root / "registry.db")
        init = dt.datetime(2026, 5, 1)

        # --- a finished job is found by what it contains, not by its id
        path = fake_forecast(root, "f_24h", 40)
        job = reg.add(Job(id=uuid.uuid4().hex[:12], init_time=init, steps=4))
        reg.update(job.id, status="done", output=str(path), size_bytes=store_size(path))

        found = reg.find_ready(init, 4)
        assert found is not None and found.id == job.id, "identical request not matched"
        assert reg.find_ready(init, 8) is None, "different lead must not match"
        assert reg.find_ready(init + dt.timedelta(hours=6), 4) is None, "different init"
        print(f"dedup: {found.id} matched, {found.size_bytes / 1024**2:.0f} MB accounted")

        # --- the total survives a reopen, which is the whole point of a file
        reg.close()
        reg = Registry(root / "registry.db")
        assert reg.total_bytes() == store_size(path), "size lost across reopen"
        assert reg.get(job.id) is not None, "job lost across reopen"
        print(f"reopen: {reg.total_bytes() / 1024**2:.0f} MB still accounted for")

        # --- eviction takes the least recently read, not the oldest
        old_hot = reg.add(Job(id="hot", init_time=init, steps=40))
        cold = reg.add(Job(id="cold", init_time=init, steps=20))
        hot_path = fake_forecast(root, "f_hot", 40)
        cold_path = fake_forecast(root, "f_cold", 40)
        reg.update(old_hot.id, status="done", output=str(hot_path),
                   size_bytes=store_size(hot_path))
        reg.update(cold.id, status="done", output=str(cold_path),
                   size_bytes=store_size(cold_path))
        # `cold` was created last but read longest ago.
        reg.update(cold.id, last_access="2026-01-01T00:00:00")
        reg.touch(old_hot.id)

        cap = int(100 * 1024**2)   # three 40 MB stores are over this
        low = int(60 * 1024**2)
        removed = reg.evict(cap_bytes=cap, low_bytes=low)
        assert "cold" in removed, f"the stale forecast survived: {removed}"
        assert "hot" not in removed, "the forecast being read was deleted"
        assert not cold_path.exists(), "row cleared but files left behind"
        assert reg.total_bytes() <= low, "eviction stopped above the low mark"
        print(f"evicted {removed}, {reg.total_bytes() / 1024**2:.0f} MB left "
              f"(low mark {low / 1024**2:.0f} MB)")

        # --- a row whose store was deleted must not be handed out as ready
        assert reg.find_ready(init, 20) is None, "evicted forecast still advertised"
        # And it says so: a reclaimed forecast is not a failed one. Without a
        # state of its own the table cannot say how many forecasts the box has
        # produced, because every deletion looked like a crash.
        gone = reg.get("cold")
        assert gone is not None and gone.status == "evicted", f"state was {gone and gone.status}"
        assert reg.total_bytes() == store_size(hot_path), "evicted bytes still counted"
        print(f"evicted rows no longer answer lookups, and read {gone.status}, not failed")

        # --- precision is part of the identity, not a note about the run
        fp32 = reg.add(Job(id="fp32job", init_time=init, steps=6, precision="off"))
        fp32_path = fake_forecast(root, "f_fp32", 40)
        reg.update(fp32.id, status="done", output=str(fp32_path),
                   size_bytes=store_size(fp32_path))
        assert reg.find_ready(init, 6, "off") is not None, "fp32 run not matched by fp32 request"
        assert reg.find_ready(init, 6, "fp16") is None, "fp16 request handed the fp32 store"
        # The direction that used to be wrong in production: the service is
        # restarted in fp32 over a db full of fp16 stores, as bench/testset.py
        # does, and must roll out again instead of being handed the fp16 one.
        fp16 = reg.add(Job(id="fp16job", init_time=init, steps=7, precision="fp16"))
        fp16_path = fake_forecast(root, "f_fp16", 40)
        reg.update(fp16.id, status="done", output=str(fp16_path),
                   size_bytes=store_size(fp16_path))
        assert reg.find_ready(init, 7, "off") is None, "fp32 request handed the fp16 store"
        assert reg.find_ready(init, 7, "fp16").id == "fp16job", "fp16 lookup broken"
        # A miss must stay a miss: the row it did not match is still usable.
        assert reg.get("fp16job").status == "done", "a precision miss corrupted the row"
        print("precision: fp16 and fp32 runs of the same init+lead stay separate")

        # The matching check that the two also write to different paths is in
        # check_sse, not here: `postprocess` imports aurora and torch, and this
        # script is the one that runs on a laptop.

        # --- a pinned reference survives even when nobody has read it
        ref = reg.add(Job(id="fp32ref", init_time=init, steps=8))
        ref_path = fake_forecast(root, "f_ref", 40)
        reg.update(ref.id, status="done", output=str(ref_path),
                   size_bytes=store_size(ref_path))
        reg.pin(ref.id)
        reg.update(ref.id, last_access="2020-01-01T00:00:00")  # oldest of all
        filler = reg.add(Job(id="filler", init_time=init, steps=12))
        filler_path = fake_forecast(root, "f_filler", 40)
        reg.update(filler.id, status="done", output=str(filler_path),
                   size_bytes=store_size(filler_path))

        removed = reg.evict(cap_bytes=int(50 * 1024**2), low_bytes=int(10 * 1024**2))
        assert "fp32ref" not in removed, "pinned reference was evicted"
        assert ref_path.exists(), "pinned reference deleted from disk"
        assert "filler" in removed, f"eviction stopped early: {removed}"
        print(f"pinned reference survived eviction of {removed}; "
              f"{reg.total_bytes() / 1024**2:.0f} MB left, all of it pinned")

        # --- the real limits, so the numbers in the config are visible here
        from app import config
        print(f"configured cap {config.DISK_CAP_BYTES / GB:.0f} GB, "
              f"low {config.DISK_LOW_BYTES / GB:.0f} GB "
              f"(~{config.DISK_CAP_BYTES / GB / 7.3:.0f} ten-day runs at 7.3 GB)")
    print("ok")


if __name__ == "__main__":
    main()
