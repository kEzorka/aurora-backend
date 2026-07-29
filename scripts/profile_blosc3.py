"""profile_step, but with zarr 3 told to keep Blosc lz4-5 shuffle.

The migration target is neither store on disk: zarr 3 writing through the codec
we already chose. Patched here rather than edited into postprocess.py, because
no migration is authorised yet.
"""
import sys

import zarr

from app import postprocess

_orig = postprocess._zarr_encoding
BLOSC = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")


def patched(ds):
    enc = _orig(ds)
    for spec in enc.values():
        spec["compressors"] = [BLOSC]
    return enc


postprocess._zarr_encoding = patched

from scripts.profile_step import main  # noqa: E402

sys.argv = ["profile_blosc3", "2026-04-05T00", "--steps", "4"]
raise SystemExit(main())
