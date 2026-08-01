"""Data this box does not hold, fetched from the centres that do.

The brief asks for two things this package answers: weather back to the 1990s,
which will never fit in 300 GB, and a three-hourly operational feed, which is
not on this disk at all until somebody goes and gets it. Three modules, and the
split between the first two is the single most important decision here:

    arco.py       deep-history *maps*   -> ARCO ERA5 on Google Cloud
    openmeteo.py  deep-history *points* -> Open-Meteo's archive API
    gfs.py        the operational feed  -> NOAA GFS on AWS, GRIB2

**Route by the shape of the query, not by the age of the data.** This is the
same lesson the local stores already taught, applied across the network. ARCO is
chunked `[1, 721, 1440]` — one chunk is one global map — so it is exactly the
right shape for `/v1/map` and exactly the wrong shape for `/v1/point`. Measured
on this box, against the same 1990 moment:

    one global map from ARCO ................... 1.30 s
    one point, one hour, from ARCO ............. 1.18 s   <- the same cost
    30 years hourly at one point, from ARCO .... 1.18 s x 262 968 = 86 days
    30 years hourly at one point, Open-Meteo ... 1.7 s

The middle line is the whole argument. A point query against a map-major store
pays for the entire map to learn one number, and over a network that is the
×1 038 240 read amplification the local two-layout split was built to avoid,
except now it is also somebody else's egress bill. So points go to a service
that is itself point-shaped, and maps go to the store whose chunk *is* a map.

**Neither source is on our contract, and both lie differently.** ARCO names its
fields `2m_temperature` and counts hours from 1900. Open-Meteo answers in °C and
hPa, and — the trap that cost a probe to find — its *default* archive is
ERA5-Land at 0.1°, not the ERA5 0.25° this backend is built on. Asking for
55.75 N without pinning the model returns 55.711773 N and a temperature 1.2 K
away from the one the archive would give. That is not a rounding difference,
it is a different product. Every adapter here therefore ends in
`contracts.validate` before its numbers are allowed out.
"""

from .arco import ArcoMaps
from .openmeteo import OpenMeteoSeries

__all__ = ["ArcoMaps", "OpenMeteoSeries"]
