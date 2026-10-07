"""Add an earlier season of Sentinel-1 data to an existing hindcast (same-season velocity baselines).

    python scripts/extend_hindcast_baseline.py rasuwa-lhende rasuwa-2026 2025-05-01 2025-10-05

Afterwards run `himsat reassess <aoi> --hindcast <name>` so assessments use the new baselines.
"""

import sys

from himsat.config import get_settings
from himsat.pipeline.hindcast import hindcast_dir
from himsat.pipeline.monitor import CycleOptions, run_cycle
from himsat.util import parse_date, setup_logging


def main(aoi: str, name: str, start: str, end: str) -> None:
    setup_logging("INFO")
    d = hindcast_dir(name)
    settings = get_settings().model_copy(update={"dispatch_enabled": False})
    opts = CycleOptions(start=parse_date(start), end=parse_date(end), dispatch=False, sensors=("S1",),
                        hindcast=True, products_dir=d / "products", progress=print)
    res = run_cycle(aoi, opts, db_url=f"sqlite:///{(d / 'himsat.db').as_posix()}", settings=settings)
    print("done", res.stats)


if __name__ == "__main__":
    main(*sys.argv[1:5])
