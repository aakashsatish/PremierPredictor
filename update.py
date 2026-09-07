#!/usr/bin/env python3
"""Weekly refresh: pull the latest results, rebuild, re-predict.

Run this to bring the predictions up to date. It re-fetches only the current
season (two requests, ~10 ScraperAPI credits); every earlier season is served
from the on-disk cache and costs nothing.
"""

import subprocess
import sys

import config
import scraper


def main() -> int:
    print("=" * 70)
    print(f"Refreshing {config.CURRENT_SEASON}")
    print("=" * 70)

    try:
        before = config.credits_remaining()["creditsLeft"]
        print(f"ScraperAPI credits available: {before}")
    except Exception as e:
        before = None
        print(f"Could not read credit balance ({e}); continuing.")

    print("\nFetching current-season results and fixtures ...")
    scraper.scrape_schedule(config.CURRENT_SEASON, force=True)
    try:
        scraper.fetch_football_data(config.CURRENT_SEASON, force=True)
    except RuntimeError as e:
        # football-data publishes on a lag; the schedule alone is enough to
        # keep fixtures and results current.
        print(f"  {e}\n  Continuing with fbref results only.")

    for step, script in [("Rebuilding dataset", "dataset.py"),
                         ("Generating predictions", "predict.py"),
                         ("Rendering the dashboard", "site.py")]:
        print(f"\n{step} ...")
        r = subprocess.run([sys.executable, script], cwd=config.ROOT)
        if r.returncode != 0:
            print(f"FAILED: {script} exited {r.returncode}")
            return r.returncode

    if before is not None:
        try:
            after = config.credits_remaining()["creditsLeft"]
            print(f"\nCredits used this run: {before - after} ({after} left)")
        except Exception:
            pass

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
