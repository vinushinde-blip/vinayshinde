"""
Resumable fetcher for 2 years of 1m/5m/15m NSE500 intraday candles via Kite Connect
historical data API.

Reads the Kite session (api_key + access_token) from a local session file that is
NEVER committed to the repo (kept outside the git working tree). Writes one parquet
file per (symbol, interval) under DATA_DIR, and maintains a checkpoint file so the
run can be interrupted and resumed without re-fetching completed (symbol, interval)
pairs.
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, timedelta

import aiohttp
import pandas as pd

SESSION_PATH = "/tmp/claude-0/-home-user-vinayshinde/b85090e9-2be6-5b7c-b09f-1f604bacd98c/scratchpad/kite_session/session.json"
REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
MAP_PATH = f"{REPO_ROOT}/data_cache/nse500_instrument_map.csv"
DATA_DIR = f"{REPO_ROOT}/data_cache/raw"
CHECKPOINT_PATH = f"{REPO_ROOT}/data_cache/fetch_checkpoint.json"
LOG_PATH = f"{REPO_ROOT}/data_cache/fetch_log.txt"

END_DATE = date(2026, 8, 17)
START_DATE = END_DATE - timedelta(days=730)

INTERVALS = {
    "minute": {"chunk_days": 60, "out_dir": "1minute"},
    "5minute": {"chunk_days": 100, "out_dir": "5minute"},
    "15minute": {"chunk_days": 200, "out_dir": "15minute"},
}

MAX_CONCURRENCY = 3
MIN_INTERVAL_SEC = 0.35  # ~2.85 req/sec globally, safely under Kite's 3 req/sec cap
MAX_RETRIES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fetch")


def date_chunks(start: date, end: date, chunk_days: int):
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(sorted(list(done)), f)


async def fetch_chunk(session, headers, limiter, token, interval, frm, to):
    url = f"https://api.kite.trade/instruments/historical/{token}/{interval}?from={frm}&to={to}"
    for attempt in range(1, MAX_RETRIES + 1):
        await limiter.wait()
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    return body.get("data", {}).get("candles", [])
                elif resp.status == 429:
                    wait = min(2 ** attempt, 30)
                    log.warning(f"429 rate-limited, backing off {wait}s (attempt {attempt})")
                    await asyncio.sleep(wait)
                elif resp.status in (403,):
                    text = await resp.text()
                    log.error(f"AUTH FAILURE (token expired?): {text}")
                    raise RuntimeError("access_token invalid/expired - need re-login")
                else:
                    text = await resp.text()
                    log.warning(f"HTTP {resp.status} for {url}: {text[:200]} (attempt {attempt})")
                    await asyncio.sleep(min(2 ** attempt, 20))
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(f"network error {e} for {url} (attempt {attempt})")
            await asyncio.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {url}")


async def fetch_symbol_interval(session, headers, limiter, symbol, token, interval, cfg, done, sem):
    key = (symbol, interval)
    if list(key) in [list(x) for x in done] or tuple(key) in done:
        return
    out_dir = f"{DATA_DIR}/{cfg['out_dir']}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{symbol}.parquet"

    async with sem:
        chunks = date_chunks(START_DATE, END_DATE, cfg["chunk_days"])
        all_candles = []
        try:
            for frm, to in chunks:
                candles = await fetch_chunk(session, headers, limiter, token, interval, frm.isoformat(), to.isoformat())
                all_candles.extend(candles)
        except RuntimeError as e:
            log.error(f"ABORT {symbol}/{interval}: {e}")
            raise

        if all_candles:
            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
            df["symbol"] = symbol
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
            df.to_parquet(out_path, index=False)
            log.info(f"OK {symbol}/{interval}: {len(df)} rows -> {out_path}")
        else:
            log.warning(f"EMPTY {symbol}/{interval}: no candles returned")

        done.add(key)
        save_checkpoint(done)


async def main():
    with open(SESSION_PATH) as f:
        sess = json.load(f)
    headers = {
        "X-Kite-Version": "3",
        "Authorization": f"token {sess['api_key']}:{sess['access_token']}",
    }

    inst = pd.read_csv(MAP_PATH)
    limit = int(os.environ.get("FETCH_LIMIT", "0"))
    if limit:
        inst = inst.head(limit)
    done = load_checkpoint()
    log.info(f"Starting fetch: {len(inst)} symbols x {len(INTERVALS)} intervals, {len(done)} already done")

    limiter = RateLimiter(MIN_INTERVAL_SEC)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        jobs = []
        for _, row in inst.iterrows():
            symbol = row["Symbol"]
            token = int(row["instrument_token"])
            for interval, cfg in INTERVALS.items():
                jobs.append((symbol, token, interval, cfg))

        total = len(jobs)
        tasks = [
            asyncio.ensure_future(fetch_symbol_interval(session, headers, limiter, s, t, iv, cfg, done, sem))
            for (s, t, iv, cfg) in jobs
        ]

        completed = 0
        errors = 0
        aborted = False
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            except RuntimeError as e:
                errors += 1
                log.error(f"task failed: {e}")
                if "access_token invalid" in str(e) and not aborted:
                    aborted = True
                    log.error("Stopping: access token expired, need re-authentication. Cancelling remaining tasks.")
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except asyncio.CancelledError:
                pass
            completed += 1
            if completed % 25 == 0:
                log.info(f"PROGRESS: {completed}/{total} symbol-intervals done ({errors} errors)")

        if aborted:
            await asyncio.gather(*tasks, return_exceptions=True)

    log.info(f"DONE. completed={completed}/{total} errors={errors}")


if __name__ == "__main__":
    asyncio.run(main())
