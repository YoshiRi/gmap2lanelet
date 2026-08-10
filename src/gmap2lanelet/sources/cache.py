"""Tiny on-disk cache for downloaded public data."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache"


def cache_dir() -> Path:
    d = DEFAULT_CACHE
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_path(url: str, suffix: str = "") -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    stem = (Path(url.split("?")[0]).name or "download")[:80]
    return cache_dir() / f"{h}_{stem}{suffix}"


def fetch_bytes(url: str, *, timeout: int = 120, retries: int = 4,
                headers: dict | None = None, use_cache: bool = True) -> bytes:
    """GET with on-disk caching and exponential backoff."""
    p = cached_path(url)
    if use_cache and p.exists() and p.stat().st_size > 0:
        return p.read_bytes()

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            r.raise_for_status()
            if use_cache:
                p.write_bytes(r.content)
            return r.content
        except Exception as exc:                       # noqa: BLE001 - retried below
            last = exc
            wait = 2 ** (attempt + 1)
            log.warning("fetch failed (%s/%s) %s: %s; retrying in %ss",
                        attempt + 1, retries, url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {url}: {last}")


def post_form(url: str, data: dict, *, timeout: int = 180, retries: int = 4,
              headers: dict | None = None, use_cache: bool = True) -> bytes:
    """POST a form (Overpass) with caching keyed on url+body."""
    key = url + "|" + repr(sorted(data.items()))
    p = cached_path(key, ".cache")
    if use_cache and p.exists() and p.stat().st_size > 0:
        return p.read_bytes()

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(url, data=data, timeout=timeout, headers=headers or {})
            r.raise_for_status()
            if use_cache:
                p.write_bytes(r.content)
            return r.content
        except Exception as exc:                       # noqa: BLE001
            last = exc
            wait = 2 ** (attempt + 1)
            log.warning("post failed (%s/%s) %s: %s; retrying in %ss",
                        attempt + 1, retries, url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not post {url}: {last}")
