"""
Download a file over HTTPS, with a certificate pin as an explicit fallback.

Normally the website's certificate is checked the standard way. The GroupLens
download server's certificate chain ends at a root whose key is malformed, so
standard checking can't succeed. In that case (and only then) we accept the
connection only if the server's certificate is EXACTLY the one recorded in
config.DATASET_TLS_PIN_SHA256. Anything else stops with an error.

After downloading, the file's SHA-256 fingerprint is compared with
config.DATASET_ZIP_SHA256 (when set), so every later download must be
byte-for-byte identical to the first one.
"""

from __future__ import annotations

import hashlib
import http.client
import ssl
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .errors import DataValidationError

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _connect(host: str, port: int, ctx: ssl.SSLContext) -> http.client.HTTPSConnection:
    proxy = urllib.request.getproxies().get("https")
    if proxy and not urllib.request.proxy_bypass(host):
        p = urlparse(proxy if "://" in proxy else "http://" + proxy)
        conn = http.client.HTTPSConnection(p.hostname, p.port or 80, context=ctx, timeout=120)
        conn.set_tunnel(host, port)
    else:
        conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=120)
    conn.connect()
    return conn


def open_connection(host: str, port: int, pin_sha256: str | None, log=print) -> http.client.HTTPSConnection:
    """Standard certificate check first; the pin is used only if that fails."""
    try:
        return _connect(host, port, ssl.create_default_context())
    except ssl.SSLCertVerificationError as exc:
        if not pin_sha256:
            raise DataValidationError(f"Could not verify {host}'s certificate: {exc}") from exc
        log(f"Standard certificate check failed for {host} ({exc.verify_message}); checking the pinned certificate instead.")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # replaced by the exact-match pin check below
    conn = _connect(host, port, ctx)
    der = conn.sock.getpeercert(binary_form=True)
    actual = hashlib.sha256(der).hexdigest()
    if actual != pin_sha256.replace(":", "").lower():
        conn.close()
        raise DataValidationError(
            f"{host} presented a certificate that doesn't match the pin in config.py "
            f"(DATASET_TLS_PIN_SHA256).\n  expected {pin_sha256}\n  got      {actual}\n"
            "Not downloading. If the site has simply renewed its certificate, see README "
            "'Downloading the data' for how to check and update the pin."
        )
    return conn


def download(url: str, dest: Path, pin_sha256: str | None, expected_sha256: str | None, log=print) -> None:
    u = urlparse(url)
    if u.scheme != "https":
        raise DataValidationError(f"Refusing to download over {u.scheme}: {url}")
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        conn = open_connection(u.hostname, u.port or 443, pin_sha256, log)
        try:
            conn.request("GET", u.path or "/", headers={"User-Agent": "curator-build"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise DataValidationError(f"Download failed: {url} returned HTTP {resp.status} {resp.reason}")
            expected_len = resp.getheader("Content-Length")
            with open(part, "wb") as out:
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
        finally:
            conn.close()
        if expected_len is not None and part.stat().st_size != int(expected_len):
            raise DataValidationError(
                f"Download incomplete: got {part.stat().st_size:,} bytes, expected {int(expected_len):,}"
            )
    except DataValidationError:
        part.unlink(missing_ok=True)
        raise
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise DataValidationError(f"Download failed from {url}: {exc}") from exc

    actual = sha256_file(part)
    if expected_sha256 and actual != expected_sha256.lower():
        part.unlink(missing_ok=True)
        raise DataValidationError(
            f"Downloaded file's fingerprint doesn't match DATASET_ZIP_SHA256 in config.py.\n"
            f"  expected {expected_sha256}\n  got      {actual}\nThe file was deleted; nothing was built."
        )
    part.rename(dest)
    log(f"Saved {dest} (SHA-256 {actual})")
