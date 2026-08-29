"""Tolerate a corrupt Windows certificate store when building SSL contexts.

On this machine the Windows ROOT store contains a malformed entry, so
ssl.create_default_context() raises:

    ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data

aiohttp builds its default SSL contexts at import time and does not guard
that call (aiohttp/connector.py: _SSL_CONTEXT_VERIFIED = _make_ssl_context(True)),
so merely importing aiohttp kills the process. MLflow's server imports aiohttp
via mlflow.assistant.providers.

apply() makes ssl.SSLContext.load_default_certs degrade gracefully:
  1. real Windows store loader, plus certifi's bundle when available;
  2. on SSLError, certifi's bundle alone;
  3. if certifi is missing too, no certs (last resort, keeps the server up).

Steps 1-2 keep certificate verification working. Only step 3 leaves a context
without trust anchors; outbound TLS from the server would then fail to verify
rather than silently accept, since verify_mode is untouched.
"""

import ssl

_applied = False


def _certifi_path():
    try:
        import certifi

        return certifi.where()
    except Exception:
        return None


def apply():
    """Patch ssl.SSLContext.load_default_certs. Idempotent."""
    global _applied
    if _applied:
        return
    _applied = True

    original = ssl.SSLContext.load_default_certs
    cafile = _certifi_path()

    def load_default_certs(self, *args, **kwargs):
        try:
            original(self, *args, **kwargs)
            if cafile is not None:
                self.load_verify_locations(cafile=cafile)
            return
        except ssl.SSLError:
            pass

        if cafile is not None:
            self.load_verify_locations(cafile=cafile)
            return

        try:
            original(self, *args, **kwargs)
        except ssl.SSLError:
            pass

    ssl.SSLContext.load_default_certs = load_default_certs
