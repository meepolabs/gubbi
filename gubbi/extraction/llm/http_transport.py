"""HTTP transport for LLM clients that takes nothing from the process environment.

httpx2 and the SDKs read proxies (``HTTP(S)_PROXY``, ``ALL_PROXY``) and CA
locations (``SSL_CERT_FILE``, ``SSL_CERT_DIR``) from the environment. LLM
requests carry conversation text and the deployment's key, so where they go
and which certificates vouch for that endpoint come from gubbi's settings and
the system trust store only.
"""

from __future__ import annotations

import os
import socket
import ssl

import httpx2

__all__: list[str] = [
    "pinned_transport",
]

# Matches the keepalive probing the SDKs configure on their own transports,
# so a long generation is not dropped by an idle-connection timeout upstream.
_KEEPALIVE_SECS = 60
_KEEPALIVE_PROBES = 5


def pinned_transport() -> httpx2.AsyncHTTPTransport:
    """A direct (proxy-less) transport trusting only the system CA store.

    Pass it as ``transport=`` together with ``trust_env=False``: the explicit
    transport keeps the SDK from mounting environment proxies, and
    ``trust_env=False`` keeps httpx2 from doing the same.
    """
    return httpx2.AsyncHTTPTransport(
        verify=_system_trust_context(),
        trust_env=False,
        socket_options=_keepalive_options(),
    )


def _system_trust_context() -> ssl.SSLContext:
    """A verifying TLS context over OpenSSL's compiled-in CA locations.

    Built directly rather than with ``ssl.create_default_context()``: that
    resolves CA locations through ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` and
    opens the file named by ``SSLKEYLOGFILE`` to log session secrets.
    """
    paths = ssl.get_default_verify_paths()
    cafile = paths.openssl_cafile if os.path.isfile(paths.openssl_cafile) else None
    capath = paths.openssl_capath if os.path.isdir(paths.openssl_capath) else None
    if cafile is None and capath is None:
        raise RuntimeError(
            "no system CA certificates found at "
            f"{paths.openssl_cafile} or {paths.openssl_capath}; install the CA bundle"
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_verify_locations(cafile=cafile, capath=capath)
    return context


def _keepalive_options() -> list[tuple[int, int, int]]:
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (
        ("TCP_KEEPIDLE", _KEEPALIVE_SECS),
        ("TCP_KEEPINTVL", _KEEPALIVE_SECS),
        ("TCP_KEEPCNT", _KEEPALIVE_PROBES),
    ):
        option = getattr(socket, name, None)
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))
    return options
