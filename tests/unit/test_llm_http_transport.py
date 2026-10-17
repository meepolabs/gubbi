"""TLS settings of the transport LLM clients use, against a hostile environment."""

from __future__ import annotations

import pathlib
import ssl

import pytest

from gubbi.extraction.llm.http_transport import pinned_transport


def _transport_context() -> ssl.SSLContext:
    context = pinned_transport()._pool._ssl_context
    assert isinstance(context, ssl.SSLContext)
    return context


def test_tls_context_does_not_log_session_secrets_to_the_environment_keylog_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    keylog = tmp_path / "keylog.txt"
    monkeypatch.setenv("SSLKEYLOGFILE", str(keylog))
    assert ssl.create_default_context().keylog_filename == str(keylog)
    keylog.unlink()

    context = _transport_context()

    assert context.keylog_filename is None
    assert not keylog.exists()


def test_tls_context_verifies_certificates_and_hostnames_on_tls_1_2_or_later() -> None:
    context = _transport_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
