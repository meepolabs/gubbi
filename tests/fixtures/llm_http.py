"""Inputs for LLM provider tests about the HTTP layer: an environment CA and unmappable bodies."""

from __future__ import annotations

import datetime
import pathlib
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

ENV_CA_NAME = "Environment Test CA"

# Deep enough to exceed the JSON decoder's recursion limit.
_NESTING_DEPTH = 100_000


def write_env_ca(directory: pathlib.Path) -> pathlib.Path:
    """Write a self-signed CA certificate named ``ENV_CA_NAME``; return its PEM path."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ENV_CA_NAME)])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    path = directory / "env-ca.pem"
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return path


def ca_subjects(context: ssl.SSLContext) -> set[str]:
    """Common names of the CA certificates loaded into ``context``."""
    return {
        value
        for cert in context.get_ca_certs()
        for rdn in cert["subject"]
        for key, value in rdn
        if key == "commonName"
    }


def deeply_nested_json(sentinel: str) -> str:
    """A JSON document too deeply nested to decode, carrying ``sentinel``."""
    return "[" * _NESTING_DEPTH + f'"{sentinel}"' + "]" * _NESTING_DEPTH
