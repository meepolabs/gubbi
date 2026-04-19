"""Application-layer content encryption via AES-256-GCM.

Used to encrypt sensitive tenant content (entries.content, entries.reasoning,
messages.content) at the repository layer. See TASK-02.11 in
llm_context/tasks/milestone-02-multitenant-auth.md for the spec and
threat model.

Versioning is carried in the first byte of the 12-byte nonce so key
rotation does not require a separate column. Byte 0 = key version
(1..255), bytes 1..11 = CSPRNG random. At 2^88 nonce-bytes per version,
collision probability is cryptographically negligible.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import secrets
from collections.abc import Mapping
from types import MappingProxyType

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_ENV_PATTERN = re.compile(r"^JOURNAL_ENCRYPTION_MASTER_KEY_V(\d+)$")
_KEY_LEN = 32
_NONCE_LEN = 12
_VERSION_MIN = 1
_VERSION_MAX = 255


class ContentCipher:
    """AES-256-GCM symmetric cipher with key-version-in-nonce.

    Construct with ``{version: 32-byte key}`` (at least one version, each
    version in [1, 255]). ``encrypt`` uses the highest-numbered version;
    ``decrypt`` selects the key by the version byte baked into the nonce
    at ``encrypt`` time.
    """

    def __init__(self, master_keys: Mapping[int, bytes]) -> None:
        if not master_keys:
            raise ValueError("master_keys must contain at least one version")
        for version, key in master_keys.items():
            if not isinstance(version, int) or isinstance(version, bool):
                raise TypeError(f"key version must be int, got {type(version).__name__}")
            if not _VERSION_MIN <= version <= _VERSION_MAX:
                raise ValueError(
                    f"key version {version} out of range " f"[{_VERSION_MIN}, {_VERSION_MAX}]"
                )
            if not isinstance(key, bytes | bytearray):
                raise TypeError(
                    f"key for version {version} must be bytes, " f"got {type(key).__name__}"
                )
            if len(key) != _KEY_LEN:
                raise ValueError(
                    f"key for version {version} must be {_KEY_LEN} bytes, " f"got {len(key)}"
                )
        frozen_keys: dict[int, bytes] = {v: bytes(k) for v, k in master_keys.items()}
        self._keys: Mapping[int, bytes] = MappingProxyType(frozen_keys)
        # AESGCM is the expensive constructor (key schedule); build one per
        # version up-front so encrypt/decrypt are hot-path cheap.
        self._ciphers: Mapping[int, AESGCM] = MappingProxyType(
            {version: AESGCM(key) for version, key in frozen_keys.items()}
        )
        self._active_version: int = max(frozen_keys)

    def __repr__(self) -> str:
        # Explicit redaction: make sure key material never leaks into logs
        # even if someone passes the cipher to structlog.bind() directly.
        return (
            f"<ContentCipher active_version={self._active_version} "
            f"known_versions={sorted(self._keys)}>"
        )

    @property
    def active_version(self) -> int:
        return self._active_version

    @property
    def known_versions(self) -> frozenset[int]:
        return frozenset(self._keys)

    def encrypt(self, plaintext: str) -> tuple[bytes, bytes]:
        """Encrypt ``plaintext`` with the active key version.

        Returns ``(ciphertext, nonce)``. ``ciphertext`` includes the GCM
        auth tag. ``nonce`` is ``bytes([active_version]) + secrets.token_bytes(11)``.
        """
        nonce = bytes([self._active_version]) + secrets.token_bytes(_NONCE_LEN - 1)
        ciphertext = self._ciphers[self._active_version].encrypt(
            nonce, plaintext.encode("utf-8"), None
        )
        return (ciphertext, nonce)

    def decrypt(self, ciphertext: bytes, nonce: bytes) -> str:
        """Decrypt the ``(ciphertext, nonce)`` pair produced by ``encrypt``.

        Accepts ``bytes`` or ``bytearray`` for either argument. Raises
        ``ValueError`` for a malformed nonce or unknown version, and
        ``cryptography.exceptions.InvalidTag`` when the GCM auth tag does
        not verify (tampered ciphertext, wrong key, truncation).

        NOTE for callers: ``ValueError`` vs ``InvalidTag`` MUST NOT be
        distinguished in any response surfaced to end users. Raising
        different HTTP status codes or log verbosity for the two types
        creates a version-existence oracle. Repository layer (TASK-02.13)
        wraps both in a single opaque error.
        """
        if not isinstance(nonce, bytes | bytearray) or len(nonce) != _NONCE_LEN:
            raise ValueError(f"nonce must be {_NONCE_LEN} bytes")
        version = nonce[0]
        aesgcm = self._ciphers.get(version)
        if aesgcm is None:
            raise ValueError(f"unknown key version {version}")
        plaintext_bytes: bytes = aesgcm.decrypt(bytes(nonce), bytes(ciphertext), None)
        return plaintext_bytes.decode("utf-8")


def load_master_keys_from_env(
    environ: Mapping[str, str] | None = None,
) -> dict[int, bytes]:
    """Scan env for ``JOURNAL_ENCRYPTION_MASTER_KEY_V<N>``, base64-decode each.

    Returns ``{N: key_bytes}`` with every discovered version. Raises
    ``ValueError`` on bad base64, wrong key length, or out-of-range
    version. Returns an empty dict when no matching vars are set --
    caller decides whether to treat that as fatal (prod) or log-and-skip
    (dev before Track B wiring).
    """
    env = os.environ if environ is None else environ
    keys: dict[int, bytes] = {}
    for name, value in env.items():
        match = _KEY_ENV_PATTERN.match(name)
        if match is None:
            continue
        raw_version = match.group(1)
        # Reject leading zeros so V01 and V1 can never alias onto the same
        # internal key. A version of literally "0" falls through to the
        # range check below and fails there with a clearer message.
        if len(raw_version) > 1 and raw_version[0] == "0":
            raise ValueError(f"{name}: version must not have leading zeros")
        version = int(raw_version)
        if not _VERSION_MIN <= version <= _VERSION_MAX:
            raise ValueError(
                f"{name}: version {version} out of range " f"[{_VERSION_MIN}, {_VERSION_MAX}]"
            )
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{name}: invalid base64") from exc
        if len(raw) != _KEY_LEN:
            raise ValueError(f"{name}: decoded key must be {_KEY_LEN} bytes, got {len(raw)}")
        keys[version] = raw
    return keys
