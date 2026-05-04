"""Tests for OAuth SQLite storage."""

import sqlite3
import time
from pathlib import Path

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

from gubbi.oauth.storage import OAuthStorage


def _make_client(client_id: str = "test-client") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret="test-secret",
        redirect_uris=["http://localhost/callback"],
    )


def _make_auth_code(
    code: str = "test-code",
    client_id: str = "test-client",
) -> AuthorizationCode:
    return AuthorizationCode(
        code=code,
        scopes=["read"],
        expires_at=time.time() + 300,
        client_id=client_id,
        code_challenge="test-challenge",
        redirect_uri="http://localhost/callback",
        redirect_uri_provided_explicitly=True,
    )


def _make_access_token(
    token: str = "access-123",
    client_id: str = "test-client",
    expires_at: int | None = None,
) -> AccessToken:
    return AccessToken(
        token=token,
        client_id=client_id,
        scopes=["read"],
        expires_at=expires_at or int(time.time()) + 3600,
    )


def _make_refresh_token(
    token: str = "refresh-123",
    client_id: str = "test-client",
) -> RefreshToken:
    return RefreshToken(
        token=token,
        client_id=client_id,
        scopes=["read"],
        expires_at=int(time.time()) + 86400,
    )


class TestClientStorage:
    def test_save_and_get_client(self, oauth_storage: OAuthStorage) -> None:
        client = _make_client()
        oauth_storage.save_client(client)
        result = oauth_storage.get_client("test-client")
        assert result is not None
        assert result.client_id == "test-client"
        assert result.client_secret == "test-secret"

    def test_get_nonexistent_client(self, oauth_storage: OAuthStorage) -> None:
        assert oauth_storage.get_client("nonexistent") is None

    def test_overwrite_client(self, oauth_storage: OAuthStorage) -> None:
        client1 = _make_client()
        oauth_storage.save_client(client1)

        client2 = OAuthClientInformationFull(
            client_id="test-client",
            client_secret="new-secret",
            redirect_uris=["http://localhost/new"],
        )
        oauth_storage.save_client(client2)

        result = oauth_storage.get_client("test-client")
        assert result is not None
        assert result.client_secret == "new-secret"


class TestAuthCodeStorage:
    def test_save_and_get_auth_code(self, oauth_storage: OAuthStorage) -> None:
        code = _make_auth_code()
        oauth_storage.save_auth_code("test-code", code)
        result = oauth_storage.get_auth_code("test-code")
        assert result is not None
        assert result.code == "test-code"
        assert result.client_id == "test-client"

    def test_get_nonexistent_code(self, oauth_storage: OAuthStorage) -> None:
        assert oauth_storage.get_auth_code("nonexistent") is None

    def test_delete_auth_code(self, oauth_storage: OAuthStorage) -> None:
        code = _make_auth_code()
        oauth_storage.save_auth_code("test-code", code)
        oauth_storage.delete_auth_code("test-code")
        assert oauth_storage.get_auth_code("test-code") is None


class TestAccessTokenStorage:
    def test_save_and_get_access_token(self, oauth_storage: OAuthStorage) -> None:
        token = _make_access_token()
        oauth_storage.save_access_token("access-123", token)
        result = oauth_storage.get_access_token("access-123")
        assert result is not None
        assert result.token == "access-123"
        assert result.client_id == "test-client"

    def test_get_nonexistent_token(self, oauth_storage: OAuthStorage) -> None:
        assert oauth_storage.get_access_token("nonexistent") is None

    def test_delete_access_token(self, oauth_storage: OAuthStorage) -> None:
        token = _make_access_token()
        oauth_storage.save_access_token("access-123", token)
        oauth_storage.delete_access_token("access-123")
        assert oauth_storage.get_access_token("access-123") is None


class TestRefreshTokenStorage:
    def test_save_and_get_refresh_token(self, oauth_storage: OAuthStorage) -> None:
        token = _make_refresh_token()
        oauth_storage.save_refresh_token("refresh-123", token)
        result = oauth_storage.get_refresh_token("refresh-123")
        assert result is not None
        assert result.token == "refresh-123"

    def test_get_nonexistent_token(self, oauth_storage: OAuthStorage) -> None:
        assert oauth_storage.get_refresh_token("nonexistent") is None

    def test_delete_refresh_token(self, oauth_storage: OAuthStorage) -> None:
        token = _make_refresh_token()
        oauth_storage.save_refresh_token("refresh-123", token)
        oauth_storage.delete_refresh_token("refresh-123")
        assert oauth_storage.get_refresh_token("refresh-123") is None


class TestTokenPairs:
    def test_save_and_get_paired_refresh(self, oauth_storage: OAuthStorage) -> None:
        oauth_storage.save_token_pair("at-1", "rt-1")
        assert oauth_storage.get_paired_refresh_token("at-1") == "rt-1"

    def test_get_paired_access_tokens(self, oauth_storage: OAuthStorage) -> None:
        oauth_storage.save_token_pair("at-1", "rt-1")
        oauth_storage.save_token_pair("at-2", "rt-1")
        result = oauth_storage.get_paired_access_tokens("rt-1")
        assert set(result) == {"at-1", "at-2"}

    def test_no_pair_returns_none(self, oauth_storage: OAuthStorage) -> None:
        assert oauth_storage.get_paired_refresh_token("nonexistent") is None

    def test_delete_pair_by_access(self, oauth_storage: OAuthStorage) -> None:
        oauth_storage.save_token_pair("at-1", "rt-1")
        oauth_storage.delete_token_pair_by_access("at-1")
        assert oauth_storage.get_paired_refresh_token("at-1") is None

    def test_delete_pair_by_refresh(self, oauth_storage: OAuthStorage) -> None:
        oauth_storage.save_token_pair("at-1", "rt-1")
        oauth_storage.save_token_pair("at-2", "rt-1")
        oauth_storage.delete_token_pair_by_refresh("rt-1")
        assert oauth_storage.get_paired_access_tokens("rt-1") == []


class TestCleanupExpired:
    def test_cleanup_expired_tokens(self, oauth_storage: OAuthStorage) -> None:
        # Save an expired access token
        expired_at = _make_access_token("expired-at", expires_at=int(time.time()) - 10)
        oauth_storage.save_access_token("expired-at", expired_at)

        # Save a valid access token
        valid_at = _make_access_token("valid-at")
        oauth_storage.save_access_token("valid-at", valid_at)

        deleted = oauth_storage.cleanup_expired()
        assert deleted >= 1
        assert oauth_storage.get_access_token("expired-at") is None
        assert oauth_storage.get_access_token("valid-at") is not None

    def test_cleanup_cascades_expired_refresh_to_access(self, oauth_storage: OAuthStorage) -> None:
        """Fix #3: expired refresh tokens should cascade-delete their paired access tokens."""
        # Create a valid access token paired with an expired refresh token
        at = _make_access_token("orphan-at", expires_at=int(time.time()) + 3600)
        oauth_storage.save_access_token("orphan-at", at)

        rt = RefreshToken(
            token="expired-rt",
            client_id="test-client",
            scopes=["read"],
            expires_at=int(time.time()) - 10,  # expired
        )
        oauth_storage.save_refresh_token("expired-rt", rt)
        oauth_storage.save_token_pair("orphan-at", "expired-rt")

        deleted = oauth_storage.cleanup_expired()
        assert deleted >= 2  # refresh token + cascaded access token
        assert oauth_storage.get_access_token("orphan-at") is None
        assert oauth_storage.get_refresh_token("expired-rt") is None

    def test_cleanup_expired_auth_codes(self, oauth_storage: OAuthStorage) -> None:
        expired_code = _make_auth_code("expired-code")
        # Override expires_at to be in the past
        expired_code_data = expired_code.model_copy(
            update={"expires_at": time.time() - 10},
        )
        oauth_storage.save_auth_code("expired-code", expired_code_data)

        deleted = oauth_storage.cleanup_expired()
        assert deleted >= 1
        assert oauth_storage.get_auth_code("expired-code") is None


class TestExpiresAtColumn:
    def test_save_populates_expires_at(self, oauth_storage: OAuthStorage) -> None:
        token = _make_access_token(expires_at=int(time.time()) + 3600)
        oauth_storage.save_access_token("t1", token)
        row = oauth_storage.conn.execute(
            "SELECT expires_at FROM access_tokens WHERE token = ?",
            ("t1",),
        ).fetchone()
        assert row["expires_at"] is not None

    def test_cleanup_uses_indexed_column(self, oauth_storage: OAuthStorage) -> None:
        """EXPLAIN QUERY PLAN should use the expires_at index."""
        plan = oauth_storage.conn.execute(
            "EXPLAIN QUERY PLAN DELETE FROM access_tokens "
            "WHERE expires_at IS NOT NULL AND expires_at < 100",
        ).fetchall()
        plan_txt = " ".join(str(row[-1]) for row in plan)
        assert "idx_access_tokens_expires_at" in plan_txt or "USING INDEX" in plan_txt


class TestExpiresAtBackfill:
    def test_backfill_populates_null_expires_at(self, tmp_path: Path) -> None:
        # Create DB with old schema — mimic pre-upgrade state
        db_path = tmp_path / "legacy.db"
        raw = sqlite3.connect(str(db_path))
        raw.executescript("""
            CREATE TABLE access_tokens (
                token TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            INSERT INTO access_tokens (token, data, created_at) VALUES
                ('legacy-at',
                 '{"token":"legacy-at","expires_at":9999999999,"client_id":"x","scopes":[]}',
                 0);
        """)
        raw.commit()
        raw.close()

        storage = OAuthStorage(db_path)
        _ = storage.conn  # triggers migration + backfill

        row = storage.conn.execute(
            "SELECT expires_at FROM access_tokens WHERE token = 'legacy-at'"
        ).fetchone()
        assert row["expires_at"] == 9999999999
        storage.close()


class TestRateLimitEvents:
    def test_record_and_count(self, oauth_storage: OAuthStorage) -> None:
        for _ in range(3):
            oauth_storage.record_rate_limit_event("test_key")
        assert oauth_storage.count_rate_limit_events("test_key", 60) == 3

    def test_count_respects_window(self, oauth_storage: OAuthStorage) -> None:
        # Insert an old row directly
        oauth_storage.conn.execute(
            "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
            ("old_key", int(time.time()) - 3600),
        )
        oauth_storage.conn.commit()
        oauth_storage.record_rate_limit_event("old_key")  # recent
        assert oauth_storage.count_rate_limit_events("old_key", 60) == 1
        assert oauth_storage.count_rate_limit_events("old_key", 7200) == 2

    def test_prune_removes_old_events(self, oauth_storage: OAuthStorage) -> None:
        oauth_storage.conn.execute(
            "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
            ("k", int(time.time()) - 10_000),
        )
        oauth_storage.conn.commit()
        oauth_storage.record_rate_limit_event("k")
        deleted = oauth_storage.prune_rate_limit_events(3600)
        assert deleted == 1
        assert oauth_storage.count_rate_limit_events("k", 60) == 1


class TestRotateRefreshToken:
    def test_rotate_is_atomic(self, oauth_storage: OAuthStorage) -> None:
        # Seed old pair
        old_at = _make_access_token("old-at", expires_at=int(time.time()) + 3600)
        old_rt = _make_refresh_token("old-rt")
        oauth_storage.save_access_token("old-at", old_at)
        oauth_storage.save_refresh_token("old-rt", old_rt)
        oauth_storage.save_token_pair("old-at", "old-rt")

        new_at = _make_access_token("new-at", expires_at=int(time.time()) + 3600)
        new_rt = _make_refresh_token("new-rt")

        oauth_storage.rotate_refresh_token(
            old_refresh_token_str="old-rt",
            new_access_token_str="new-at",
            new_access_token=new_at,
            new_refresh_token_str="new-rt",
            new_refresh_token=new_rt,
        )

        # Old gone
        assert oauth_storage.get_access_token("old-at") is None
        assert oauth_storage.get_refresh_token("old-rt") is None
        assert oauth_storage.get_paired_refresh_token("old-at") is None
        # New present and paired
        assert oauth_storage.get_access_token("new-at") is not None
        assert oauth_storage.get_refresh_token("new-rt") is not None
        assert oauth_storage.get_paired_refresh_token("new-at") == "new-rt"
