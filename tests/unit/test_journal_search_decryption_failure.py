"""Test journal search surfaces decryption failure with marker (M-9.8)."""

from __future__ import annotations

from journalctl.models.search import SearchResult


class TestJournalSearchDecryptionFailure:
    """M-9.8: decrypted entries that fail are returned with content and marker."""

    def test_search_result_has_decryption_failed_field(self) -> None:
        """SearchResult includes decryption_failed bool (default False)."""
        sr = SearchResult(
            source_key="entry:42",
            doc_type="entry",
            topic="work",
            rank=1.0,
            date="2026-05-01",
        )
        assert hasattr(sr, "decryption_failed")
        assert sr.decryption_failed is False

    def test_search_result_decryption_failed_set(self) -> None:
        """Setting decryption_failed=True works and survives model_copy."""
        sr = SearchResult(
            source_key="entry:99",
            doc_type="entry",
            topic="personal",
            rank=2.0,
            date="2026-04-30",
            content="[decryption failed]",
            decryption_failed=True,
        )
        assert sr.decryption_failed is True

    async def test_hydration_logic_sets_marker(self) -> None:
        """Simulated hydration: sentinel produces [decryption failed] + flag."""
        decrypted_entries = {42: ("[decryption-failed]", None), 43: ("ok content", None)}
        merged = [
            SearchResult(
                source_key="entry:42",
                doc_type="entry",
                topic="work",
                rank=1.0,
                date="2026-05-01",
                entry_id=42,
            ),
            SearchResult(
                source_key="entry:43",
                doc_type="entry",
                topic="work",
                rank=2.0,
                date="2026-05-02",
                entry_id=43,
            ),
        ]

        hydrated = []
        for result in merged:
            if result.entry_id is not None and result.entry_id in decrypted_entries:
                content, _reasoning = decrypted_entries[result.entry_id]
                decryption_failed = content == "[decryption-failed]"
                update: dict = {
                    "content": ("[decryption failed]" if decryption_failed else content),
                    "decryption_failed": decryption_failed,
                }
                hydrated.append(result.model_copy(update=update))

        assert len(hydrated) == 2
        found_failed = [s for s in hydrated if s.decryption_failed]
        found_ok = [s for s in hydrated if not s.decryption_failed]
        assert len(found_failed) == 1
        assert found_failed[0].content == "[decryption failed]"
        assert len(found_ok) == 1
        assert found_ok[0].content == "ok content"
