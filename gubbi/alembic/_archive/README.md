# gubbi alembic chain archive

This directory holds the original 0001-0031 migration chain that was
squashed into `versions/20260524_0001_squashed_baseline.py` on
2026-05-23. The archive exists so:

1. The historical chain stays inspectable for postmortems / forensics.
2. A dev DB stamped at any old revision can be migrated forward
   without round-tripping through a fresh-baseline reset.

Alembic does NOT load this directory. Only files under `versions/` are
in the active chain.

## Squash provenance

- Original gubbi head at squash time: `0031_audit_log_dedup_actor_scope`
- Original gubbi-cloud head at squash time: `0020_revoke_journal_app_delete_billing_identity`
- Squash source: `pg_dump --schema-only` of a fully-migrated testbench
  DB. Round-trip into a fresh DB produced an empty manifest diff,
  proving mechanical equivalence with the chain.
- Capture artifacts (pg_dump output + manifest fingerprints):
  see the squash-source artifact bundle dated 2026-05-23.

## Resurrecting for dev DB forward migration

If you have a dev DB stamped at e.g. `0017_add_platform_id_to_conversations`
and want to bring it to the new baseline:

1. Move the desired old chain file(s) back into `versions/` temporarily.
2. `alembic upgrade head` to walk the chain to `0031`.
3. Manually update `alembic_version` to `0001_squashed_baseline`:
   ```sql
   UPDATE alembic_version SET version_num = '0001_squashed_baseline';
   ```
4. Move the old chain files back to `_archive/`.

Step 3 is safe because the squashed baseline is mechanically equivalent
to the chain head. After step 3 the DB is indistinguishable from a
fresh baseline install at the same head.

## Schema source coupling

The original `0001_baseline.py` reads `gubbi/storage/schema.sql`. That
file is still present in the repo (not moved or deleted) so the
archived chain remains runnable. Future work that drops `schema.sql`
must first drop this archive.
