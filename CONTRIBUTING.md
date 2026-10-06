# Contributing to gubbi

Thanks for your interest. Before submitting a PR:

1. Read [CLA.md](./CLA.md). Submitting a PR signals your agreement to
   its terms.
2. Set up the dev environment:
   ```
   poetry install
   poetry run pre-commit install
   ```
   `pre-commit install` installs two hooks: the pre-commit hook runs the
   lint, format, type and secret checks; the pre-push hook runs
   `make prepush`, every database-backed suite against a local test stack.
3. Start the local test stack (two Postgres containers in docker, private to
   this checkout; no host Postgres or psql needed):
   ```
   make test-stack-up        # start it; writes .testdb.env
   make test-stack-status    # show each container's state
   make test-stack-down      # remove it
   ```
   With the stack down, `git push` stops at once with
   `test stack not running: make test-stack-up`. Each `make prepush` resets
   the databases first, so the stack can stay up between pushes. After
   `tools/testdb/testdb.env` changes, run `make test-stack-down` then
   `make test-stack-up`.
4. Add tests for new behaviour. Aim for 80%+ coverage on touched code.
5. Run all checks locally before pushing:
   ```
   poetry run pre-commit run --all-files
   poetry run pytest
   make prepush
   ```
6. Keep PRs small and focused. One logical change per PR.
7. Use conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
   `chore:`, `perf:`, `ci:`).

For larger changes, open an issue first to discuss the approach.

For deployment, security, and operational topics, see the docs in
[`docs/`](./docs/).
