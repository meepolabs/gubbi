# Contributing to journalctl

Thanks for your interest. Before submitting a PR:

1. Read [CLA.md](./CLA.md). Submitting a PR signals your agreement to
   its terms.
2. Set up the dev environment:
   ```
   poetry install
   poetry run pre-commit install
   ```
3. Add tests for new behaviour. Aim for 80%+ coverage on touched code.
4. Run all checks locally before pushing:
   ```
   poetry run pre-commit run --all-files
   poetry run pytest
   ```
5. Keep PRs small and focused. One logical change per PR.
6. Use conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
   `chore:`, `perf:`, `ci:`).

For larger changes, open an issue first to discuss the approach.

For deployment, security, and operational topics, see the docs in
[`docs/`](./docs/).
