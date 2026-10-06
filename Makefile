# gubbi Makefile: the local test stack and the pre-push suite.
#
#   make test-stack-up                    start this checkout's two Postgres clusters
#   make test-stack-down                  remove them
#   make test-stack-status [REQUIRE_READY=1]
#   make prepush                          every database-backed suite (needs the stack)

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Fixed recipe text; `override` keeps a command-line assignment out of it.
# PROFILE is the one tools/run_db_suites.py resets. SUITE_ENV holds the test
# values security-tests.yml sets for the same suites: reproducible, not secret.
override TESTDB := tools/testdb/testdb.py
override PROFILE := --repo gubbi --role pg --role pg-disposable --db journal_test --db journal_rls_test
override SUITE_ENV := JOURNAL_OPERATOR_EMAIL=operator@test.local JOURNAL_ENCRYPTION_MASTER_KEY_V1=AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=

.PHONY: help
help:
	@sed -n '3,6p' Makefile | sed 's/^# \{0,1\}//'

.PHONY: test-stack-up
test-stack-up:
	@python3 $(TESTDB) up $(PROFILE)

.PHONY: test-stack-down
test-stack-down:
	@python3 $(TESTDB) down $(PROFILE)

.PHONY: test-stack-status
test-stack-status:
	@python3 $(TESTDB) status $(PROFILE) $(if $(filter 1,$(REQUIRE_READY)),--require-ready)

# On a stack miss the status check prints exactly
# `test stack not running: make test-stack-up` and nothing else runs. It is
# called inline rather than through a sub-make so make adds only one
# `*** Error` line after it.
.PHONY: prepush
prepush:
	@python3 $(TESTDB) status $(PROFILE) --require-ready
	@env $(SUITE_ENV) python3 tools/run_db_suites.py
