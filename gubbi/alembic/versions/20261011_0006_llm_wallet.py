"""Add the per-user LLM wallet: append-only credit grants and charges.

``credit_grants`` -- one row per credit given to a user. ``amount`` is in
weighted tokens, always positive and at most 1000000000, a sanity bound on any
single grant. ``idempotency_key`` is unique, so a grant source that retries
(``onboarding:<user id>``, a payment id) inserts at most once.

``llm_charges`` -- one row per worker attempt that used the model, from the
usage the provider reported: raw input, cached input and output tokens, the
weighted total, and how it split between the monthly allowance and credit.
``month`` is the first day of the UTC month the attempt ended in. A CHECK
keeps both parts non-negative and summing to ``weighted``; ``(job_id,
attempt)`` is unique, so an attempt is charged at most once. ``job_id`` has no
foreign key: a charge may belong to a job kind with no ``extraction_jobs`` row.
``job_kind`` has no CHECK; the application's spend-rule table names the kinds.
``user_id`` is nullable with ON DELETE SET NULL: deleting a user keeps their
charges as anonymous rows, so a purge never lowers the deployment-wide monthly
total. A grant has no use without its user, so ``credit_grants`` cascades.

A user's balance is a sum over these rows, never stored. Two indexes answer
the sums from the index alone: ``(user_id, month) INCLUDE (monthly_part,
credit_part)`` for a user's monthly use and credit spent, ``(month) INCLUDE
(weighted)`` for the deployment-wide monthly total. ``(user_id) INCLUDE
(amount)`` does the same for grants.

Both tables carry FORCE ROW LEVEL SECURITY with a user-isolation policy for
journal_app matching ``extraction_jobs``; an anonymous charge matches no user,
so journal_app neither sees nor writes one. journal_app may only SELECT and
INSERT: the tables are append-only. journal_admin holds ALL, as on every
other table.

A RESTRICTIVE INSERT policy narrows what journal_app may grant to the
onboarding shape: reason ``onboarding``, key ``onboarding:<user id>`` and an
amount of at most 250000000. With the unique key, journal_app can write at
most one grant per user, so a fault on an app path cannot mint credit beyond
it. Every other grant kind (paid top-ups, admin comps) is written through
journal_admin, which bypasses RLS.

Every user present at upgrade is granted 25000000 weighted tokens of
onboarding credit under the key ``onboarding:<user id>``, the same key later
onboarding grants use, so no user is granted twice.

No statement uses IF [NOT] EXISTS: an object of the same name already present
means the schema drifted from this chain, so the upgrade fails and rolls back.
The downgrade drops both tables, which removes their indexes and policies.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_llm_wallet"
down_revision = "0005_extraction_queue_columns"
branch_labels = None
depends_on = None

_CREATE_CREDIT_GRANTS = """
CREATE TABLE public.credit_grants (
    id uuid DEFAULT gen_random_uuid() NOT NULL
        CONSTRAINT credit_grants_pkey PRIMARY KEY,
    user_id uuid NOT NULL
        CONSTRAINT credit_grants_user_id_fkey
        REFERENCES public.users(id) ON DELETE CASCADE,
    amount bigint NOT NULL
        CONSTRAINT credit_grants_amount_check CHECK (amount > 0 AND amount <= 1000000000),
    reason text NOT NULL,
    idempotency_key text NOT NULL
        CONSTRAINT credit_grants_idempotency_key_key UNIQUE,
    created_at timestamp with time zone DEFAULT now() NOT NULL
)
"""

_CREATE_LLM_CHARGES = """
CREATE TABLE public.llm_charges (
    id uuid DEFAULT gen_random_uuid() NOT NULL
        CONSTRAINT llm_charges_pkey PRIMARY KEY,
    user_id uuid
        CONSTRAINT llm_charges_user_id_fkey
        REFERENCES public.users(id) ON DELETE SET NULL,
    job_kind text NOT NULL,
    job_id uuid NOT NULL,
    attempt integer NOT NULL,
    input_tokens bigint NOT NULL,
    cached_input_tokens bigint NOT NULL,
    output_tokens bigint NOT NULL,
    weighted bigint NOT NULL,
    monthly_part bigint NOT NULL,
    credit_part bigint NOT NULL,
    month date NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT llm_charges_parts_check CHECK (
        monthly_part >= 0 AND credit_part >= 0 AND monthly_part + credit_part = weighted
    ),
    CONSTRAINT llm_charges_job_attempt_key UNIQUE (job_id, attempt)
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX idx_credit_grants_user "
    "ON public.credit_grants USING btree (user_id) INCLUDE (amount)",
    "CREATE INDEX idx_llm_charges_user_month "
    "ON public.llm_charges USING btree (user_id, month) INCLUDE (monthly_part, credit_part)",
    "CREATE INDEX idx_llm_charges_month "
    "ON public.llm_charges USING btree (month) INCLUDE (weighted)",
)

_TABLES = ("credit_grants", "llm_charges")

_USER_ISOLATION_POLICY = """
CREATE POLICY {table}_user_isolation ON public.{table} TO journal_app
    USING (user_id = (SELECT public.app_current_user_active()))
    WITH CHECK (user_id = (SELECT public.app_current_user_active()))
"""

_APP_INSERT_ONBOARDING_ONLY_POLICY = """
CREATE POLICY credit_grants_app_insert_onboarding_only ON public.credit_grants
    AS RESTRICTIVE FOR INSERT TO journal_app
    WITH CHECK (
        reason = 'onboarding'
        AND idempotency_key = 'onboarding:' || user_id::text
        AND amount <= 250000000
    )
"""

_SEED_ONBOARDING_CREDIT = """
INSERT INTO public.credit_grants (user_id, amount, reason, idempotency_key)
SELECT id, 25000000, 'onboarding', 'onboarding:' || id::text
FROM public.users
WHERE deleted_at IS NULL
ON CONFLICT (idempotency_key) DO NOTHING
"""


def _secure(table: str) -> list[str]:
    """RLS, the user-isolation policy and the append-only grants for ``table``."""
    return [
        f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY",
        _USER_ISOLATION_POLICY.format(table=table),
        f"REVOKE ALL ON TABLE public.{table} FROM journal_app",
        f"GRANT SELECT, INSERT ON TABLE public.{table} TO journal_app",
        f"GRANT ALL ON TABLE public.{table} TO journal_admin",
    ]


def upgrade() -> None:
    """Create both tables, indexes, RLS, policies and grants; seed onboarding credit."""
    op.execute(_CREATE_CREDIT_GRANTS)
    op.execute(_CREATE_LLM_CHARGES)
    for statement in _CREATE_INDEXES:
        op.execute(statement)
    for table in _TABLES:
        for statement in _secure(table):
            op.execute(statement)
    op.execute(_APP_INSERT_ONBOARDING_ONLY_POLICY)
    op.execute(_SEED_ONBOARDING_CREDIT)


def downgrade() -> None:
    """Drop both tables with everything created on them."""
    op.execute("DROP TABLE public.llm_charges")
    op.execute("DROP TABLE public.credit_grants")
