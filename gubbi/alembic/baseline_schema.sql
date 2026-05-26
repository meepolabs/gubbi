--
-- gubbi squashed baseline schema (revision 0001_squashed_baseline).
--
-- Mechanically derived 2026-05-23 from a pg_dump --schema-only of a
-- testbench DB at gubbi=0031 + gubbi-cloud=0020. Captures the gubbi-owned
-- subset: pgvector extension; audit_log functions; otel_ro role; the 8
-- gubbi tables (users, topics, conversations, entries, messages,
-- entry_embeddings, audit_log, extraction_jobs); their PKs / UKs / CHECK
-- constraints / indexes / triggers / FKs / RLS / policies / GRANTs;
-- schema public ACL; default privileges for the journal superuser.
--
-- Cloud-owned objects (tenants, subscriptions, llm_budgets, outbox_events,
-- stripe_events) live in the gubbi-cloud baseline and depend on this one
-- having run first (FK to users(id)).
--
-- Roles journal_app + journal_admin are pre-created with passwords by
-- testbench config/postgres/init.sh and prod gubbi-stack/postgres-init.sh.
-- This migration creates otel_ro: LOGIN, read-only monitoring role for the
-- OTel collector's postgresql receiver, pg_monitor-only (no data access).
-- The password is set out-of-band at deploy time (a LOGIN role with no
-- password cannot authenticate, so the role is inert until then).
--
-- The old chain (0001-0031) lives at _archive/ for dev-DB forward
-- migration; it is not loaded by alembic.

SET default_tablespace = '';
SET default_table_access_method = heap;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: otel_ro role; Type: ROLE; Schema: -; Owner: -
--
-- pg_dump --schema-only does not capture cluster-global roles, so we
-- recreate the role declaration from the original migration 0026 here.
-- LOGIN + pg_monitor is the read-only monitoring role for the OTel
-- collector's postgresql receiver; pg_monitor is the standard built-in
-- role for read-only observability scrapers (no data-table grants). The
-- password is set out-of-band at deploy time -- a LOGIN role with no
-- password cannot authenticate, so the role stays inert until then.
--

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otel_ro') THEN
        CREATE ROLE otel_ro LOGIN;
    END IF;
END $$;

GRANT pg_monitor TO otel_ro;


--
-- Name: audit_log_admin_no_user_actor(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.audit_log_admin_no_user_actor() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
        BEGIN
            IF current_user = 'journal_admin' AND NEW.actor_type = 'user' THEN
                RAISE EXCEPTION
                    'journal_admin cannot insert audit_log row with actor_type=user;'
                    ' use app_pool/user_scoped_connection or set actor_type to system/admin/hydra_subject'
                    USING ERRCODE = 'insufficient_privilege';
            END IF;
            RETURN NEW;
        END;
        $$;


--
-- Name: audit_log_immutable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.audit_log_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log rows are append-only';
        END;
        $$;


--
-- Name: audit_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.audit_log (
    id bigint NOT NULL,
    occurred_at timestamp with time zone DEFAULT now() NOT NULL,
    actor_type text NOT NULL,
    actor_id text NOT NULL,
    action text NOT NULL,
    target_type text,
    target_id text,
    reason text,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    ip_address inet,
    user_agent text,
    target_kind text,
    CONSTRAINT audit_log_actor_type_check CHECK ((actor_type = ANY (ARRAY['user'::text, 'admin'::text, 'system'::text, 'hydra_subject'::text]))),
    CONSTRAINT audit_log_target_kind_invariant CHECK (((target_id IS NULL) OR (target_kind IS NOT NULL)))
);

ALTER TABLE ONLY public.audit_log FORCE ROW LEVEL SECURITY;


--
-- Name: audit_log_id_seq1; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.audit_log ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.audit_log_id_seq1
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: conversations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.conversations (
    id integer NOT NULL,
    topic_id integer NOT NULL,
    slug text NOT NULL,
    source text DEFAULT 'claude'::text NOT NULL,
    tags text[] DEFAULT '{}'::text[] NOT NULL,
    participants text[] DEFAULT '{}'::text[] NOT NULL,
    message_count integer DEFAULT 0 NOT NULL,
    json_path text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    user_id uuid NOT NULL,
    search_vector tsvector,
    title_encrypted bytea NOT NULL,
    title_nonce bytea NOT NULL,
    summary_encrypted bytea NOT NULL,
    summary_nonce bytea NOT NULL,
    platform text,
    platform_id text,
    processed_at timestamp with time zone,
    CONSTRAINT conversations_summary_nonce_len CHECK (((summary_nonce IS NULL) OR (octet_length(summary_nonce) = 12))),
    CONSTRAINT conversations_title_nonce_len CHECK (((title_nonce IS NULL) OR (octet_length(title_nonce) = 12)))
);

ALTER TABLE ONLY public.conversations FORCE ROW LEVEL SECURITY;


--
-- Name: conversations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.conversations ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.conversations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: entries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entries (
    id integer NOT NULL,
    topic_id integer NOT NULL,
    date date DEFAULT CURRENT_DATE NOT NULL,
    conversation_id integer,
    tags text[] DEFAULT '{}'::text[] NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    deleted_at timestamp with time zone,
    indexed_at timestamp with time zone,
    user_id uuid NOT NULL,
    content_encrypted bytea NOT NULL,
    content_nonce bytea NOT NULL,
    reasoning_encrypted bytea,
    reasoning_nonce bytea,
    search_vector tsvector,
    CONSTRAINT entries_content_nonce_len CHECK (((content_nonce IS NULL) OR (octet_length(content_nonce) = 12))),
    CONSTRAINT entries_reasoning_nonce_len CHECK (((reasoning_nonce IS NULL) OR (octet_length(reasoning_nonce) = 12)))
);

ALTER TABLE ONLY public.entries FORCE ROW LEVEL SECURITY;


--
-- Name: entries_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.entries ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.entries_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: entry_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entry_embeddings (
    entry_id integer NOT NULL,
    embedding public.vector(384) NOT NULL,
    indexed_at timestamp with time zone DEFAULT now() NOT NULL,
    user_id uuid NOT NULL
);

ALTER TABLE ONLY public.entry_embeddings FORCE ROW LEVEL SECURITY;


--
-- Name: extraction_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.extraction_jobs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    conversation_id integer NOT NULL,
    source text NOT NULL,
    status text NOT NULL,
    topics_created integer DEFAULT 0 NOT NULL,
    entries_created integer DEFAULT 0 NOT NULL,
    cents_spent integer DEFAULT 0 NOT NULL,
    error_code text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    period_start date NOT NULL
);

ALTER TABLE ONLY public.extraction_jobs FORCE ROW LEVEL SECURITY;


--
-- Name: messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.messages (
    id integer NOT NULL,
    conversation_id integer NOT NULL,
    role text NOT NULL,
    "timestamp" timestamp with time zone,
    "position" integer DEFAULT 0 NOT NULL,
    user_id uuid NOT NULL,
    content_encrypted bytea NOT NULL,
    content_nonce bytea NOT NULL,
    search_vector tsvector,
    CONSTRAINT messages_content_nonce_len CHECK (((content_nonce IS NULL) OR (octet_length(content_nonce) = 12)))
);

ALTER TABLE ONLY public.messages FORCE ROW LEVEL SECURITY;


--
-- Name: messages_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.messages ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.messages_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: topics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.topics (
    id integer NOT NULL,
    path text NOT NULL,
    title text NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    user_id uuid NOT NULL
);

ALTER TABLE ONLY public.topics FORCE ROW LEVEL SECURITY;


--
-- Name: topics_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.topics ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.topics_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id uuid NOT NULL,
    email text NOT NULL,
    timezone text DEFAULT 'UTC'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    deleted_at timestamp with time zone,
    email_verified_at timestamp with time zone
);

ALTER TABLE ONLY public.users FORCE ROW LEVEL SECURITY;


--
-- Name: audit_log audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_log
    ADD CONSTRAINT audit_log_pkey PRIMARY KEY (id);


--
-- Name: conversations conversations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.conversations
    ADD CONSTRAINT conversations_pkey PRIMARY KEY (id);


--
-- Name: conversations conversations_topic_id_slug_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.conversations
    ADD CONSTRAINT conversations_topic_id_slug_key UNIQUE (topic_id, slug);


--
-- Name: entries entries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entries
    ADD CONSTRAINT entries_pkey PRIMARY KEY (id);


--
-- Name: entry_embeddings entry_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entry_embeddings
    ADD CONSTRAINT entry_embeddings_pkey PRIMARY KEY (entry_id);


--
-- Name: extraction_jobs extraction_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_jobs
    ADD CONSTRAINT extraction_jobs_pkey PRIMARY KEY (id);


--
-- Name: messages messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_pkey PRIMARY KEY (id);


--
-- Name: topics topics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics
    ADD CONSTRAINT topics_pkey PRIMARY KEY (id);


--
-- Name: topics topics_user_path_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics
    ADD CONSTRAINT topics_user_path_key UNIQUE (user_id, path);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: audit_log_content_hash_uidx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX audit_log_content_hash_uidx ON public.audit_log USING btree (actor_id, target_kind, target_id, action, ((metadata ->> 'content_hash'::text))) WHERE (metadata ? 'content_hash'::text);


--
-- Name: idx_audit_log_action; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_log_action ON public.audit_log USING btree (action);


--
-- Name: idx_audit_log_actor_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_log_actor_id ON public.audit_log USING btree (actor_id, occurred_at DESC);


--
-- Name: idx_audit_log_occurred_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_log_occurred_at ON public.audit_log USING btree (occurred_at DESC);


--
-- Name: idx_audit_log_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_log_target ON public.audit_log USING btree (target_type, target_id);


--
-- Name: idx_audit_log_target_kind_target_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_log_target_kind_target_id ON public.audit_log USING btree (target_kind, target_id);


--
-- Name: idx_conv_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_conv_created ON public.conversations USING btree (created_at DESC);


--
-- Name: idx_conv_fts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_conv_fts ON public.conversations USING gin (search_vector);


--
-- Name: idx_conv_platform_dedup; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_conv_platform_dedup ON public.conversations USING btree (user_id, platform, platform_id) WHERE (platform_id IS NOT NULL);


--
-- Name: idx_conv_topic; Type: INDEX; Schema: public; Owner: -
--
-- (idx_conv_slug WAS HERE on (topic_id, slug); dropped from the squashed
--  baseline because it duplicated the implicit btree from
--  conversations_topic_id_slug_key UNIQUE constraint.)

CREATE INDEX idx_conv_topic ON public.conversations USING btree (topic_id);


--
-- Name: idx_conv_user_topic; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_conv_user_topic ON public.conversations USING btree (user_id, topic_id);


--
-- Name: idx_embeddings_hnsw; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embeddings_hnsw ON public.entry_embeddings USING hnsw (embedding public.vector_cosine_ops) WITH (m='32', ef_construction='128');


--
-- Name: idx_embeddings_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embeddings_user ON public.entry_embeddings USING btree (user_id);


--
-- Name: idx_entries_conv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_conv ON public.entries USING btree (conversation_id);


--
-- Name: idx_entries_fts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_fts ON public.entries USING gin (search_vector);


--
-- Name: idx_entries_indexed_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_indexed_at ON public.entries USING btree (id) WHERE (indexed_at IS NULL);


--
-- Name: idx_entries_topic; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_topic ON public.entries USING btree (topic_id);


--
-- Name: idx_entries_topic_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_topic_date ON public.entries USING btree (topic_id, date DESC) WHERE (deleted_at IS NULL);


--
-- Name: idx_entries_user_indexed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_user_indexed ON public.entries USING btree (user_id, id) WHERE (indexed_at IS NULL);


--
-- Name: idx_entries_user_topic_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entries_user_topic_date ON public.entries USING btree (user_id, topic_id, date DESC) WHERE (deleted_at IS NULL);


--
-- Name: idx_extraction_jobs_active_per_conversation; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_extraction_jobs_active_per_conversation ON public.extraction_jobs USING btree (user_id, conversation_id, source) WHERE (status <> ALL (ARRAY['completed'::text, 'failed'::text]));


--
-- Name: idx_extraction_jobs_conv; Type: INDEX; Schema: public; Owner: -
--
-- Plain btree on conversation_id covers the FK cascade path (the partial
-- unique index above excludes terminal-state rows, so cascade deletes from
-- conversations would seq-scan completed/failed jobs without this index).

CREATE INDEX idx_extraction_jobs_conv ON public.extraction_jobs USING btree (conversation_id);


--
-- Name: idx_extraction_jobs_user_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_extraction_jobs_user_created ON public.extraction_jobs USING btree (user_id, created_at DESC);


--
-- Name: idx_messages_conv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_messages_conv ON public.messages USING btree (conversation_id, "position");


--
-- Name: idx_messages_fts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_messages_fts ON public.messages USING gin (search_vector);


--
-- Name: idx_messages_user_conv_pos; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_messages_user_conv_pos ON public.messages USING btree (user_id, conversation_id, "position");


--
-- Name: idx_topics_updated; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_topics_updated ON public.topics USING btree (updated_at DESC);


--
-- Name: idx_users_email_active; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_users_email_active ON public.users USING btree (email) WHERE (deleted_at IS NULL);


--
-- Name: audit_log trg_audit_log_admin_no_user_actor; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_audit_log_admin_no_user_actor BEFORE INSERT ON public.audit_log FOR EACH ROW EXECUTE FUNCTION public.audit_log_admin_no_user_actor();


--
-- Name: audit_log trg_audit_log_no_delete; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_audit_log_no_delete BEFORE DELETE ON public.audit_log FOR EACH ROW EXECUTE FUNCTION public.audit_log_immutable();


--
-- Name: audit_log trg_audit_log_no_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_audit_log_no_update BEFORE UPDATE ON public.audit_log FOR EACH ROW EXECUTE FUNCTION public.audit_log_immutable();


--
-- Name: conversations conversations_topic_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.conversations
    ADD CONSTRAINT conversations_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES public.topics(id) ON DELETE RESTRICT;


--
-- Name: conversations conversations_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.conversations
    ADD CONSTRAINT conversations_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: entries entries_conversation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entries
    ADD CONSTRAINT entries_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES public.conversations(id) ON DELETE RESTRICT;


--
-- Name: entries entries_topic_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entries
    ADD CONSTRAINT entries_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES public.topics(id) ON DELETE RESTRICT;


--
-- Name: entries entries_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entries
    ADD CONSTRAINT entries_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: entry_embeddings entry_embeddings_entry_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entry_embeddings
    ADD CONSTRAINT entry_embeddings_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.entries(id) ON DELETE CASCADE;


--
-- Name: entry_embeddings entry_embeddings_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entry_embeddings
    ADD CONSTRAINT entry_embeddings_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: extraction_jobs extraction_jobs_conversation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_jobs
    ADD CONSTRAINT extraction_jobs_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES public.conversations(id) ON DELETE CASCADE;


--
-- Name: extraction_jobs extraction_jobs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_jobs
    ADD CONSTRAINT extraction_jobs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: messages messages_conversation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES public.conversations(id) ON DELETE CASCADE;


--
-- Name: messages messages_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: topics topics_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics
    ADD CONSTRAINT topics_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: app_current_user_active(); Type: FUNCTION; Schema: public; Owner: -
--
-- Defense-in-depth helper for tenant-isolation RLS policies. Returns the
-- session-scoped user id (from the app.current_user_id GUC) only when the
-- corresponding users row is not soft-deleted; returns NULL otherwise.
--
-- RLS predicates compare user_id against this function instead of the GUC
-- directly so a soft-deleted user whose session somehow still carries the
-- GUC (e.g. Kratos cookie path that didn't check users.deleted_at server-
-- side) is denied at the schema layer. The auth layer SHOULD also reject;
-- this is the second line.
--
-- STABLE so the planner caches the result per query (single users-row PK
-- lookup amortized across the whole RLS scan).
--
-- NOT used by users_self_read / users_self_update -- those policies guard
-- deleted_at directly to avoid a function-into-table-into-policy recursion
-- (the function reads from users, which is RLS-checked by users_self_read).
--
-- Defined here (after users table + FKs are created, before RLS policies
-- that use it) so the function body's reference to public.users resolves
-- at CREATE FUNCTION time.

CREATE FUNCTION public.app_current_user_active() RETURNS uuid
    LANGUAGE sql
    STABLE
    AS $$
        SELECT u.id
        FROM public.users u
        WHERE u.id = NULLIF(current_setting('app.current_user_id'::text, true), ''::text)::uuid
          AND u.deleted_at IS NULL
        LIMIT 1
    $$;


--
-- Name: audit_log; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;

--
-- Name: audit_log audit_log_app_insert_self_only; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY audit_log_app_insert_self_only ON public.audit_log FOR INSERT TO journal_app WITH CHECK (((actor_id = ( SELECT NULLIF(current_setting('app.current_user_id'::text, true), ''::text))) AND (actor_id <> ''::text) AND (actor_type = 'user'::text)));


--
-- Name: conversations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;

--
-- Name: entries; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.entries ENABLE ROW LEVEL SECURITY;

--
-- Name: entry_embeddings; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.entry_embeddings ENABLE ROW LEVEL SECURITY;

--
-- Name: extraction_jobs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.extraction_jobs ENABLE ROW LEVEL SECURITY;

--
-- Name: extraction_jobs extraction_jobs_user_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY extraction_jobs_user_isolation ON public.extraction_jobs TO journal_app USING ((user_id = ( SELECT public.app_current_user_active()))) WITH CHECK ((user_id = ( SELECT public.app_current_user_active())));


COMMENT ON POLICY extraction_jobs_user_isolation ON public.extraction_jobs IS 'Default-deny user isolation for extraction_jobs. Compares user_id against app_current_user_active() so soft-deleted users (whose GUC may still be set if the auth layer missed deleted_at) see zero rows. BYPASSRLS roles (journal_admin) skip this policy entirely and see every row.';


--
-- Name: messages; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

--
-- Name: conversations tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.conversations TO journal_app USING ((user_id = ( SELECT public.app_current_user_active()))) WITH CHECK ((user_id = ( SELECT public.app_current_user_active())));


COMMENT ON POLICY tenant_isolation ON public.conversations IS 'Default-deny tenant isolation. Compares user_id against app_current_user_active() so soft-deleted users (whose GUC may still be set if the auth layer missed deleted_at) see zero rows. BYPASSRLS roles (journal_admin) skip this policy entirely and see every row.';


--
-- Name: entries tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.entries TO journal_app USING ((user_id = ( SELECT public.app_current_user_active()))) WITH CHECK ((user_id = ( SELECT public.app_current_user_active())));


COMMENT ON POLICY tenant_isolation ON public.entries IS 'Default-deny tenant isolation. Compares user_id against app_current_user_active() so soft-deleted users (whose GUC may still be set if the auth layer missed deleted_at) see zero rows. BYPASSRLS roles (journal_admin) skip this policy entirely and see every row.';


--
-- Name: entry_embeddings tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.entry_embeddings TO journal_app USING ((user_id = ( SELECT public.app_current_user_active()))) WITH CHECK ((user_id = ( SELECT public.app_current_user_active())));


COMMENT ON POLICY tenant_isolation ON public.entry_embeddings IS 'Default-deny tenant isolation. Compares user_id against app_current_user_active() so soft-deleted users (whose GUC may still be set if the auth layer missed deleted_at) see zero rows. BYPASSRLS roles (journal_admin) skip this policy entirely and see every row.';


--
-- Name: messages tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.messages TO journal_app USING ((user_id = ( SELECT public.app_current_user_active()))) WITH CHECK ((user_id = ( SELECT public.app_current_user_active())));


COMMENT ON POLICY tenant_isolation ON public.messages IS 'Default-deny tenant isolation. Compares user_id against app_current_user_active() so soft-deleted users (whose GUC may still be set if the auth layer missed deleted_at) see zero rows. BYPASSRLS roles (journal_admin) skip this policy entirely and see every row.';


--
-- Name: topics tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.topics TO journal_app USING ((user_id = ( SELECT public.app_current_user_active()))) WITH CHECK ((user_id = ( SELECT public.app_current_user_active())));


COMMENT ON POLICY tenant_isolation ON public.topics IS 'Default-deny tenant isolation. Compares user_id against app_current_user_active() so soft-deleted users (whose GUC may still be set if the auth layer missed deleted_at) see zero rows. BYPASSRLS roles (journal_admin) skip this policy entirely and see every row.';


--
-- Name: topics; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.topics ENABLE ROW LEVEL SECURITY;

--
-- Name: users; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

--
-- Name: users users_self_read; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY users_self_read ON public.users FOR SELECT TO journal_app USING (((id = ( SELECT (NULLIF(current_setting('app.current_user_id'::text, true), ''::text))::uuid AS "nullif")) AND (deleted_at IS NULL)));


--
-- Name: users users_self_update; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY users_self_update ON public.users FOR UPDATE TO journal_app USING (((id = ( SELECT (NULLIF(current_setting('app.current_user_id'::text, true), ''::text))::uuid AS "nullif")) AND (deleted_at IS NULL))) WITH CHECK (((id = ( SELECT (NULLIF(current_setting('app.current_user_id'::text, true), ''::text))::uuid AS "nullif")) AND (deleted_at IS NULL)));


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: -
--

GRANT ALL ON SCHEMA public TO journal_admin;
GRANT USAGE ON SCHEMA public TO journal_app;


--
-- Name: TABLE audit_log; Type: ACL; Schema: public; Owner: -
--

GRANT INSERT ON TABLE public.audit_log TO journal_app;
GRANT SELECT,INSERT ON TABLE public.audit_log TO journal_admin;


--
-- Name: SEQUENCE audit_log_id_seq1; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,USAGE ON SEQUENCE public.audit_log_id_seq1 TO journal_app;
GRANT ALL ON SEQUENCE public.audit_log_id_seq1 TO journal_admin;


--
-- Name: TABLE conversations; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.conversations TO journal_app;
GRANT ALL ON TABLE public.conversations TO journal_admin;


--
-- Name: SEQUENCE conversations_id_seq; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,USAGE ON SEQUENCE public.conversations_id_seq TO journal_app;
GRANT ALL ON SEQUENCE public.conversations_id_seq TO journal_admin;


--
-- Name: TABLE entries; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.entries TO journal_app;
GRANT ALL ON TABLE public.entries TO journal_admin;


--
-- Name: SEQUENCE entries_id_seq; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,USAGE ON SEQUENCE public.entries_id_seq TO journal_app;
GRANT ALL ON SEQUENCE public.entries_id_seq TO journal_admin;


--
-- Name: TABLE entry_embeddings; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.entry_embeddings TO journal_app;
GRANT ALL ON TABLE public.entry_embeddings TO journal_admin;


--
-- Name: TABLE extraction_jobs; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.extraction_jobs TO journal_app;
GRANT ALL ON TABLE public.extraction_jobs TO journal_admin;


--
-- Name: TABLE messages; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.messages TO journal_app;
GRANT ALL ON TABLE public.messages TO journal_admin;


--
-- Name: SEQUENCE messages_id_seq; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,USAGE ON SEQUENCE public.messages_id_seq TO journal_app;
GRANT ALL ON SEQUENCE public.messages_id_seq TO journal_admin;


--
-- Name: TABLE topics; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.topics TO journal_app;
GRANT ALL ON TABLE public.topics TO journal_admin;


--
-- Name: SEQUENCE topics_id_seq; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,USAGE ON SEQUENCE public.topics_id_seq TO journal_app;
GRANT ALL ON SEQUENCE public.topics_id_seq TO journal_admin;


--
-- Name: TABLE users; Type: ACL; Schema: public; Owner: -
--

GRANT SELECT,UPDATE ON TABLE public.users TO journal_app;
GRANT ALL ON TABLE public.users TO journal_admin;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public GRANT SELECT,USAGE ON SEQUENCES TO journal_app;
ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public GRANT ALL ON SEQUENCES TO journal_admin;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: -
--
-- Tightened from the captured dump: DELETE is dropped from journal_app's
-- default priv. Out of the 7 tables in the original chain that had to
-- narrow journal_app post-CREATE, 5 were DELETE-only narrowings; making
-- DELETE opt-in (via explicit GRANT in the migration that owns the table)
-- eliminates that class of footgun. journal_admin keeps ALL by default.

ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public GRANT SELECT,INSERT,UPDATE ON TABLES TO journal_app;
ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public GRANT ALL ON TABLES TO journal_admin;
