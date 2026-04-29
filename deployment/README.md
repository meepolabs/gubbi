# Deployment

## OpenTelemetry (TASK-03.19)

journalctl uses OpenTelemetry for distributed tracing, metrics, and
structured logging. All instrumentation is behind the `OTEL_ENABLED`
feature flag.

### Env vars

| Variable | Default | Description |
|---|---|---|
| `OTEL_ENABLED` | `true` | Master switch. Set to `false` to disable OTel exports (NoOp providers are wired, so instrumentation calls are safe no-ops). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | gRPC endpoint of the OTel Collector relay. |
| `OTEL_EXPORTER_OTLP_HEADERS` | (empty) | Headers for OTLP export. Empty for journalctl -> Collector (internal network). |
| `OTEL_SERVICE_NAME` | `journalctl` | Service name reported in traces and metrics. |
| `OTEL_RESOURCE_ATTRIBUTES` | (empty) | Comma-separated `key=value` pairs, e.g. `env=prod,version=abc1234,region=hel1`. |

### Disable OTel in dev

```bash
export OTEL_ENABLED=false
```

When disabled, the OTel SDK is configured with NoOp tracer and meter
providers. Auto-instrumentation modules are still loaded but produce
no output. This protects startup if the Collector is not running.

### Collector deployment

The OTel Collector relay runs on a separate VPS (`obs`). The app exports
OTLP gRPC to the Collector, which relays to HyperDX. See the
`journalctl-cloud/deployment/otel-collector/` directory for Collector
configuration.
