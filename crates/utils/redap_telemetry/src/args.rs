#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogFormat {
    Pretty,
    Compact,
    Json,
}

impl std::fmt::Display for LogFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Pretty => "pretty",
            Self::Compact => "compact",
            Self::Json => "json",
        })
    }
}

impl std::str::FromStr for LogFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "pretty" => Self::Pretty,
            "compact" => Self::Compact,
            "json" => Self::Json,
            unknown => anyhow::bail!("unknown LogFormat: '{unknown}"),
        })
    }
}

// ---

const fn default_telemetry_attributes() -> &'static str {
    concat!(
        "service.namespace=redap,service.version=",
        env!("CARGO_PKG_VERSION")
    )
}

const fn default_log_filter() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "info"
    }
}

/// Complete configuration for all things telemetry.
///
/// Many of these are part of the official `OpenTelemetry` spec and can be configured directly via
/// the environment. Refer to this command's help as well as [the spec].
///
/// [the spec]: https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
#[derive(Clone, Debug, clap::Parser)]
#[clap(author, version, about)]
pub struct TelemetryArgs {
    /// Should telemetry be disabled entirely?
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(long, env = "OTEL_SDK_DISABLED", default_value_t = false)]
    pub disabled: bool,

    /// The service name used for all things telemetry.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(long, env = "OTEL_SERVICE_NAME")]
    pub service_name: String,

    /// The service attributes used for all things telemetry.
    ///
    /// Expects a comma-separated string of key=value pairs, e.g. `a=b,c=d`.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(
        long,
        env = "OTEL_RESOURCE_ATTRIBUTES",
        default_value = default_telemetry_attributes(),
    )]
    pub attributes: String,

    /// This is the same as `RUST_LOG`.
    ///
    /// This only affects logs, not traces nor metrics.
    #[clap(long, env = "RUST_LOG", default_value_t = default_log_filter().to_owned())]
    pub log_filter: String,

    /// Capture test output as part of the logs.
    #[clap(long, env = "RUST_LOG_CAPTURE_TEST_OUTPUT", default_value_t = false)]
    pub log_test_output: bool,

    /// Use `json` in production. Pick between `pretty` and `compact` during development according
    /// to your preferences.
    #[clap(long, env = "RUST_LOG_FORMAT", default_value_t = LogFormat::Pretty)]
    pub log_format: LogFormat,

    /// If true, log extra information about all retired spans, including their timings.
    #[clap(long, env = "RUST_LOG_CLOSED_SPANS", default_value_t = false)]
    pub log_closed_spans: bool,

    /// Same as `RUST_LOG`, but for traces.
    ///
    /// This only affects traces, not logs nor metrics.
    #[clap(long, env = "RUST_TRACE", default_value = "info")]
    pub trace_filter: String,

    /// The gRPC OTLP endpoint to send the traces to.
    ///
    /// It's fine for the target endpoint to be down.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(
        long,
        env = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        default_value = "http://localhost:4317"
    )]
    pub trace_endpoint: String,

    /// How are spans sampled?
    ///
    /// This is applied _after_ `RUST_TRACE`.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(
        long,
        env = "OTEL_TRACES_SAMPLER",
        default_value = "parentbased_traceidratio"
    )]
    pub trace_sampler: String,

    /// The specified value will only be used if `OTEL_TRACES_SAMPLER` is set.
    ///
    /// Each Sampler type defines its own expected input, if any. Invalid or unrecognized input
    /// MUST be logged and MUST be otherwise ignored, i.e. the implementation MUST behave as if
    /// `OTEL_TRACES_SAMPLER_ARG` is not set.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(long, env = "OTEL_TRACES_SAMPLER_ARG", default_value = "1.0")]
    pub trace_sampler_args: String,

    /// The HTTP OTLP endpoint to send the metrics to.
    ///
    /// It's fine for the target endpoint to be down.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(
        long,
        env = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        default_value = "http://localhost:9090/api/v1/otlp/v1/metrics"
    )]
    pub metric_endpoint: String,

    /// The interval in milliseconds at which metrics are pushed to the collector.
    ///
    /// Part of the `OpenTelemetry` spec.
    #[clap(long, env = "OTEL_METRIC_EXPORT_INTERVAL", default_value = "10000")]
    pub metric_interval: String,
}
