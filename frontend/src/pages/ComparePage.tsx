import React, { useState, useEffect, useCallback, useRef } from "react";
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Checkbox,
    CircularProgress,
    Divider,
    FormControl,
    FormControlLabel,
    FormGroup,
    FormLabel,
    Grid,
    InputLabel,
    MenuItem,
    Select,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Tab,
    Tabs,
    Tooltip,
    Typography,
} from "@mui/material";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartTooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";
import {
    fetchWorkloads,
    fetchEngines,
    fetchCapabilities,
    submitRunsForEngines,
    pollRun,
    type WorkloadInfo,
    type EngineInfo,
    type Capabilities,
    type RunStatus,
    type SchemaField,
} from "../api/client";
import { fmtMs, fmtPrice, fmtWorkload } from "../utils/format";
import MatrixTab from "../components/MatrixTab";

// ── Constants ─────────────────────────────────────────────────────────────

const ENGINE_COLOURS: Record<string, string> = {
    cpu: "#1565c0",
    jax: "#2e7d32",
    cpp: "#b71c1c",
    cuda: "#6a1b9a",
};

const POLL_INTERVAL_MS = 1500;
const MAX_POLLS = 120;

// ── Helpers ───────────────────────────────────────────────────────────────

function buildDefaults(schema: SchemaField[]): Record<string, number | string> {
    const out: Record<string, number | string> = {};
    for (const f of schema) out[f.key] = f.default;
    return out;
}

function analyticalBS(S0: number, K: number, r: number, sigma: number, T: number, type: string): number {
    const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    const nc = (x: number) => {
        const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        const sign = x < 0 ? -1 : 1;
        const t = 1 / (1 + p * Math.abs(x));
        const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
        return 0.5 * (1 + sign * (1 - poly * Math.exp(-Math.abs(x) * Math.abs(x) / 2)));
    };
    return type === "call"
        ? S0 * nc(d1) - K * Math.exp(-r * T) * nc(d2)
        : K * Math.exp(-r * T) * nc(-d2) - S0 * nc(-d1);
}

// ── CompareForm ───────────────────────────────────────────────────────────

interface CompareFormProps {
    workloads: Record<string, WorkloadInfo>;
    engines: Record<string, EngineInfo>;
    capabilities: Capabilities;
    loading: boolean;
    onSubmit: (
        workloadType: string,
        config: Record<string, number | string>,
        adMode: string,
        engines: string[]
    ) => void;
}

function CompareForm({ workloads, engines, capabilities, loading, onSubmit }: CompareFormProps) {
    const [workloadType, setWorkloadType] = useState("european");
    const [adMode, setAdMode] = useState("none");
    const [config, setConfig] = useState<Record<string, number | string>>({});
    const [selectedEngines, setSelectedEngines] = useState<Set<string>>(new Set(["cpu", "jax"]));
    const [error, setError] = useState<string | null>(null);

    const schema = workloads[workloadType]?.schema ?? [];
    const mainFields = schema.filter((f) => f.key !== "seed");

    useEffect(() => {
        setConfig(buildDefaults(schema));
    }, [workloadType, workloads]);

    // Engines that support the current workload + AD mode
    const eligibleEngines = Object.keys(engines).filter(
        (eng) =>
            engines[eng].supported_workloads.includes(workloadType) &&
            engines[eng].supported_ad_modes.includes(adMode)
    );

    // Remove any previously selected engines that are no longer eligible
    useEffect(() => {
        setSelectedEngines((prev) => {
            const next = new Set([...prev].filter((e) => eligibleEngines.includes(e)));
            if (next.size === 0 && eligibleEngines.length > 0) next.add(eligibleEngines[0]);
            return next;
        });
    }, [workloadType, adMode, engines]);

    // Collect all unique AD modes supported by eligible engines
    const availableAdModes = Array.from(
        new Set(eligibleEngines.flatMap((eng) => engines[eng]?.supported_ad_modes ?? []))
    );

    function toggleEngine(eng: string) {
        setSelectedEngines((prev) => {
            const next = new Set(prev);
            if (next.has(eng)) { next.delete(eng); } else { next.add(eng); }
            return next;
        });
    }

    function setField(key: string, val: number | string) {
        setConfig((prev) => ({ ...prev, [key]: val }));
    }

    function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        setError(null);
        if (selectedEngines.size < 2) {
            setError("Select at least 2 engines to compare.");
            return;
        }
        onSubmit(workloadType, config, adMode, [...selectedEngines]);
    }

    function renderField(f: SchemaField) {
        const val = config[f.key];
        if (f.type === "select") {
            return (
                <Grid item xs={12} sm={4} key={f.key}>
                    <FormControl size="small" fullWidth>
                        <InputLabel>{f.label}</InputLabel>
                        <Select
                            label={f.label}
                            value={String(val ?? f.default)}
                            onChange={(e) => setField(f.key, e.target.value)}
                        >
                            {(f.options ?? []).map((opt) => (
                                <MenuItem key={opt} value={opt}>
                                    {opt.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Grid>
            );
        }
        const isInt = f.type === "integer";
        return (
            <Grid item xs={6} sm={3} key={f.key}>
                <TextField
                    label={f.label}
                    type="number"
                    size="small"
                    fullWidth
                    value={val ?? f.default}
                    inputProps={{ step: isInt ? 1 : "any", min: f.min, max: f.max }}
                    onChange={(e) => {
                        const raw = isInt ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
                        if (!isNaN(raw)) setField(f.key, raw);
                    }}
                />
            </Grid>
        );
    }

    return (
        <Card variant="outlined">
            <CardContent>
                <form onSubmit={handleSubmit}>
                    <Grid container spacing={2}>
                        {/* Workload + AD Mode */}
                        <Grid item xs={12} sm={4}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Workload</InputLabel>
                                <Select
                                    label="Workload"
                                    value={workloadType}
                                    onChange={(e) => setWorkloadType(e.target.value)}
                                >
                                    {(Object.keys(workloads).length
                                        ? Object.keys(workloads)
                                        : ["european", "asian", "barrier", "basket"]
                                    ).map((w) => (
                                        <MenuItem key={w} value={w}>
                                            {workloads[w]?.label ?? fmtWorkload(w)}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>AD Mode</InputLabel>
                                <Select
                                    label="AD Mode"
                                    value={availableAdModes.includes(adMode) ? adMode : (availableAdModes[0] ?? "none")}
                                    onChange={(e) => setAdMode(e.target.value)}
                                >
                                    {availableAdModes.map((m) => (
                                        <MenuItem key={m} value={m}>
                                            {m === "none" ? "None" : m.charAt(0).toUpperCase() + m.slice(1)}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>

                        {/* Engine checkboxes */}
                        <Grid item xs={12}>
                            <FormControl component="fieldset">
                                <FormLabel component="legend" sx={{ fontSize: 13, mb: 0.5 }}>
                                    Engines to compare
                                </FormLabel>
                                <FormGroup row>
                                    {eligibleEngines.length > 0
                                        ? eligibleEngines.map((eng) => {
                                            const capKey = eng as keyof Capabilities;
                                            const unavailable =
                                                capKey in capabilities && !capabilities[capKey];
                                            return (
                                                <FormControlLabel
                                                    key={eng}
                                                    control={
                                                        <Checkbox
                                                            size="small"
                                                            checked={selectedEngines.has(eng)}
                                                            disabled={unavailable}
                                                            onChange={() => toggleEngine(eng)}
                                                            sx={{
                                                                color: ENGINE_COLOURS[eng] ?? undefined,
                                                                "&.Mui-checked": {
                                                                    color: ENGINE_COLOURS[eng] ?? undefined,
                                                                },
                                                            }}
                                                        />
                                                    }
                                                    label={
                                                        <Typography variant="body2">
                                                            {eng.toUpperCase()}
                                                            {unavailable ? " (unavailable)" : ""}
                                                        </Typography>
                                                    }
                                                />
                                            );
                                        })
                                        : <Typography variant="body2" color="text.secondary">
                                            No engines match the selected workload + AD mode.
                                        </Typography>
                                    }
                                </FormGroup>
                            </FormControl>
                        </Grid>

                        {/* Config fields */}
                        {mainFields.map(renderField)}

                        {error && (
                            <Grid item xs={12}>
                                <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>
                            </Grid>
                        )}

                        <Grid item xs={12}>
                            <Button
                                type="submit"
                                variant="contained"
                                size="large"
                                fullWidth
                                disabled={loading || selectedEngines.size < 2}
                                startIcon={
                                    loading
                                        ? <CircularProgress size={18} color="inherit" />
                                        : <CompareArrowsIcon />
                                }
                            >
                                {loading ? "Running…" : `Run Comparison (${selectedEngines.size} engines)`}
                            </Button>
                        </Grid>
                    </Grid>
                </form>
            </CardContent>
        </Card>
    );
}

// ── CompareResultsTable ───────────────────────────────────────────────────

interface EngineResult {
    engine: string;
    run: RunStatus | null;
    status: "pending" | "running" | "completed" | "failed" | "polling";
}

function CompareResultsTable({
    results,
    workloadType,
    config,
}: {
    results: EngineResult[];
    workloadType: string;
    config: Record<string, number | string>;
}) {
    const completed = results.filter((r) => r.run?.status === "completed");

    // Find fastest runtime for highlighting
    const runtimes = completed
        .map((r) => r.run?.mean_runtime_ms)
        .filter((v): v is number => v !== null && v !== undefined);
    const minRuntime = runtimes.length > 0 ? Math.min(...runtimes) : null;
    const maxRuntime = runtimes.length > 0 ? Math.max(...runtimes) : null;

    // CPU price used as baseline for relative error
    const cpuRun = completed.find((r) => r.engine === "cpu")?.run;
    const cpuPrice = cpuRun?.result_value ?? null;

    // Black-Scholes reference (European only)
    let bsPrice: number | null = null;
    if (workloadType === "european") {
        const { S0, K, r, sigma, T, option_type } = config as Record<string, number | string>;
        if (S0 && K && r !== undefined && sigma && T) {
            bsPrice = analyticalBS(
                Number(S0), Number(K), Number(r), Number(sigma), Number(T),
                String(option_type ?? "call")
            );
        }
    }

    return (
        <Table size="small">
            <TableHead>
                <TableRow>
                    <TableCell sx={{ fontWeight: 700 }}>Engine</TableCell>
                    <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
                    <TableCell sx={{ fontWeight: 700 }} align="right">Price</TableCell>
                    {bsPrice !== null && (
                        <TableCell sx={{ fontWeight: 700 }} align="right">BS Error</TableCell>
                    )}
                    {cpuPrice !== null && (
                        <TableCell sx={{ fontWeight: 700 }} align="right">vs CPU</TableCell>
                    )}
                    <TableCell sx={{ fontWeight: 700 }} align="right">Runtime (ms)</TableCell>
                    <TableCell sx={{ fontWeight: 700 }} align="right">± Std</TableCell>
                    <TableCell sx={{ fontWeight: 700 }} align="right">Throughput</TableCell>
                    <TableCell sx={{ fontWeight: 700 }} align="right">Speedup vs CPU</TableCell>
                </TableRow>
            </TableHead>
            <TableBody>
                {results.map(({ engine, run, status }) => {
                    const done = run?.status === "completed";
                    const failed = run?.status === "failed";
                    const rt = run?.mean_runtime_ms ?? null;
                    const price = run?.result_value ?? null;

                    const isFastest = done && rt !== null && rt === minRuntime && runtimes.length > 1;
                    const isSlowest = done && rt !== null && rt === maxRuntime && runtimes.length > 1;

                    const cpuRt = completed.find((r) => r.engine === "cpu")?.run?.mean_runtime_ms ?? null;
                    const speedup =
                        done && cpuRt !== null && rt !== null && rt > 0 && engine !== "cpu"
                            ? cpuRt / rt
                            : null;

                    const bsErr =
                        done && bsPrice !== null && price !== null
                            ? Math.abs(price - bsPrice) / bsPrice * 100
                            : null;

                    const cpuErr =
                        done && cpuPrice !== null && price !== null && engine !== "cpu"
                            ? Math.abs(price - cpuPrice) / Math.abs(cpuPrice) * 100
                            : null;

                    const tp = run?.throughput_paths_per_sec;

                    return (
                        <TableRow key={engine} hover>
                            <TableCell>
                                <Box display="flex" alignItems="center" gap={1}>
                                    <Box
                                        sx={{
                                            width: 10,
                                            height: 10,
                                            borderRadius: "50%",
                                            bgcolor: ENGINE_COLOURS[engine] ?? "#546e7a",
                                        }}
                                    />
                                    <Typography variant="body2" fontWeight={600}>
                                        {engine.toUpperCase()}
                                    </Typography>
                                </Box>
                            </TableCell>
                            <TableCell>
                                {failed ? (
                                    <Typography variant="caption" color="error">
                                        {run?.error_message?.slice(0, 60) ?? "Failed"}
                                    </Typography>
                                ) : done ? (
                                    <Typography variant="caption" color="success.main">
                                        Done
                                    </Typography>
                                ) : (
                                    <Box display="flex" alignItems="center" gap={0.5}>
                                        <CircularProgress size={12} />
                                        <Typography variant="caption" color="text.secondary">
                                            {status}
                                        </Typography>
                                    </Box>
                                )}
                            </TableCell>
                            <TableCell align="right" sx={{ fontFamily: "monospace", fontSize: 12 }}>
                                {fmtPrice(price)}
                            </TableCell>
                            {bsPrice !== null && (
                                <TableCell align="right" sx={{ fontFamily: "monospace", fontSize: 12 }}>
                                    {bsErr !== null ? `${bsErr.toFixed(3)}%` : "—"}
                                </TableCell>
                            )}
                            {cpuPrice !== null && (
                                <TableCell align="right" sx={{ fontFamily: "monospace", fontSize: 12 }}>
                                    {engine === "cpu"
                                        ? "baseline"
                                        : cpuErr !== null
                                            ? `${cpuErr.toFixed(3)}%`
                                            : "—"}
                                </TableCell>
                            )}
                            <Tooltip title={isFastest ? "Fastest" : isSlowest ? "Slowest" : ""}>
                                <TableCell
                                    align="right"
                                    sx={{
                                        fontFamily: "monospace",
                                        fontSize: 12,
                                        fontWeight: isFastest || isSlowest ? 700 : 400,
                                        color: isFastest
                                            ? "success.main"
                                            : isSlowest
                                                ? "error.main"
                                                : "text.primary",
                                    }}
                                >
                                    {fmtMs(rt)}
                                </TableCell>
                            </Tooltip>
                            <TableCell
                                align="right"
                                sx={{ fontFamily: "monospace", fontSize: 12, color: "text.secondary" }}
                            >
                                {fmtMs(run?.std_runtime_ms)}
                            </TableCell>
                            <TableCell align="right" sx={{ fontFamily: "monospace", fontSize: 12 }}>
                                {tp != null
                                    ? tp >= 1_000_000
                                        ? `${(tp / 1_000_000).toFixed(2)}M/s`
                                        : tp >= 1_000
                                            ? `${(tp / 1_000).toFixed(1)}k/s`
                                            : `${tp}/s`
                                    : "—"}
                            </TableCell>
                            <TableCell
                                align="right"
                                sx={{
                                    fontFamily: "monospace",
                                    fontSize: 12,
                                    fontWeight: speedup !== null ? 700 : 400,
                                    color:
                                        speedup !== null
                                            ? speedup >= 1
                                                ? "success.main"
                                                : "error.main"
                                            : "text.disabled",
                                }}
                            >
                                {engine === "cpu"
                                    ? "—"
                                    : speedup !== null
                                        ? `${speedup.toFixed(2)}×`
                                        : "—"}
                            </TableCell>
                        </TableRow>
                    );
                })}
            </TableBody>
        </Table>
    );
}

// ── Charts ────────────────────────────────────────────────────────────────

function CompareBarCharts({ results }: { results: EngineResult[] }) {
    const done = results.filter((r) => r.run?.status === "completed" && r.run.mean_runtime_ms !== null);
    if (done.length === 0) return null;

    const runtimeData = done.map((r) => ({
        engine: r.engine.toUpperCase(),
        runtime: r.run!.mean_runtime_ms,
        fill: ENGINE_COLOURS[r.engine] ?? "#546e7a",
    }));

    const tpData = done
        .filter((r) => r.run!.throughput_paths_per_sec != null)
        .map((r) => ({
            engine: r.engine.toUpperCase(),
            throughput: r.run!.throughput_paths_per_sec,
            fill: ENGINE_COLOURS[r.engine] ?? "#546e7a",
        }));

    return (
        <Grid container spacing={2}>
            <Grid item xs={12} md={tpData.length > 0 ? 6 : 12}>
                <Card variant="outlined">
                    <CardContent>
                        <Typography variant="subtitle2" gutterBottom>
                            Runtime (ms) — lower is better
                        </Typography>
                        <Divider sx={{ mb: 1 }} />
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={runtimeData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="engine" tick={{ fontSize: 12 }} />
                                <YAxis unit=" ms" tick={{ fontSize: 11 }} />
                                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)} ms`] : [v]} />
                                <Bar dataKey="runtime" name="Runtime" radius={[4, 4, 0, 0]}>
                                    {runtimeData.map((d, i) => (
                                        <rect key={i} fill={d.fill} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </Grid>
            {tpData.length > 0 && (
                <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                                Throughput (paths/s) — higher is better
                            </Typography>
                            <Divider sx={{ mb: 1 }} />
                            <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={tpData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="engine" tick={{ fontSize: 12 }} />
                                    <YAxis
                                        tick={{ fontSize: 11 }}
                                        tickFormatter={(v: number) =>
                                            v >= 1_000_000
                                                ? `${(v / 1_000_000).toFixed(1)}M`
                                                : v >= 1_000
                                                    ? `${(v / 1_000).toFixed(0)}k`
                                                    : String(v)
                                        }
                                    />
                                    <RechartTooltip
                                        formatter={(v) =>
                                            typeof v === "number"
                                                ? [`${v.toLocaleString()} paths/s`]
                                                : [v]
                                        }
                                    />
                                    <Bar dataKey="throughput" name="Throughput" radius={[4, 4, 0, 0]}>
                                        {tpData.map((d, i) => (
                                            <rect key={i} fill={d.fill} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>
            )}
        </Grid>
    );
}

// ── GreeksPanel ───────────────────────────────────────────────────────────

function GreeksPanel({ results }: { results: EngineResult[] }) {
    const adResults = results.filter(
        (r) => r.run?.status === "completed" && r.run.ad_mode !== "none" && r.run.greeks
    );
    if (adResults.length === 0) return null;

    const allGreeks = Array.from(
        new Set(adResults.flatMap((r) => Object.keys(r.run!.greeks ?? {})))
    );

    return (
        <Card variant="outlined">
            <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                    Greeks (AD mode)
                </Typography>
                <Divider sx={{ mb: 1 }} />
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{ fontWeight: 700 }}>Engine</TableCell>
                            {allGreeks.map((g) => (
                                <TableCell key={g} sx={{ fontWeight: 700 }} align="right">
                                    {g.charAt(0).toUpperCase() + g.slice(1)}
                                </TableCell>
                            ))}
                            <TableCell sx={{ fontWeight: 700 }} align="right">AD Overhead</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {adResults.map(({ engine, run }) => (
                            <TableRow key={engine} hover>
                                <TableCell>
                                    <Box display="flex" alignItems="center" gap={1}>
                                        <Box
                                            sx={{
                                                width: 10,
                                                height: 10,
                                                borderRadius: "50%",
                                                bgcolor: ENGINE_COLOURS[engine] ?? "#546e7a",
                                            }}
                                        />
                                        <Typography variant="body2">{engine.toUpperCase()}</Typography>
                                    </Box>
                                </TableCell>
                                {allGreeks.map((g) => (
                                    <TableCell
                                        key={g}
                                        align="right"
                                        sx={{ fontFamily: "monospace", fontSize: 12 }}
                                    >
                                        {run!.greeks?.[g]?.toFixed(6) ?? "—"}
                                    </TableCell>
                                ))}
                                <TableCell
                                    align="right"
                                    sx={{
                                        fontFamily: "monospace",
                                        fontSize: 12,
                                        fontWeight: 700,
                                        color:
                                            (run!.ad_overhead_ratio ?? 0) > 3
                                                ? "error.main"
                                                : "success.main",
                                    }}
                                >
                                    {run!.ad_overhead_ratio != null
                                        ? `${run!.ad_overhead_ratio.toFixed(2)}×`
                                        : "—"}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function ComparePage() {
    const [workloads, setWorkloads] = useState<Record<string, WorkloadInfo>>({});
    const [engines, setEngines] = useState<Record<string, EngineInfo>>({});
    const [capabilities, setCapabilities] = useState<Capabilities>({
        cpu: true,
        jax: true,
        cpp: false,
        cuda: false,
    });
    const [metaLoading, setMetaLoading] = useState(false);
    const [submitting, setSubmitting] = useState(false);

    const [results, setResults] = useState<EngineResult[]>([]);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [activeWorkload, setActiveWorkload] = useState("");
    const [activeConfig, setActiveConfig] = useState<Record<string, number | string>>({});

    const [activeTab, setActiveTab] = useState(0);
    const pollTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

    useEffect(() => {
        setMetaLoading(true);
        Promise.all([fetchWorkloads(), fetchEngines(), fetchCapabilities()])
            .then(([wl, eng, caps]) => {
                setWorkloads(wl);
                setEngines(eng);
                setCapabilities(caps);
            })
            .catch(() => { })
            .finally(() => setMetaLoading(false));

        return () => {
            Object.values(pollTimers.current).forEach(clearTimeout);
        };
    }, []);

    function pollEngine(engine: string, runId: string, count = 0) {
        if (count >= MAX_POLLS) {
            setResults((prev) =>
                prev.map((r) =>
                    r.engine === engine
                        ? { ...r, status: "failed", run: r.run ? { ...r.run, status: "failed", error_message: "Polling timed out" } : r.run }
                        : r
                )
            );
            setSubmitting(false);
            return;
        }

        pollTimers.current[engine] = setTimeout(async () => {
            try {
                const run = await pollRun(runId);
                setResults((prev) =>
                    prev.map((r) => (r.engine === engine ? { ...r, run, status: run.status } : r))
                );
                if (run.status === "completed" || run.status === "failed") {
                    setResults((prev) => {
                        const allDone = prev.every(
                            (r) => r.run?.status === "completed" || r.run?.status === "failed"
                        );
                        if (allDone) setSubmitting(false);
                        return prev;
                    });
                } else {
                    pollEngine(engine, runId, count + 1);
                }
            } catch {
                setResults((prev) =>
                    prev.map((r) =>
                        r.engine === engine ? { ...r, status: "failed" } : r
                    )
                );
                setSubmitting(false);
            }
        }, POLL_INTERVAL_MS);
    }

    async function handleCompare(
        workloadType: string,
        config: Record<string, number | string>,
        adMode: string,
        selectedEngines: string[]
    ) {
        // Clear any existing polls
        Object.values(pollTimers.current).forEach(clearTimeout);
        pollTimers.current = {};

        setSubmitError(null);
        setSubmitting(true);
        setActiveWorkload(workloadType);
        setActiveConfig(config);

        const initial: EngineResult[] = selectedEngines.map((eng) => ({
            engine: eng,
            run: null,
            status: "pending",
        }));
        setResults(initial);

        try {
            const ids = await submitRunsForEngines(selectedEngines, {
                workload_type: workloadType,
                ad_mode: adMode,
                config: { ...config, workload_type: workloadType },
            });
            // Start polling each engine
            for (const [eng, id] of Object.entries(ids)) {
                setResults((prev) =>
                    prev.map((r) => (r.engine === eng ? { ...r, status: "running" } : r))
                );
                pollEngine(eng, id, 0);
            }
        } catch (err) {
            setSubmitError("Failed to submit comparison runs. Is the backend running?");
            setSubmitting(false);
            setResults([]);
        }
    }

    const anyResults = results.length > 0;

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            {/* Page header */}
            <Box display="flex" alignItems="center" gap={2}>
                <CompareArrowsIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>
                        Engine Comparison
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Live run comparison and historical benchmark matrix
                    </Typography>
                </Box>
            </Box>

            {/* Tab bar */}
            <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
                <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)}>
                    <Tab label="Run Comparison" />
                    <Tab label="Benchmark Matrix" />
                </Tabs>
            </Box>

            {/* Tab 0: live run comparison */}
            {activeTab === 0 && (
                <Box display="flex" flexDirection="column" gap={3}>
                    {submitError && (
                        <Alert severity="error" onClose={() => setSubmitError(null)}>
                            {submitError}
                        </Alert>
                    )}

                    {metaLoading ? (
                        <CircularProgress />
                    ) : (
                        <CompareForm
                            workloads={workloads}
                            engines={engines}
                            capabilities={capabilities}
                            loading={submitting}
                            onSubmit={handleCompare}
                        />
                    )}

                    {anyResults && (
                        <>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                        Results — {fmtWorkload(activeWorkload)}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                                        Runs in progress will update automatically. Fastest runtime highlighted green, slowest red.
                                    </Typography>
                                    <Divider sx={{ mb: 2 }} />
                                    <Box sx={{ overflowX: "auto" }}>
                                        <CompareResultsTable
                                            results={results}
                                            workloadType={activeWorkload}
                                            config={activeConfig}
                                        />
                                    </Box>
                                </CardContent>
                            </Card>

                            <CompareBarCharts results={results} />
                            <GreeksPanel results={results} />
                        </>
                    )}
                </Box>
            )}

            {/* Tab 1: benchmark matrix */}
            {activeTab === 1 && <MatrixTab />}
        </Box>
    );
}
