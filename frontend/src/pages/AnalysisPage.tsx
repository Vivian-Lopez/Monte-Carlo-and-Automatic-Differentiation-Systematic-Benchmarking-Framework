import React, { useEffect, useState } from "react";
import {
    Box,
    Card,
    CardContent,
    Chip,
    CircularProgress,
    Divider,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Tabs,
    Tooltip,
    Typography,
} from "@mui/material";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import {
    BarChart,
    Bar,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartTooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";
import {
    fetchRuns,
    fetchCapabilities,
    type RunStatus,
    type Capabilities,
} from "../api/client";
import { fmtEngine, fmtMs, fmtOverhead, fmtPrice, fmtWorkload } from "../utils/format";

// ── Constants ─────────────────────────────────────────────────────────────

const AD_COLOURS: Record<string, string> = {
    none: "#90a4ae",
    forward: "#1565c0",
    reverse: "#2e7d32",
};

const ENGINE_COLOURS: Record<string, string> = {
    cpu: "#1565c0",
    cuda: "#6a1b9a",
};

// ── AD Tab ────────────────────────────────────────────────────────────────

function OverheadChart({ runs }: { runs: RunStatus[] }) {
    const grouped: Record<string, { runtimes: number[] }> = {};
    for (const r of runs) {
        if (r.status !== "completed" || r.mean_runtime_ms === null) continue;
        const key = `${r.engine}/${r.ad_mode}`;
        if (!grouped[key]) grouped[key] = { runtimes: [] };
        grouped[key].runtimes.push(r.mean_runtime_ms);
    }
    const data = Object.entries(grouped).map(([key, v]) => {
        const [eng, ad] = key.split("/");
        const avg = v.runtimes.reduce((a, b) => a + b, 0) / v.runtimes.length;
        return { key: key.replace("/", " / "), engine: eng, ad_mode: ad, avgRuntime: parseFloat(avg.toFixed(2)) };
    });

    if (data.length === 0) {
        return (
            <Typography variant="body2" color="text.secondary" py={2}>
                No completed runs yet. Run simulations with JAX engine and forward/reverse AD mode.
            </Typography>
        );
    }
    return (
        <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="key" tick={{ fontSize: 11 }} />
                <YAxis unit=" ms" tick={{ fontSize: 11 }} />
                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)} ms`, "Mean runtime"] : [v]} />
                <Bar dataKey="avgRuntime" name="Mean runtime" fill="#1565c0" radius={[3, 3, 0, 0]} />
            </BarChart>
        </ResponsiveContainer>
    );
}

function OverheadTable({ runs }: { runs: RunStatus[] }) {
    const adRuns = runs.filter(
        (r) => r.status === "completed" && r.ad_mode !== "none" && r.ad_overhead_ratio !== null
    );
    if (adRuns.length === 0) {
        return <Typography variant="body2" color="text.secondary" py={1}>No AD runs completed yet.</Typography>;
    }
    return (
        <Table size="small">
            <TableHead>
                <TableRow>
                    {["Engine", "AD Mode", "Workload", "Price", "Runtime (ms)", "AD Overhead"].map((h) => (
                        <TableCell key={h} sx={{ fontWeight: 700, fontSize: 12 }}>{h}</TableCell>
                    ))}
                </TableRow>
            </TableHead>
            <TableBody>
                {adRuns.slice(0, 30).map((r) => (
                    <TableRow key={r.id} hover>
                        <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{fmtEngine(r.engine)}</TableCell>
                        <TableCell>
                            <Chip
                                label={r.ad_mode}
                                size="small"
                                sx={{ bgcolor: AD_COLOURS[r.ad_mode] ?? "#546e7a", color: "#fff", fontSize: 10, height: 20 }}
                            />
                        </TableCell>
                        <TableCell sx={{ fontSize: 12 }}>{fmtWorkload(r.workload_type)}</TableCell>
                        <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{fmtPrice(r.result_value)}</TableCell>
                        <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{fmtMs(r.mean_runtime_ms)}</TableCell>
                        <Tooltip title="Ratio of AD runtime to no-AD runtime" placement="left">
                            <TableCell
                                sx={{
                                    fontFamily: "monospace",
                                    fontSize: 12,
                                    fontWeight: 700,
                                    color: (r.ad_overhead_ratio ?? 0) > 3 ? "error.main" : "success.main",
                                }}
                            >
                                {fmtOverhead(r.ad_overhead_ratio, r.ad_mode)}
                            </TableCell>
                        </Tooltip>
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    );
}

function AdTab({ runs }: { runs: RunStatus[] }) {
    const completed = runs.filter((r) => r.status === "completed");
    const byMode: Record<string, number> = { none: 0, forward: 0, reverse: 0 };
    for (const r of completed) {
        if (r.ad_mode in byMode) byMode[r.ad_mode]++;
    }

    return (
        <Box display="flex" flexDirection="column" gap={3} pt={2}>
            <Box display="flex" gap={2} flexWrap="wrap">
                {Object.entries(byMode).map(([mode, count]) => (
                    <Card key={mode} variant="outlined" sx={{ minWidth: 120 }}>
                        <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
                            <Chip
                                label={mode}
                                size="small"
                                sx={{ bgcolor: AD_COLOURS[mode], color: "#fff", fontSize: 10, mb: 0.5 }}
                            />
                            <Typography variant="h5" fontWeight={700}>{count}</Typography>
                            <Typography variant="caption" color="text.secondary">completed runs</Typography>
                        </CardContent>
                    </Card>
                ))}
            </Box>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle2" gutterBottom>Runtime by Engine / AD Mode</Typography>
                    <Divider sx={{ mb: 2 }} />
                    <OverheadChart runs={runs} />
                </CardContent>
            </Card>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle2" gutterBottom>AD Run Detail</Typography>
                    <Divider sx={{ mb: 1 }} />
                    <OverheadTable runs={runs} />
                </CardContent>
            </Card>
        </Box>
    );
}

// ── GPU Tab ───────────────────────────────────────────────────────────────

function fmtM(m: number): string {
    if (m >= 1_000_000) return `${m / 1_000_000}M`;
    if (m >= 1_000) return `${m / 1_000}k`;
    return String(m);
}

interface PairPoint {
    M: number;
    label: string;
    cpu_ms: number | null;
    cuda_ms: number | null;
    cpu_tp: number | null;
    cuda_tp: number | null;
    speedup: number | null;
}

function buildGpuChartData(runs: RunStatus[]): PairPoint[] {
    const relevant = runs.filter(
        (r) =>
            r.status === "completed" &&
            r.workload_type === "european" &&
            r.ad_mode === "none" &&
            r.mean_runtime_ms !== null
    );
    const byM = new Map<number, { cpu: RunStatus | null; cuda: RunStatus | null }>();
    for (const r of relevant) {
        const M = Number((r.config as Record<string, unknown>)?.M ?? 0);
        if (!M) continue;
        if (!byM.has(M)) byM.set(M, { cpu: null, cuda: null });
        const g = byM.get(M)!;
        if (r.engine === "cpu") g.cpu = r;
        if (r.engine === "cuda") g.cuda = r;
    }

    return [...byM.entries()]
        .sort(([a], [b]) => a - b)
        .map(([M, { cpu, cuda }]) => {
            const cpu_ms = cpu?.mean_runtime_ms ?? null;
            const cuda_ms = cuda?.mean_runtime_ms ?? null;
            const tpFrom = (r: RunStatus | null): number | null => {
                if (!r) return null;
                if (r.throughput_paths_per_sec !== null) return r.throughput_paths_per_sec;
                const m = Number((r.config as Record<string, unknown>)?.M);
                return m && r.mean_runtime_ms ? m / (r.mean_runtime_ms / 1000) : null;
            };
            const speedup =
                cpu_ms !== null && cuda_ms !== null && cuda_ms > 0
                    ? parseFloat((cpu_ms / cuda_ms).toFixed(2))
                    : null;
            return { M, label: fmtM(M), cpu_ms, cuda_ms, cpu_tp: tpFrom(cpu), cuda_tp: tpFrom(cuda), speedup };
        });
}

function GpuTab({ runs, capabilities }: { runs: RunStatus[]; capabilities: Capabilities }) {
    const chartData = buildGpuChartData(runs);
    const hasCuda = chartData.some((p) => p.cuda_ms !== null);

    if (!hasCuda) {
        return (
            <Box pt={2}>
                <Card variant="outlined">
                    <CardContent>
                        <Box py={4} textAlign="center">
                            <Typography variant="body1" color="text.secondary">No CUDA runs available.</Typography>
                            <Typography variant="body2" color="text.secondary" mt={1}>
                                {capabilities.cuda
                                    ? "Run a simulation with Engine = CUDA to see GPU performance data."
                                    : "CUDA is not available in this environment. Seed a synthetic run for UI testing:"}
                            </Typography>
                            {!capabilities.cuda && (
                                <Typography
                                    variant="caption"
                                    fontFamily="monospace"
                                    display="block"
                                    mt={1}
                                    sx={{ bgcolor: "grey.100", px: 2, py: 1, borderRadius: 1, display: "inline-block" }}
                                >
                                    python scripts/seed_cuda_run.py
                                </Typography>
                            )}
                        </Box>
                    </CardContent>
                </Card>
            </Box>
        );
    }

    return (
        <Box display="flex" flexDirection="column" gap={3} pt={2}>
            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>Runtime: CPU vs CUDA</Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                        Mean wall-clock time per run across path counts (ms) — lower is better
                    </Typography>
                    <Divider sx={{ mb: 2 }} />
                    <ResponsiveContainer width="100%" height={240}>
                        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="label" tick={{ fontSize: 12 }} label={{ value: "Paths (M)", position: "insideBottom", offset: -2, style: { fontSize: 11 } }} />
                            <YAxis unit=" ms" tick={{ fontSize: 11 }} />
                            <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)} ms`] : [v]} />
                            <Legend />
                            <Line type="monotone" dataKey="cpu_ms" name="CPU" stroke={ENGINE_COLOURS.cpu} strokeWidth={2} dot activeDot={{ r: 5 }} connectNulls />
                            <Line type="monotone" dataKey="cuda_ms" name="CUDA" stroke={ENGINE_COLOURS.cuda} strokeWidth={2} dot activeDot={{ r: 5 }} connectNulls />
                        </LineChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>Throughput: CPU vs CUDA (paths/s)</Typography>
                    <Divider sx={{ mb: 2 }} />
                    <ResponsiveContainer width="100%" height={240}>
                        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : v >= 1_000 ? `${(v / 1_000).toFixed(0)}k` : String(v)} />
                            <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toLocaleString()} paths/s`] : [v]} />
                            <Legend />
                            <Line type="monotone" dataKey="cpu_tp" name="CPU" stroke={ENGINE_COLOURS.cpu} strokeWidth={2} dot connectNulls />
                            <Line type="monotone" dataKey="cuda_tp" name="CUDA" stroke={ENGINE_COLOURS.cuda} strokeWidth={2} dot connectNulls />
                        </LineChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>Speedup: CUDA vs CPU</Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                        speedup = cpu_time / cuda_time — requires matched CPU + CUDA runs at the same M
                    </Typography>
                    <Divider sx={{ mb: 2 }} />
                    {chartData.filter((p) => p.speedup !== null).length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                            Need both CPU and CUDA runs for the same M to compute speedup.
                        </Typography>
                    ) : (
                        <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={chartData.filter((p) => p.speedup !== null)} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                                <YAxis unit="×" tick={{ fontSize: 11 }} />
                                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)}× faster than CPU`] : [v]} />
                                <Bar dataKey="speedup" name="CUDA Speedup vs CPU" fill={ENGINE_COLOURS.cuda} radius={[3, 3, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                </CardContent>
            </Card>
        </Box>
    );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function AnalysisPage() {
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [capabilities, setCapabilities] = useState<Capabilities>({ cpu: true, jax: true, cpp: false, cuda: false });
    const [loading, setLoading] = useState(false);
    const [tab, setTab] = useState(0);

    useEffect(() => {
        setLoading(true);
        Promise.all([fetchRuns(500), fetchCapabilities()])
            .then(([r, caps]) => { setRuns(r); setCapabilities(caps); })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box display="flex" alignItems="center" gap={2}>
                <AnalyticsIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>Analysis</Typography>
                    <Typography variant="body2" color="text.secondary">
                        AD overhead and GPU performance
                    </Typography>
                </Box>
                {capabilities.cuda && (
                    <Chip label="CUDA Available" color="success" size="small" sx={{ ml: "auto" }} />
                )}
            </Box>

            <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
                <Tabs value={tab} onChange={(_, v) => setTab(v)}>
                    <Tab label="AD Mode" />
                    <Tab label="GPU" />
                </Tabs>
            </Box>

            {loading ? (
                <CircularProgress />
            ) : tab === 0 ? (
                <AdTab runs={runs} />
            ) : (
                <GpuTab runs={runs} capabilities={capabilities} />
            )}
        </Box>
    );
}
