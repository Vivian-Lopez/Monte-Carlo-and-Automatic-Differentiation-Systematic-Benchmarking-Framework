import React, { useEffect, useState } from "react";
import {
    Box,
    Card,
    CardContent,
    Chip,
    CircularProgress,
    Divider,
    Typography,
} from "@mui/material";
import MemoryIcon from "@mui/icons-material/Memory";
import {
    LineChart,
    Line,
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
    fetchRuns,
    fetchCapabilities,
    type RunStatus,
    type Capabilities,
} from "../api/client";

// ── Constants ─────────────────────────────────────────────────────────────

const ENGINE_COLOURS: Record<string, string> = {
    cpu: "#1565c0",
    cuda: "#6a1b9a",
};

// Format raw path counts as "10k", "1M", etc. for axis labels
function fmtM(m: number): string {
    if (m >= 1_000_000) return `${m / 1_000_000}M`;
    if (m >= 1_000) return `${m / 1_000}k`;
    return String(m);
}

// ── Data transformation ───────────────────────────────────────────────────

interface PairPoint {
    M: number;
    label: string;          // human-readable M for chart axis
    cpu_ms: number | null;
    cuda_ms: number | null;
    cpu_tp: number | null;   // paths / second
    cuda_tp: number | null;
    speedup: number | null;   // cpu_ms / cuda_ms
}

/**
 * Build one data point per unique M found in completed European (no-AD) runs.
 * For each M we take the MOST RECENT cpu and cuda run so re-runs update the chart.
 */
function buildChartData(runs: RunStatus[]): PairPoint[] {
    const relevant = runs.filter(
        (r) =>
            r.status === "completed" &&
            r.workload_type === "european" &&
            r.ad_mode === "none" &&
            r.mean_runtime_ms !== null,
    );

    // Group by M → { cpu, cuda } (last write wins → most recent run)
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
            // Throughput: prefer the pre-computed field, fall back to M / runtime_s
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
            return {
                M, label: fmtM(M),
                cpu_ms, cuda_ms,
                cpu_tp: tpFrom(cpu),
                cuda_tp: tpFrom(cuda),
                speedup,
            };
        });
}

// ── Chart sub-components ──────────────────────────────────────────────────

function RuntimeLineChart({ data }: { data: PairPoint[] }) {
    return (
        <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} label={{ value: "Paths (M)", position: "insideBottom", offset: -2, style: { fontSize: 11 } }} />
                <YAxis unit=" ms" tick={{ fontSize: 11 }} />
                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)} ms`] : [v]} />
                <Legend />
                <Line type="monotone" dataKey="cpu_ms" name="CPU" stroke={ENGINE_COLOURS.cpu} strokeWidth={2} dot activeDot={{ r: 5 }} connectNulls />
                <Line type="monotone" dataKey="cuda_ms" name="CUDA" stroke={ENGINE_COLOURS.cuda} strokeWidth={2} dot activeDot={{ r: 5 }} connectNulls />
            </LineChart>
        </ResponsiveContainer>
    );
}

function ThroughputLineChart({ data }: { data: PairPoint[] }) {
    return (
        <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) =>
                        v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M`
                            : v >= 1_000 ? `${(v / 1_000).toFixed(0)}k`
                                : String(v)
                    }
                />
                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toLocaleString()} paths/s`] : [v]} />
                <Legend />
                <Line type="monotone" dataKey="cpu_tp" name="CPU" stroke={ENGINE_COLOURS.cpu} strokeWidth={2} dot connectNulls />
                <Line type="monotone" dataKey="cuda_tp" name="CUDA" stroke={ENGINE_COLOURS.cuda} strokeWidth={2} dot connectNulls />
            </LineChart>
        </ResponsiveContainer>
    );
}

function SpeedupBarChart({ data }: { data: PairPoint[] }) {
    const pts = data.filter((p) => p.speedup !== null);
    if (pts.length === 0) return (
        <Typography variant="body2" color="text.secondary">
            Need both CPU and CUDA runs for the same M to compute speedup.
        </Typography>
    );
    return (
        <ResponsiveContainer width="100%" height={220}>
            <BarChart data={pts} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis unit="×" tick={{ fontSize: 11 }} />
                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)}× faster than CPU`] : [v]} />
                <Bar dataKey="speedup" name="CUDA Speedup vs CPU" fill={ENGINE_COLOURS.cuda} radius={[3, 3, 0, 0]} />
            </BarChart>
        </ResponsiveContainer>
    );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function GpuPage() {
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [capabilities, setCapabilities] = useState<Capabilities>(
        { cpu: true, jax: true, cpp: false, cuda: false }
    );
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        Promise.all([fetchRuns(500), fetchCapabilities()])
            .then(([r, caps]) => { setRuns(r); setCapabilities(caps); })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    const chartData = buildChartData(runs);
    const hasCuda = chartData.some((p) => p.cuda_ms !== null);

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            {/* Header */}
            <Box display="flex" alignItems="center" gap={2}>
                <MemoryIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>
                        GPU Implementation
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        CUDA Monte Carlo — European option pricing, one thread per path
                    </Typography>
                </Box>
                <Chip
                    label={capabilities.cuda ? "CUDA Available" : "CUDA Unavailable"}
                    color={capabilities.cuda ? "success" : "default"}
                    size="small"
                    sx={{ ml: "auto" }}
                />
            </Box>

            {loading ? (
                <CircularProgress />
            ) : !hasCuda ? (
                /* ── No CUDA data placeholder ─────────────────────────────── */
                <Card variant="outlined">
                    <CardContent>
                        <Box py={4} textAlign="center">
                            <Typography variant="body1" color="text.secondary">
                                No CUDA runs available.
                            </Typography>
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
            ) : (
                /* ── Data-driven charts ────────────────────────────────────── */
                <>
                    {/* Runtime: CPU vs CUDA */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                Runtime: CPU vs CUDA
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                                Mean wall-clock time per run across path counts (ms) — lower is better
                            </Typography>
                            <Divider sx={{ mb: 2 }} />
                            <RuntimeLineChart data={chartData} />
                        </CardContent>
                    </Card>

                    {/* Throughput: CPU vs CUDA */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                Throughput: CPU vs CUDA (paths / second)
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                                Computed as M ÷ mean_runtime — higher is better
                            </Typography>
                            <Divider sx={{ mb: 2 }} />
                            <ThroughputLineChart data={chartData} />
                        </CardContent>
                    </Card>

                    {/* Speedup bar chart */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                Speedup: CUDA vs CPU
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                                speedup = cpu_time / cuda_time — requires matched CPU + CUDA runs at the same M
                            </Typography>
                            <Divider sx={{ mb: 2 }} />
                            <SpeedupBarChart data={chartData} />
                        </CardContent>
                    </Card>
                </>
            )}
        </Box>
    );
}
