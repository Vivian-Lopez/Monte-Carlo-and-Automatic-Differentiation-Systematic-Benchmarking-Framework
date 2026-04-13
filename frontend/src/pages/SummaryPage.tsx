import React, { useEffect, useState } from "react";
import {
    Box,
    Card,
    CardContent,
    CircularProgress,
    Divider,
    Grid,
    Typography,
} from "@mui/material";
import RunTable from "../components/RunTable";
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
    fetchSummary,
    fetchRuns,
    fetchCapabilities,
    type SummaryResponse,
    type RunStatus,
    type Capabilities,
} from "../api/client";

// ── Constants ─────────────────────────────────────────────────────────────

/** Canonical engine order — always shown, regardless of run history. */
export const ALL_ENGINES = ["cpu", "cpp", "jax", "cuda"] as const;
export type EngineName = (typeof ALL_ENGINES)[number];

const ENGINE_COLOURS: Record<string, string> = {
    cpu: "#1565c0",
    jax: "#2e7d32",
    cpp: "#b71c1c",
    cuda: "#6a1b9a",
};

const WORKLOAD_LABELS: Record<string, string> = {
    european: "European",
    asian: "Asian",
    barrier: "Barrier",
    basket: "Basket",
};

// ── Recent runs table ─────────────────────────────────────────────────────

function RecentRunsTable({ runs }: { runs: RunStatus[] }) {
    const recent = runs.slice(0, 20);
    if (recent.length === 0) {
        return (
            <Typography variant="body2" color="text.secondary">
                No runs yet.
            </Typography>
        );
    }
    return <RunTable runs={recent} variant="compact" />;
}

// ── Stat card ─────────────────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
    return (
        <Card variant="outlined">
            <CardContent>
                <Typography variant="caption" color="text.secondary">
                    {label}
                </Typography>
                <Typography variant="h4" fontWeight={700} mt={0.5}>
                    {value}
                </Typography>
                {sub && (
                    <Typography variant="caption" color="text.secondary">
                        {sub}
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
}

// ── Engine × Workload grouped bar chart ───────────────────────────────────

function RuntimeBarChart({ runs, engines }: { runs: RunStatus[]; engines: string[] }) {
    const completed = runs.filter(
        (r) => r.status === "completed" && r.mean_runtime_ms !== null
    );

    if (completed.length === 0) {
        return (
            <Box py={4} textAlign="center">
                <Typography variant="body2" color="text.secondary">
                    No completed runs yet — run a simulation to see results here.
                </Typography>
            </Box>
        );
    }

    // Build: workload → engine → avg runtime
    const acc: Record<string, Record<string, number[]>> = {};
    for (const r of completed) {
        const wl = r.workload_type;
        const eng = r.engine;
        if (!acc[wl]) acc[wl] = {};
        if (!acc[wl][eng]) acc[wl][eng] = [];
        acc[wl][eng].push(r.mean_runtime_ms as number);
    }

    // Only render bars for engines that actually appear in completed runs
    const activeEngines = engines.filter(
        (eng) => Object.values(acc).some((engMap) => eng in engMap)
    );

    const data = Object.entries(acc).map(([wl, engMap]) => {
        const row: Record<string, number | string> = {
            workload: WORKLOAD_LABELS[wl] ?? wl,
        };
        for (const eng of activeEngines) {
            const times = engMap[eng] ?? [];
            if (times.length > 0) {
                row[eng] = parseFloat(
                    (times.reduce((a, b) => a + b, 0) / times.length).toFixed(2)
                );
            }
        }
        return row;
    });

    return (
        <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="workload" tick={{ fontSize: 12 }} />
                <YAxis
                    unit=" ms"
                    tick={{ fontSize: 11 }}
                    label={{
                        value: "Mean runtime (ms)",
                        angle: -90,
                        position: "insideLeft",
                        offset: 10,
                        style: { fontSize: 11 },
                    }}
                />
                <RechartTooltip formatter={(v) => typeof v === "number" ? [`${v.toFixed(2)} ms`] : []} />
                <Legend />
                {activeEngines.map((eng) => (
                    <Bar
                        key={eng}
                        dataKey={eng}
                        name={eng.toUpperCase()}
                        fill={ENGINE_COLOURS[eng] ?? "#546e7a"}
                        radius={[3, 3, 0, 0]}
                    />
                ))}
            </BarChart>
        </ResponsiveContainer>
    );
}

// ── Price comparison chart ────────────────────────────────────────────────

function PriceChart({ runs, engines }: { runs: RunStatus[]; engines: string[] }) {
    const completed = runs.filter(
        (r) => r.status === "completed" && r.result_value !== null && r.ad_mode === "none"
    );

    if (completed.length === 0) return null;

    // Group by workload type, compute mean price per engine
    const acc: Record<string, Record<string, number[]>> = {};
    for (const r of completed) {
        const wl = r.workload_type;
        const eng = r.engine;
        if (!acc[wl]) acc[wl] = {};
        if (!acc[wl][eng]) acc[wl][eng] = [];
        acc[wl][eng].push(r.result_value as number);
    }

    const activeEngines = engines.filter(
        (eng) => Object.values(acc).some((engMap) => eng in engMap)
    );

    const data = Object.entries(acc).map(([wl, engMap]) => {
        const row: Record<string, number | string> = {
            workload: WORKLOAD_LABELS[wl] ?? wl,
        };
        for (const eng of activeEngines) {
            const prices = engMap[eng] ?? [];
            if (prices.length > 0) {
                row[eng] = parseFloat(
                    (prices.reduce((a, b) => a + b, 0) / prices.length).toFixed(4)
                );
            }
        }
        return row;
    });

    return (
        <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="workload" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `$${v.toFixed(2)}`} />
                <RechartTooltip formatter={(v) => typeof v === "number" ? [`$${v.toFixed(4)}`] : []} />
                <Legend />
                {activeEngines.map((eng) => (
                    <Bar
                        key={eng}
                        dataKey={eng}
                        name={eng.toUpperCase()}
                        fill={ENGINE_COLOURS[eng] ?? "#546e7a"}
                        radius={[3, 3, 0, 0]}
                    />
                ))}
            </BarChart>
        </ResponsiveContainer>
    );
}

// ── Recent runs table ─────────────────────────────────────────────────────

// ── Page ──────────────────────────────────────────────────────────────────

export default function SummaryPage() {
    const [summary, setSummary] = useState<SummaryResponse | null>(null);
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [loading, setLoading] = useState(false);
    const [capabilities, setCapabilities] = useState<Capabilities>({ cpu: true, jax: true, cpp: false, cuda: false });

    useEffect(() => {
        setLoading(true);
        Promise.all([fetchSummary(), fetchRuns(500), fetchCapabilities()])
            .then(([s, r, caps]) => {
                setSummary(s);
                setRuns(r);
                setCapabilities(caps);
            })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    // Canonical engine list ordered by ALL_ENGINES, filtered to those the
    // backend reports as available. Any engine that appears in run history
    // but isn't in the canonical list is appended at the end.
    const activeEngines = [
        ...ALL_ENGINES.filter((eng) => capabilities[eng as keyof Capabilities]),
        ...Array.from(new Set(runs.map((r) => r.engine))).filter(
            (eng) => !(ALL_ENGINES as readonly string[]).includes(eng)
        ),
    ];

    const completedRuns = runs.filter((r) => r.status === "completed");
    const uniqueWorkloads = Array.from(new Set(completedRuns.map((r) => r.workload_type))).length;

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Typography variant="h5" fontWeight={600}>
                Dashboard
            </Typography>

            {loading ? (
                <CircularProgress />
            ) : (
                <>
                    {/* Stat cards */}
                    <Grid container spacing={2}>
                        {[
                            { label: "Total Runs", value: summary?.total ?? 0 },
                            { label: "Completed", value: summary?.completed ?? 0 },
                            { label: "Pending", value: summary?.pending ?? 0 },
                            { label: "Failed", value: summary?.failed ?? 0 },
                            {
                                label: "Fastest",
                                value:
                                    summary?.fastest_ms != null
                                        ? `${summary.fastest_ms.toFixed(1)} ms`
                                        : "—",
                            },
                            {
                                label: "Workloads Tested",
                                value: uniqueWorkloads,
                                sub: "distinct",
                            },
                        ].map(({ label, value, sub }) => (
                            <Grid item xs={6} sm={4} md={2} key={label}>
                                <StatCard label={label} value={value} sub={sub} />
                            </Grid>
                        ))}
                    </Grid>

                    {/* Runtime comparison chart */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                Engine Runtime Comparison
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                                Average mean runtime per engine across workloads (ms, lower is better)
                            </Typography>
                            <Divider sx={{ mb: 2 }} />
                            <RuntimeBarChart runs={runs} engines={activeEngines} />
                        </CardContent>
                    </Card>

                    {/* Price comparison */}
                    {completedRuns.some((r) => r.result_value !== null) && (
                        <Card variant="outlined">
                            <CardContent>
                                <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                    Option Price by Workload &amp; Engine
                                </Typography>
                                <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                                    Average MC price (no-AD runs only) — engines should agree within MC error
                                </Typography>
                                <Divider sx={{ mb: 2 }} />
                                <PriceChart runs={runs} engines={activeEngines} />
                            </CardContent>
                        </Card>
                    )}

                    {/* Run counts by workload / by engine side-by-side */}
                    <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Runs by Workload
                                    </Typography>
                                    {!summary || Object.keys(summary.by_workload).length === 0 ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No completed runs yet.
                                        </Typography>
                                    ) : (
                                        Object.entries(summary.by_workload).map(([w, n]) => (
                                            <Box
                                                key={w}
                                                display="flex"
                                                justifyContent="space-between"
                                                mb={0.5}
                                            >
                                                <Typography variant="body2">
                                                    {WORKLOAD_LABELS[w] ?? w}
                                                </Typography>
                                                <Typography variant="body2" fontWeight={600}>
                                                    {n}
                                                </Typography>
                                            </Box>
                                        ))
                                    )}
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Runs by Engine
                                    </Typography>
                                    {!summary ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No completed runs yet.
                                        </Typography>
                                    ) : (
                                        activeEngines.map((eng) => (
                                            <Box
                                                key={eng}
                                                display="flex"
                                                justifyContent="space-between"
                                                mb={0.5}
                                            >
                                                <Box display="flex" alignItems="center" gap={1}>
                                                    <Box
                                                        sx={{
                                                            width: 10,
                                                            height: 10,
                                                            borderRadius: "50%",
                                                            bgcolor:
                                                                ENGINE_COLOURS[eng] ?? "#546e7a",
                                                        }}
                                                    />
                                                    <Typography variant="body2">
                                                        {eng.toUpperCase()}
                                                    </Typography>
                                                </Box>
                                                <Typography variant="body2" fontWeight={600}>
                                                    {summary.by_engine[eng] ?? 0}
                                                </Typography>
                                            </Box>
                                        ))
                                    )}
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>

                    {/* Recent runs table */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                Recent Runs
                            </Typography>
                            <Divider sx={{ mb: 1 }} />
                            <RecentRunsTable runs={runs} />
                        </CardContent>
                    </Card>
                </>
            )}
        </Box>
    );
}

