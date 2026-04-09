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
import {
    fetchSummary,
    fetchRuns,
    type SummaryResponse,
    type RunStatus,
} from "../api/client";

// ── Stat card ─────────────────────────────────────────────────────────────

function StatCard({ label, value }: { label: string; value: string | number }) {
    return (
        <Card variant="outlined">
            <CardContent>
                <Typography variant="caption" color="text.secondary">
                    {label}
                </Typography>
                <Typography variant="h4" fontWeight={700} mt={0.5}>
                    {value}
                </Typography>
            </CardContent>
        </Card>
    );
}

// ── Inline bar comparison chart ───────────────────────────────────────────

const ENGINE_COLOURS: Record<string, string> = {
    cpu: "#1565c0",
    jax: "#2e7d32",
    cpp: "#b71c1c",
};

function ComparisonChart({ runs }: { runs: RunStatus[] }) {
    const completed = runs.filter(
        (r) => r.status === "completed" && r.mean_runtime_ms !== null
    );

    if (completed.length === 0) {
        return (
            <Typography variant="body2" color="text.secondary" py={2}>
                No completed runs to compare yet.
            </Typography>
        );
    }

    // Group: workload → engine → list of runtimes
    const grouped: Record<string, Record<string, number[]>> = {};
    for (const r of completed) {
        if (!grouped[r.workload_type]) grouped[r.workload_type] = {};
        if (!grouped[r.workload_type][r.engine]) grouped[r.workload_type][r.engine] = [];
        grouped[r.workload_type][r.engine].push(r.mean_runtime_ms as number);
    }

    // Compute averages and global max for proportional scaling
    const avgMap: Record<string, Record<string, number>> = {};
    let maxMs = 0;
    for (const [wl, engines] of Object.entries(grouped)) {
        avgMap[wl] = {};
        for (const [eng, times] of Object.entries(engines)) {
            const avg = times.reduce((a, b) => a + b, 0) / times.length;
            avgMap[wl][eng] = avg;
            if (avg > maxMs) maxMs = avg;
        }
    }
    if (maxMs === 0) maxMs = 1;

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            {Object.entries(avgMap).map(([wl, engines]) => (
                <Box key={wl}>
                    <Typography
                        variant="subtitle2"
                        mb={1}
                        sx={{ textTransform: "capitalize" }}
                    >
                        {wl}
                    </Typography>
                    <Box display="flex" flexDirection="column" gap={1}>
                        {Object.entries(engines).map(([eng, avgMs]) => (
                            <Box key={eng} display="flex" alignItems="center" gap={1.5}>
                                <Typography
                                    variant="caption"
                                    sx={{
                                        width: 36,
                                        flexShrink: 0,
                                        textTransform: "uppercase",
                                        fontWeight: 700,
                                    }}
                                >
                                    {eng}
                                </Typography>
                                <Box
                                    sx={{
                                        height: 16,
                                        width: `${(avgMs / maxMs) * 100}%`,
                                        minWidth: 4,
                                        bgcolor: ENGINE_COLOURS[eng] ?? "#546e7a",
                                        borderRadius: 0.5,
                                        transition: "width 0.4s ease",
                                    }}
                                />
                                <Typography variant="caption" color="text.secondary">
                                    {avgMs.toFixed(1)} ms
                                </Typography>
                            </Box>
                        ))}
                    </Box>
                </Box>
            ))}
        </Box>
    );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function SummaryPage() {
    const [summary, setSummary] = useState<SummaryResponse | null>(null);
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        Promise.all([fetchSummary(), fetchRuns(500)])
            .then(([s, r]) => {
                setSummary(s);
                setRuns(r);
            })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Typography variant="h5" fontWeight={600}>
                Summary
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
                        ].map(({ label, value }) => (
                            <Grid item xs={6} sm={4} md={2} key={label}>
                                <StatCard label={label} value={value} />
                            </Grid>
                        ))}
                    </Grid>

                    {/* By workload / by engine */}
                    <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="subtitle2" gutterBottom>
                                        By Workload
                                    </Typography>
                                    {!summary ||
                                        Object.keys(summary.by_workload).length === 0 ? (
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
                                                <Typography variant="body2">{w}</Typography>
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
                                        By Engine
                                    </Typography>
                                    {!summary ||
                                        Object.keys(summary.by_engine).length === 0 ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No completed runs yet.
                                        </Typography>
                                    ) : (
                                        Object.entries(summary.by_engine).map(([eng, n]) => (
                                            <Box
                                                key={eng}
                                                display="flex"
                                                justifyContent="space-between"
                                                mb={0.5}
                                            >
                                                <Typography variant="body2">
                                                    {eng.toUpperCase()}
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
                    </Grid>

                    {/* Engine comparison chart */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                Engine Comparison — Avg Mean Runtime (ms)
                            </Typography>
                            <Divider sx={{ mb: 2 }} />
                            <ComparisonChart runs={runs} />
                        </CardContent>
                    </Card>
                </>
            )}
        </Box>
    );
}

