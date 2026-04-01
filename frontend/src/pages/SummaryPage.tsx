import React, { useEffect, useState } from "react";
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    CircularProgress,
} from "@mui/material";
import { fetchRuns, type RunStatus } from "../api/client";

interface SummaryStats {
    total: number;
    completed: number;
    failed: number;
    avgRuntimeMs: number | null;
    fastestMs: number | null;
    slowestMs: number | null;
}

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

function computeStats(runs: RunStatus[]): SummaryStats {
    const completed = runs.filter((r) => r.status === "completed");
    const failed = runs.filter((r) => r.status === "failed");
    const runtimes = completed
        .map((r) => r.mean_runtime_ms)
        .filter((v): v is number => v !== null);

    return {
        total: runs.length,
        completed: completed.length,
        failed: failed.length,
        avgRuntimeMs: runtimes.length
            ? runtimes.reduce((a, b) => a + b, 0) / runtimes.length
            : null,
        fastestMs: runtimes.length ? Math.min(...runtimes) : null,
        slowestMs: runtimes.length ? Math.max(...runtimes) : null,
    };
}

export default function SummaryPage() {
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        fetchRuns(200)
            .then(setRuns)
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    const stats = computeStats(runs);

    const byWorkload: Record<string, number> = {};
    const byEngine: Record<string, number> = {};
    runs.forEach((r) => {
        if (r.status === "completed") {
            byWorkload[r.workload_type] = (byWorkload[r.workload_type] ?? 0) + 1;
            byEngine[r.engine] = (byEngine[r.engine] ?? 0) + 1;
        }
    });

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Typography variant="h5" fontWeight={600}>
                Summary
            </Typography>

            {loading ? (
                <CircularProgress />
            ) : (
                <>
                    <Grid container spacing={2}>
                        <Grid item xs={6} sm={4} md={2}>
                            <StatCard label="Total Runs" value={stats.total} />
                        </Grid>
                        <Grid item xs={6} sm={4} md={2}>
                            <StatCard label="Completed" value={stats.completed} />
                        </Grid>
                        <Grid item xs={6} sm={4} md={2}>
                            <StatCard label="Failed" value={stats.failed} />
                        </Grid>
                        <Grid item xs={6} sm={4} md={3}>
                            <StatCard
                                label="Avg Runtime"
                                value={
                                    stats.avgRuntimeMs !== null
                                        ? `${stats.avgRuntimeMs.toFixed(1)} ms`
                                        : "—"
                                }
                            />
                        </Grid>
                        <Grid item xs={6} sm={4} md={2}>
                            <StatCard
                                label="Fastest"
                                value={
                                    stats.fastestMs !== null ? `${stats.fastestMs.toFixed(1)} ms` : "—"
                                }
                            />
                        </Grid>
                        <Grid item xs={6} sm={4} md={2}>
                            <StatCard
                                label="Slowest"
                                value={
                                    stats.slowestMs !== null ? `${stats.slowestMs.toFixed(1)} ms` : "—"
                                }
                            />
                        </Grid>
                    </Grid>

                    <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                            <Card variant="outlined">
                                <CardContent>
                                    <Typography variant="subtitle2" gutterBottom>
                                        By Workload
                                    </Typography>
                                    {Object.keys(byWorkload).length === 0 ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No completed runs yet.
                                        </Typography>
                                    ) : (
                                        Object.entries(byWorkload).map(([w, n]) => (
                                            <Box key={w} display="flex" justifyContent="space-between" mb={0.5}>
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
                                    {Object.keys(byEngine).length === 0 ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No completed runs yet.
                                        </Typography>
                                    ) : (
                                        Object.entries(byEngine).map(([eng, n]) => (
                                            <Box key={eng} display="flex" justifyContent="space-between" mb={0.5}>
                                                <Typography variant="body2">{eng.toUpperCase()}</Typography>
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
                </>
            )}
        </Box>
    );
}
