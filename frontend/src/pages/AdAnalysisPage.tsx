import React, { useEffect, useState } from "react";
import {
    Box,
    Card,
    CardContent,
    Chip,
    CircularProgress,
    Divider,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Tooltip,
    Typography,
} from "@mui/material";
import FunctionsIcon from "@mui/icons-material/Functions";
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
import { fetchRuns, type RunStatus } from "../api/client";

const AD_COLOURS: Record<string, string> = {
    none: "#90a4ae",
    forward: "#1565c0",
    reverse: "#2e7d32",
};

// ── AD overhead chart ─────────────────────────────────────────────────────

function OverheadChart({ runs }: { runs: RunStatus[] }) {
    // Group by engine+ad_mode, compute mean overhead ratio and mean runtime
    const grouped: Record<string, { ratios: number[]; runtimes: number[] }> = {};

    for (const r of runs) {
        if (r.status !== "completed" || r.mean_runtime_ms === null) continue;
        const key = `${r.engine}/${r.ad_mode}`;
        if (!grouped[key]) grouped[key] = { ratios: [], runtimes: [] };
        grouped[key].runtimes.push(r.mean_runtime_ms);
        if (r.ad_overhead_ratio !== null) grouped[key].ratios.push(r.ad_overhead_ratio);
    }

    const data = Object.entries(grouped).map(([key, v]) => {
        const [eng, ad] = key.split("/");
        const avgRuntime = v.runtimes.reduce((a, b) => a + b, 0) / v.runtimes.length;
        const avgOverhead =
            v.ratios.length > 0
                ? v.ratios.reduce((a, b) => a + b, 0) / v.ratios.length
                : null;
        return { key, engine: eng, ad_mode: ad, avgRuntime, avgOverhead };
    });

    if (data.length === 0) {
        return (
            <Typography variant="body2" color="text.secondary" py={2}>
                No completed runs with AD data yet. Run a simulation with JAX engine and
                forward/reverse AD mode.
            </Typography>
        );
    }

    // Runtime comparison chart
    return (
        <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                    dataKey="key"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v) => v.replace("/", " / ")}
                />
                <YAxis
                    unit=" ms"
                    tick={{ fontSize: 11 }}
                    label={{ value: "Mean runtime (ms)", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 11 } }}
                />
<RechartTooltip
                    formatter={(val) =>
                        typeof val === "number" ? [`${val.toFixed(2)} ms`, "Mean runtime"] : [val]
                    }
                />
                <Bar
                    dataKey="avgRuntime"
                    name="Mean runtime"
                    fill="#1565c0"
                    radius={[3, 3, 0, 0]}
                />
            </BarChart>
        </ResponsiveContainer>
    );
}

// ── Overhead ratio scatter ────────────────────────────────────────────────

function OverheadTable({ runs }: { runs: RunStatus[] }) {
    const adRuns = runs.filter(
        (r) => r.status === "completed" && r.ad_mode !== "none" && r.ad_overhead_ratio !== null
    );

    if (adRuns.length === 0) {
        return (
            <Typography variant="body2" color="text.secondary" py={1}>
                No AD runs completed yet.
            </Typography>
        );
    }

    return (
        <Table size="small">
            <TableHead>
                <TableRow>
                    {["Engine", "AD Mode", "Workload", "Price", "Runtime (ms)", "AD Overhead ×"].map(
                        (h) => (
                            <TableCell key={h} sx={{ fontWeight: 700, fontSize: 12 }}>
                                {h}
                            </TableCell>
                        )
                    )}
                </TableRow>
            </TableHead>
            <TableBody>
                {adRuns.slice(0, 30).map((r) => (
                    <TableRow key={r.id} hover>
                        <TableCell sx={{ textTransform: "uppercase", fontSize: 12 }}>
                            {r.engine}
                        </TableCell>
                        <TableCell>
                            <Chip
                                label={r.ad_mode}
                                size="small"
                                sx={{
                                    bgcolor: AD_COLOURS[r.ad_mode] ?? "#546e7a",
                                    color: "#fff",
                                    fontSize: 10,
                                    height: 20,
                                }}
                            />
                        </TableCell>
                        <TableCell sx={{ fontSize: 12 }}>{r.workload_type}</TableCell>
                        <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>
                            {r.result_value?.toFixed(4) ?? "—"}
                        </TableCell>
                        <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>
                            {r.mean_runtime_ms?.toFixed(2) ?? "—"}
                        </TableCell>
                        <Tooltip
                            title="Ratio of AD runtime to no-AD runtime for the same config"
                            placement="left"
                        >
                            <TableCell
                                sx={{
                                    fontFamily: "monospace",
                                    fontSize: 12,
                                    fontWeight: 700,
                                    color:
                                        (r.ad_overhead_ratio ?? 0) > 3 ? "error.main" : "success.main",
                                }}
                            >
                                {r.ad_overhead_ratio?.toFixed(2) ?? "—"}×
                            </TableCell>
                        </Tooltip>
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function AdAnalysisPage() {
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        fetchRuns(500)
            .then(setRuns)
            .catch(() => {})
            .finally(() => setLoading(false));
    }, []);

    const adModes = ["none", "forward", "reverse"];
    const adRuns = runs.filter((r) => r.status === "completed");
    const byMode: Record<string, number> = { none: 0, forward: 0, reverse: 0 };
    for (const r of adRuns) {
        if (r.ad_mode in byMode) byMode[r.ad_mode]++;
    }

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box display="flex" alignItems="center" gap={2}>
                <FunctionsIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>
                        AD Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Automatic differentiation overhead — forward vs reverse mode (Day 6–7)
                    </Typography>
                </Box>
                <Chip label="Active" color="success" size="small" sx={{ ml: "auto" }} />
            </Box>

            {loading ? (
                <CircularProgress />
            ) : (
                <>
                    {/* Mode breakdown chips */}
                    <Box display="flex" gap={2} flexWrap="wrap">
                        {adModes.map((mode) => (
                            <Card key={mode} variant="outlined" sx={{ minWidth: 120 }}>
                                <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
                                    <Chip
                                        label={mode}
                                        size="small"
                                        sx={{
                                            bgcolor: AD_COLOURS[mode],
                                            color: "#fff",
                                            fontSize: 10,
                                            mb: 0.5,
                                        }}
                                    />
                                    <Typography variant="h5" fontWeight={700}>
                                        {byMode[mode]}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        completed runs
                                    </Typography>
                                </CardContent>
                            </Card>
                        ))}
                    </Box>

                    {/* Runtime chart */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                                Runtime by Engine / AD Mode
                            </Typography>
                            <Divider sx={{ mb: 2 }} />
                            <OverheadChart runs={runs} />
                        </CardContent>
                    </Card>

                    {/* AD runs table */}
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                                AD Run Detail
                            </Typography>
                            <Divider sx={{ mb: 1 }} />
                            <OverheadTable runs={runs} />
                        </CardContent>
                    </Card>

                    {/* Upcoming: framework comparison */}
                    <Card variant="outlined" sx={{ bgcolor: "grey.50", borderStyle: "dashed" }}>
                        <CardContent>
                            <Box display="flex" alignItems="center" gap={1} mb={1}>
                                <Typography variant="subtitle2">
                                    Framework Comparison (Day 6–7)
                                </Typography>
                                <Chip label="Planned" color="default" size="small" />
                            </Box>
                            <Typography variant="body2" color="text.secondary">
                                PyTorch autograd vs JAX forward-mode vs JAX reverse-mode on identical
                                configs. Will compare: overhead ratio, memory footprint, numerical accuracy
                                of delta estimate vs analytic Black-Scholes delta.
                            </Typography>
                        </CardContent>
                    </Card>
                </>
            )}
        </Box>
    );
}
