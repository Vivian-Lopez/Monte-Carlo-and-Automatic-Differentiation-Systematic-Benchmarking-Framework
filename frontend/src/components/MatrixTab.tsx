/**
 * MatrixTab — Benchmark Matrix view.
 *
 * Displays a 2-D matrix where:
 *   Rows    = (engine, ad_mode) stack
 *   Columns = unique (config, ad_mode) from historical completed runs
 *
 * Each cell shows throughput (primary), runtime ± std (secondary), run count.
 * Best-per-column is highlighted green; worst is highlighted red.
 *
 * A "Speedup" toggle replaces raw values with speedup vs baseline (cpu/none).
 * A per-column bar chart is rendered below the table for any selected column.
 */
import React, { useEffect, useState } from "react";
import {
    Alert,
    Box,
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
    ToggleButton,
    ToggleButtonGroup,
    Tooltip,
    Typography,
} from "@mui/material";
import {
    BarChart,
    Bar,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartTooltip,
    ResponsiveContainer,
} from "recharts";
import {
    fetchCompareMatrix,
    fetchEngines,
    type BenchmarkMatrix,
    type MatrixCell,
    type MatrixColumn,
} from "../api/client";
import { fmtWorkload } from "../utils/format";

// ── Constants ─────────────────────────────────────────────────────────────

const ENGINE_COLOURS: Record<string, string> = {
    cpu: "#1565c0",
    jax: "#2e7d32",
    cpp: "#b71c1c",
    cuda: "#6a1b9a",
};

const WORKLOADS = ["european", "asian", "barrier", "basket"];

type SortMode = "throughput" | "runtime" | "speedup";
type ViewMode = "raw" | "speedup";

// ── Formatting helpers ────────────────────────────────────────────────────

function fmtThroughput(v: number | null): string {
    if (v == null) return "—";
    if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M/s`;
    if (v >= 1_000) return `${(v / 1_000).toFixed(1)}k/s`;
    return `${v.toFixed(0)}/s`;
}

function fmtRt(v: number | null): string {
    return v != null ? `${v.toFixed(2)}` : "—";
}

function fmtStd(v: number | null): string {
    return v != null ? `±${v.toFixed(2)}` : "";
}

function fmtSpeedup(v: number | null): string {
    if (v == null) return "—";
    const sign = v >= 1 ? "+" : "";
    return `${sign}${((v - 1) * 100).toFixed(1)}%`;
}

/** Short column header: "10k / fwd" */
function colLabel(col: MatrixColumn): string {
    const M = col.config.M;
    const mStr = M != null
        ? Number(M) >= 1_000_000
            ? `${Number(M) / 1_000_000}M`
            : Number(M) >= 1_000
                ? `${Number(M) / 1_000}k`
                : String(M)
        : "?";
    const adShort = col.ad_mode === "none" ? "" : ` / ${col.ad_mode.slice(0, 3)}`;
    return `${mStr}${adShort}`;
}

/** Full tooltip for column header */
function colTooltip(col: MatrixColumn): string {
    const params = Object.entries(col.config)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ");
    return `${params}, ad_mode=${col.ad_mode}`;
}

/** Row label: "JAX / forward" */
function rowLabel(engine: string, adMode: string): string {
    return `${engine.toUpperCase()}${adMode !== "none" ? ` / ${adMode}` : ""}`;
}

// ── Per-column sort ───────────────────────────────────────────────────────

/**
 * For a given column index, compute per-row sort value.
 * Used to find best/worst and to re-order rows by selected metric.
 */
function sortValue(
    cell: MatrixCell | null,
    mode: SortMode,
    baselineRt: number | null
): number | null {
    if (!cell) return null;
    if (mode === "throughput") return cell.throughput_paths_per_sec;
    if (mode === "runtime") return cell.mean_runtime_ms !== null ? -cell.mean_runtime_ms : null; // negate so higher = better
    if (mode === "speedup" && baselineRt != null && cell.mean_runtime_ms != null && cell.mean_runtime_ms > 0)
        return baselineRt / cell.mean_runtime_ms;
    return null;
}

// ── Cell renderer ─────────────────────────────────────────────────────────

interface CellProps {
    cell: MatrixCell | null;
    isBest: boolean;
    isWorst: boolean;
    view: ViewMode;
    baselineRt: number | null;  // baseline mean_runtime_ms for speedup
    isBaseline: boolean;
}

function MatrixCellDisplay({ cell, isBest, isWorst, view, baselineRt, isBaseline }: CellProps) {
    if (!cell) {
        return (
            <TableCell align="center" sx={{ color: "text.disabled", fontSize: 11 }}>—</TableCell>
        );
    }

    const speedup = (baselineRt != null && cell.mean_runtime_ms != null && cell.mean_runtime_ms > 0)
        ? baselineRt / cell.mean_runtime_ms
        : null;

    const bgColor = isBest
        ? "rgba(46,125,50,0.10)"
        : isWorst
            ? "rgba(183,28,28,0.07)"
            : undefined;

    const textColor = isBest
        ? "success.main"
        : isWorst
            ? "error.main"
            : "text.primary";

    if (view === "speedup") {
        return (
            <TableCell
                align="center"
                sx={{ bgcolor: bgColor, p: 1 }}
            >
                {isBaseline ? (
                    <Typography variant="caption" color="text.secondary" fontFamily="monospace">baseline</Typography>
                ) : (
                    <Typography variant="caption" fontWeight={700} color={textColor} fontFamily="monospace">
                        {fmtSpeedup(speedup)}
                    </Typography>
                )}
                <Typography variant="caption" display="block" color="text.disabled" sx={{ fontSize: 10 }}>
                    n={cell.run_count}
                </Typography>
            </TableCell>
        );
    }

    return (
        <TableCell
            align="center"
            sx={{ bgcolor: bgColor, p: 1, verticalAlign: "top" }}
        >
            {/* Primary: throughput */}
            <Typography
                variant="caption"
                fontFamily="monospace"
                fontWeight={700}
                color={textColor}
                display="block"
            >
                {fmtThroughput(cell.throughput_paths_per_sec)}
            </Typography>
            {/* Secondary: runtime ± std */}
            <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: 10 }}>
                {fmtRt(cell.mean_runtime_ms)} ms {fmtStd(cell.std_runtime_ms)}
            </Typography>
            {/* Tertiary: run count */}
            <Typography variant="caption" display="block" color="text.disabled" sx={{ fontSize: 10 }}>
                n={cell.run_count}
            </Typography>
            {/* AD overhead (if applicable) */}
            {cell.ad_overhead_ratio != null && (
                <Typography
                    variant="caption"
                    display="block"
                    sx={{ fontSize: 10 }}
                    color={cell.ad_overhead_ratio > 3 ? "error.main" : "text.secondary"}
                >
                    {cell.ad_overhead_ratio.toFixed(2)}× AD
                </Typography>
            )}
        </TableCell>
    );
}

// ── Speedup bar chart for a selected column ───────────────────────────────

interface SpeedupChartProps {
    matrix: BenchmarkMatrix;
    colIdx: number;
    baselineRowKey: string;
    view: ViewMode;
}

function ColumnBarChart({ matrix, colIdx, baselineRowKey, view }: SpeedupChartProps) {
    const col = matrix.columns[colIdx];
    if (!col) return null;

    const baselineRow = matrix.rows.find(
        (r) => `${r.engine}/${r.ad_mode}` === baselineRowKey
    );
    const baselineRt = baselineRow?.cells[colIdx]?.mean_runtime_ms ?? null;

    const data = matrix.rows
        .map((row) => {
            const cell = row.cells[colIdx];
            if (!cell) return null;
            const label = rowLabel(row.engine, row.ad_mode);
            const colour = ENGINE_COLOURS[row.engine] ?? "#546e7a";

            if (view === "speedup") {
                const speedup = baselineRt != null && cell.mean_runtime_ms != null && cell.mean_runtime_ms > 0
                    ? baselineRt / cell.mean_runtime_ms
                    : null;
                return { label, value: speedup, colour };
            }
            // Raw: throughput
            return { label, value: cell.throughput_paths_per_sec, colour };
        })
        .filter((d): d is { label: string; value: number | null; colour: string } => d !== null && d.value !== null);

    if (data.length === 0) return null;

    const isSpeedup = view === "speedup";

    return (
        <Card variant="outlined">
            <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                    {isSpeedup ? "Speedup vs baseline" : "Throughput (paths/s)"} — {colLabel(col)}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                    {colTooltip(col)}
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                        <YAxis
                            tick={{ fontSize: 11 }}
                            tickFormatter={isSpeedup
                                ? (v) => `${v.toFixed(1)}×`
                                : (v: number) =>
                                    v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M`
                                        : v >= 1_000 ? `${(v / 1_000).toFixed(0)}k`
                                            : String(v)
                            }
                        />
                        <RechartTooltip
                            formatter={(v) =>
                                isSpeedup
                                    ? typeof v === "number" ? [`${v.toFixed(2)}×`] : [v]
                                    : typeof v === "number" ? [`${v.toLocaleString()} paths/s`] : [v]
                            }
                        />
                        <Bar dataKey="value" name={isSpeedup ? "Speedup" : "Throughput"} radius={[4, 4, 0, 0]}>
                            {data.map((d, i) => (
                                <Cell key={i} fill={d.colour} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}

// ── Main matrix table ─────────────────────────────────────────────────────

interface MatrixTableProps {
    matrix: BenchmarkMatrix;
    view: ViewMode;
    sortMode: SortMode;
    selectedColIdx: number | null;
    onSelectCol: (idx: number) => void;
}

function MatrixTable({ matrix, view, sortMode, selectedColIdx, onSelectCol }: MatrixTableProps) {
    const { columns, rows } = matrix;

    // For each column, find best and worst performing row (by sortValue)
    const baselineRowKey = matrix.baseline;
    const baselineRow = rows.find((r) => `${r.engine}/${r.ad_mode}` === baselineRowKey);

    const colStats = columns.map((_, ci) => {
        const baselineRt = baselineRow?.cells[ci]?.mean_runtime_ms ?? null;
        const vals = rows
            .map((r) => ({ rk: `${r.engine}/${r.ad_mode}`, sv: sortValue(r.cells[ci], sortMode, baselineRt) }))
            .filter((x) => x.sv !== null) as { rk: string; sv: number }[];
        if (vals.length === 0) return { best: null, worst: null };
        const best = vals.reduce((a, b) => (b.sv > a.sv ? b : a)).rk;
        const worst = vals.reduce((a, b) => (b.sv < a.sv ? b : a)).rk;
        return { best, worst };
    });

    const hasAnyData = rows.some((r) => r.cells.some((c) => c !== null));

    if (!hasAnyData) {
        return (
            <Box py={4} textAlign="center">
                <Typography variant="body2" color="text.secondary">
                    No completed runs found for this workload. Submit some runs first.
                </Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ overflowX: "auto" }}>
            <Table size="small" sx={{ minWidth: 500 }}>
                <TableHead>
                    <TableRow>
                        {/* Row label column */}
                        <TableCell sx={{ fontWeight: 700, minWidth: 120, whiteSpace: "nowrap" }}>
                            Engine / AD
                        </TableCell>
                        {columns.map((col, ci) => (
                            <Tooltip key={col.col_key} title={colTooltip(col)} placement="top" arrow>
                                <TableCell
                                    align="center"
                                    onClick={() => onSelectCol(ci === selectedColIdx ? -1 : ci)}
                                    sx={{
                                        fontWeight: 700,
                                        fontSize: 12,
                                        cursor: "pointer",
                                        whiteSpace: "nowrap",
                                        bgcolor: ci === selectedColIdx ? "primary.50" : undefined,
                                        borderBottom: ci === selectedColIdx ? "2px solid" : undefined,
                                        borderColor: ci === selectedColIdx ? "primary.main" : undefined,
                                        "&:hover": { bgcolor: "action.hover" },
                                        minWidth: 100,
                                    }}
                                >
                                    {colLabel(col)}
                                </TableCell>
                            </Tooltip>
                        ))}
                    </TableRow>
                    {/* Sub-header: sort mode indicator */}
                    <TableRow sx={{ bgcolor: "grey.50" }}>
                        <TableCell sx={{ fontSize: 10, color: "text.secondary", py: 0.25 }}>
                            {view === "raw"
                                ? sortMode === "throughput" ? "throughput · runtime ± std · n"
                                    : sortMode === "runtime" ? "runtime (ms) · n"
                                        : "speedup · n"
                                : "speedup vs baseline · n"
                            }
                        </TableCell>
                        {columns.map((col) => (
                            <TableCell key={col.col_key} sx={{ py: 0.25 }} />
                        ))}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {rows.map((row) => {
                        const rk = `${row.engine}/${row.ad_mode}`;
                        const isBaselineRow = rk === baselineRowKey;
                        return (
                            <TableRow key={rk} hover sx={{ "&:hover": { bgcolor: "action.hover" } }}>
                                {/* Row label */}
                                <TableCell sx={{ whiteSpace: "nowrap" }}>
                                    <Box display="flex" alignItems="center" gap={1}>
                                        <Box
                                            sx={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: "50%",
                                                bgcolor: ENGINE_COLOURS[row.engine] ?? "#546e7a",
                                                flexShrink: 0,
                                            }}
                                        />
                                        <Typography variant="body2" fontWeight={600} sx={{ fontSize: 12 }}>
                                            {rowLabel(row.engine, row.ad_mode)}
                                        </Typography>
                                        {isBaselineRow && (
                                            <Typography variant="caption" color="text.disabled" sx={{ fontSize: 9 }}>
                                                [base]
                                            </Typography>
                                        )}
                                    </Box>
                                </TableCell>
                                {row.cells.map((cell, ci) => {
                                    const baselineRt = baselineRow?.cells[ci]?.mean_runtime_ms ?? null;
                                    const { best, worst } = colStats[ci];
                                    return (
                                        <MatrixCellDisplay
                                            key={ci}
                                            cell={cell}
                                            isBest={best === rk && rows.filter((r) => r.cells[ci] !== null).length > 1}
                                            isWorst={worst === rk && rows.filter((r) => r.cells[ci] !== null).length > 1}
                                            view={view}
                                            baselineRt={baselineRt}
                                            isBaseline={isBaselineRow}
                                        />
                                    );
                                })}
                            </TableRow>
                        );
                    })}
                </TableBody>
            </Table>
        </Box>
    );
}

// ── Controls ──────────────────────────────────────────────────────────────

interface MatrixControlsProps {
    workload: string;
    onWorkloadChange: (w: string) => void;
    availableEngines: string[];
    selectedEngines: Set<string>;
    onToggleEngine: (e: string) => void;
    baseline: string;
    onBaselineChange: (b: string) => void;
    baselineOptions: string[];
    view: ViewMode;
    onViewChange: (v: ViewMode) => void;
    sortMode: SortMode;
    onSortChange: (s: SortMode) => void;
    loading: boolean;
    onRefresh: () => void;
}

function MatrixControls({
    workload, onWorkloadChange,
    availableEngines, selectedEngines, onToggleEngine,
    baseline, onBaselineChange, baselineOptions,
    view, onViewChange,
    sortMode, onSortChange,
    loading, onRefresh,
}: MatrixControlsProps) {
    return (
        <Card variant="outlined">
            <CardContent>
                <Grid container spacing={2} alignItems="flex-start">
                    {/* Workload */}
                    <Grid item xs={12} sm={3}>
                        <FormControl size="small" fullWidth>
                            <InputLabel>Workload</InputLabel>
                            <Select label="Workload" value={workload} onChange={(e) => onWorkloadChange(e.target.value)}>
                                {WORKLOADS.map((w) => (
                                    <MenuItem key={w} value={w}>{fmtWorkload(w)}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>

                    {/* Baseline */}
                    <Grid item xs={12} sm={3}>
                        <FormControl size="small" fullWidth>
                            <InputLabel>Baseline (speedup ref.)</InputLabel>
                            <Select
                                label="Baseline (speedup ref.)"
                                value={baselineOptions.includes(baseline) ? baseline : (baselineOptions[0] ?? "cpu/none")}
                                onChange={(e) => onBaselineChange(e.target.value)}
                            >
                                {baselineOptions.map((opt) => (
                                    <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>

                    {/* View toggle */}
                    <Grid item xs={12} sm={3}>
                        <Box>
                            <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>View</Typography>
                            <ToggleButtonGroup
                                size="small"
                                value={view}
                                exclusive
                                onChange={(_, v) => { if (v) onViewChange(v); }}
                            >
                                <ToggleButton value="raw">Raw</ToggleButton>
                                <ToggleButton value="speedup">Speedup</ToggleButton>
                            </ToggleButtonGroup>
                        </Box>
                    </Grid>

                    {/* Sort */}
                    <Grid item xs={12} sm={3}>
                        <Box>
                            <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
                                Highlight best by
                            </Typography>
                            <ToggleButtonGroup
                                size="small"
                                value={sortMode}
                                exclusive
                                onChange={(_, v) => { if (v) onSortChange(v); }}
                            >
                                <ToggleButton value="throughput">Throughput</ToggleButton>
                                <ToggleButton value="runtime">Runtime</ToggleButton>
                            </ToggleButtonGroup>
                        </Box>
                    </Grid>

                    {/* Engine filter */}
                    {availableEngines.length > 0 && (
                        <Grid item xs={12}>
                            <FormLabel component="legend" sx={{ fontSize: 13, mb: 0.5 }}>
                                Filter engines
                            </FormLabel>
                            <FormGroup row>
                                {availableEngines.map((eng) => (
                                    <FormControlLabel
                                        key={eng}
                                        control={
                                            <Checkbox
                                                size="small"
                                                checked={selectedEngines.has(eng)}
                                                onChange={() => onToggleEngine(eng)}
                                                sx={{
                                                    color: ENGINE_COLOURS[eng] ?? undefined,
                                                    "&.Mui-checked": { color: ENGINE_COLOURS[eng] ?? undefined },
                                                }}
                                            />
                                        }
                                        label={<Typography variant="body2">{eng.toUpperCase()}</Typography>}
                                    />
                                ))}
                            </FormGroup>
                        </Grid>
                    )}
                </Grid>
            </CardContent>
        </Card>
    );
}

// ── MatrixTab (exported) ──────────────────────────────────────────────────

export default function MatrixTab() {
    const [workload, setWorkload] = useState("european");
    const [availableEngines, setAvailableEngines] = useState<string[]>([]);
    const [selectedEngines, setSelectedEngines] = useState<Set<string>>(new Set());
    const [baseline, setBaseline] = useState("cpu/none");
    const [view, setView] = useState<ViewMode>("raw");
    const [sortMode, setSortMode] = useState<SortMode>("throughput");
    const [selectedColIdx, setSelectedColIdx] = useState<number | null>(null);

    const [matrix, setMatrix] = useState<BenchmarkMatrix | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Load engine list once
    useEffect(() => {
        fetchEngines()
            .then((eng) => {
                const names = Object.keys(eng);
                setAvailableEngines(names);
                setSelectedEngines(new Set(names));
            })
            .catch(() => {
                setAvailableEngines(["cpu", "jax"]);
                setSelectedEngines(new Set(["cpu", "jax"]));
            });
    }, []);

    // Fetch matrix whenever controls change
    useEffect(() => {
        if (selectedEngines.size === 0) return;
        setLoading(true);
        setError(null);
        setSelectedColIdx(null);
        fetchCompareMatrix({
            workload,
            engines: [...selectedEngines],
            baseline,
        })
            .then(setMatrix)
            .catch(() => setError("Could not load matrix. Is the backend running?"))
            .finally(() => setLoading(false));
    }, [workload, selectedEngines, baseline]);

    function toggleEngine(eng: string) {
        setSelectedEngines((prev) => {
            const next = new Set(prev);
            if (next.has(eng)) {
                if (next.size > 1) next.delete(eng); // always keep at least one
            } else {
                next.add(eng);
            }
            return next;
        });
    }

    // Derive baseline options from current matrix rows
    const baselineOptions = matrix
        ? matrix.rows.map((r) => `${r.engine}/${r.ad_mode}`)
        : ["cpu/none"];

    const effectiveBaseline = baselineOptions.includes(baseline) ? baseline : (baselineOptions[0] ?? "cpu/none");

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box>
                <Typography variant="body2" color="text.secondary">
                    Rows = engine stack · Columns = (config, AD mode) · Click a column header to see a bar chart.
                    Green = best per column, red = worst. Speedup toggle shows % improvement vs baseline.
                </Typography>
            </Box>

            <MatrixControls
                workload={workload}
                onWorkloadChange={setWorkload}
                availableEngines={availableEngines}
                selectedEngines={selectedEngines}
                onToggleEngine={toggleEngine}
                baseline={effectiveBaseline}
                onBaselineChange={setBaseline}
                baselineOptions={baselineOptions.length > 0 ? baselineOptions : ["cpu/none"]}
                view={view}
                onViewChange={setView}
                sortMode={sortMode}
                onSortChange={setSortMode}
                loading={loading}
                onRefresh={() => {
                    setMatrix(null);
                    setLoading(true);
                    fetchCompareMatrix({ workload, engines: [...selectedEngines], baseline })
                        .then(setMatrix)
                        .catch(() => setError("Could not load matrix."))
                        .finally(() => setLoading(false));
                }}
            />

            {error && <Alert severity="error">{error}</Alert>}

            {loading ? (
                <Box display="flex" justifyContent="center" py={4}>
                    <CircularProgress />
                </Box>
            ) : matrix ? (
                <>
                    <Card variant="outlined">
                        <CardContent sx={{ p: 0, "&:last-child": { pb: 0 } }}>
                            <Box px={2} pt={1.5} pb={0.5} display="flex" justifyContent="space-between" alignItems="center">
                                <Typography variant="subtitle1" fontWeight={600}>
                                    {fmtWorkload(matrix.workload)} — Benchmark Matrix
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {matrix.rows.length} engine stack{matrix.rows.length !== 1 ? "s" : ""} ·{" "}
                                    {matrix.columns.length} config{matrix.columns.length !== 1 ? "s" : ""}
                                </Typography>
                            </Box>
                            <Divider />
                            <Box p={0}>
                                <MatrixTable
                                    matrix={matrix}
                                    view={view}
                                    sortMode={sortMode}
                                    selectedColIdx={selectedColIdx}
                                    onSelectCol={(idx) => setSelectedColIdx(prev => prev === idx ? null : idx)}
                                />
                            </Box>
                        </CardContent>
                    </Card>

                    {/* Per-column bar chart (shown when a column is selected) */}
                    {selectedColIdx !== null && (
                        <ColumnBarChart
                            matrix={matrix}
                            colIdx={selectedColIdx}
                            baselineRowKey={effectiveBaseline}
                            view={view}
                        />
                    )}

                    {/* Legend */}
                    <Box display="flex" gap={3} flexWrap="wrap">
                        <Box display="flex" alignItems="center" gap={0.5}>
                            <Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: "rgba(46,125,50,0.25)" }} />
                            <Typography variant="caption" color="text.secondary">Best per column</Typography>
                        </Box>
                        <Box display="flex" alignItems="center" gap={0.5}>
                            <Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: "rgba(183,28,28,0.15)" }} />
                            <Typography variant="caption" color="text.secondary">Worst per column</Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                            Click a column header to see a per-engine bar chart.
                        </Typography>
                    </Box>
                </>
            ) : null}
        </Box>
    );
}
