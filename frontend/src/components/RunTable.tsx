/**
 * RunTable — shared run-results table used by RunHistory and SummaryPage.
 *
 * variant="full"    ID · Workload · Engine · AD · Status · Price · Runtime (ms) · Std (ms) · AD Overhead · Error · Submitted
 * variant="compact" Status · Workload · Engine · AD · Price · Runtime (ms) · Std (ms) · AD Overhead
 *
 * In "full" variant, clicking a row opens a detail drawer.
 */
import React, { useState } from "react";
import {
    Box,
    Chip,
    Divider,
    Drawer,
    IconButton,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Tooltip,
    Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import type { RunStatus } from "../api/client";
import {
    fmtWorkload,
    fmtEngine,
    fmtPrice,
    fmtMs,
    fmtOverhead,
    fmtDatetime,
} from "../utils/format";

// ── Detail drawer ─────────────────────────────────────────────────────────

function RunDetailDrawer({ run, onClose }: { run: RunStatus; onClose: () => void }) {
    const config = run.config as Record<string, unknown>;
    return (
        <Drawer anchor="right" open onClose={onClose} PaperProps={{ sx: { width: 360, p: 3 } }}>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography variant="h6" fontWeight={600}>Run Detail</Typography>
                <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
            </Box>
            <Divider sx={{ mb: 2 }} />

            <Typography variant="caption" color="text.secondary">ID</Typography>
            <Typography variant="body2" fontFamily="monospace" mb={1.5}>{run.id}</Typography>

            <Box display="flex" gap={1} flexWrap="wrap" mb={2}>
                <Chip label={fmtWorkload(run.workload_type)} size="small" variant="outlined" />
                <Chip label={fmtEngine(run.engine)} size="small" variant="outlined" color="primary" />
                {run.ad_mode !== "none" && (
                    <Chip label={`AD: ${run.ad_mode}`} size="small" variant="outlined" color="secondary" />
                )}
                <Chip
                    label={run.status}
                    size="small"
                    color={run.status === "completed" ? "success" : run.status === "failed" ? "error" : "default"}
                />
            </Box>

            <Typography variant="subtitle2" mb={0.5}>Config</Typography>
            <Box sx={{ bgcolor: "grey.100", borderRadius: 1, p: 1.5, mb: 2 }}>
                {Object.entries(config).filter(([k]) => k !== "workload_type").map(([k, v]) => (
                    <Box key={k} display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">{k}</Typography>
                        <Typography variant="caption" fontFamily="monospace">{String(v)}</Typography>
                    </Box>
                ))}
            </Box>

            <Typography variant="subtitle2" mb={0.5}>Results</Typography>
            <Box sx={{ bgcolor: "grey.100", borderRadius: 1, p: 1.5, mb: 2 }}>
                {[
                    ["Price", fmtPrice(run.result_value)],
                    ["Runtime (ms)", fmtMs(run.mean_runtime_ms)],
                    ["Std (ms)", fmtMs(run.std_runtime_ms)],
                    ["AD Overhead", fmtOverhead(run.ad_overhead_ratio, run.ad_mode)],
                    ["Throughput", run.throughput_paths_per_sec != null
                        ? `${run.throughput_paths_per_sec.toLocaleString(undefined, { maximumFractionDigits: 0 })} paths/s`
                        : "—"],
                ].map(([label, value]) => (
                    <Box key={label} display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">{label}</Typography>
                        <Typography variant="caption" fontFamily="monospace">{value}</Typography>
                    </Box>
                ))}
            </Box>

            {run.greeks && Object.keys(run.greeks).length > 0 && (
                <>
                    <Typography variant="subtitle2" mb={0.5}>Greeks</Typography>
                    <Box sx={{ bgcolor: "grey.100", borderRadius: 1, p: 1.5, mb: 2 }}>
                        {Object.entries(run.greeks).map(([g, v]) => (
                            <Box key={g} display="flex" justifyContent="space-between">
                                <Typography variant="caption" color="text.secondary">
                                    {g.charAt(0).toUpperCase() + g.slice(1)}
                                </Typography>
                                <Typography variant="caption" fontFamily="monospace">{v.toFixed(6)}</Typography>
                            </Box>
                        ))}
                    </Box>
                </>
            )}

            {run.error_message && (
                <>
                    <Typography variant="subtitle2" color="error" mb={0.5}>Error</Typography>
                    <Typography variant="caption" color="error">{run.error_message}</Typography>
                </>
            )}

            <Box mt={2}>
                {[
                    ["Submitted", fmtDatetime(run.created_at)],
                    ["Started", fmtDatetime(run.started_at)],
                    ["Completed", fmtDatetime(run.completed_at)],
                ].map(([label, value]) => (
                    <Box key={label} display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">{label}</Typography>
                        <Typography variant="caption">{value}</Typography>
                    </Box>
                ))}
            </Box>
        </Drawer>
    );
}

export type RunTableVariant = "full" | "compact";

interface Props {
    runs: RunStatus[];
    variant?: RunTableVariant;
}

function statusColor(s: string): "success" | "warning" | "error" | "default" {
    if (s === "completed") return "success";
    if (s === "failed") return "error";
    if (s === "running") return "warning";
    return "default";
}

const MONO = { fontFamily: "monospace", fontSize: 12 } as const;
const HDR = { fontWeight: 700 } as const;

export default function RunTable({ runs, variant = "full" }: Props) {
    const full = variant === "full";
    const [selectedRun, setSelectedRun] = useState<RunStatus | null>(null);

    return (
        <>
            {selectedRun && (
                <RunDetailDrawer run={selectedRun} onClose={() => setSelectedRun(null)} />
            )}
            <Table size="small">
                <TableHead>
                    <TableRow>
                        {full && <TableCell sx={HDR}>ID</TableCell>}
                        <TableCell sx={HDR}>Workload</TableCell>
                        <TableCell sx={HDR}>Engine</TableCell>
                        <TableCell sx={HDR}>AD</TableCell>
                        <TableCell sx={HDR}>Status</TableCell>
                        <TableCell sx={HDR} align="right">Price</TableCell>
                        <TableCell sx={HDR} align="right">Runtime (ms)</TableCell>
                        <TableCell sx={HDR} align="right">Std (ms)</TableCell>
                        <TableCell sx={HDR} align="right">AD Overhead</TableCell>
                        {full && <TableCell sx={HDR}>Error</TableCell>}
                        {full && <TableCell sx={HDR}>Submitted</TableCell>}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {runs.map((run) => (
                        <TableRow
                            key={run.id}
                            hover
                            onClick={full ? () => setSelectedRun(run) : undefined}
                            sx={full ? { cursor: "pointer" } : undefined}
                        >
                            {full && (
                                <TableCell>
                                    <Typography variant="caption" fontFamily="monospace">
                                        {run.id.slice(0, 8)}
                                    </Typography>
                                </TableCell>
                            )}
                            <TableCell sx={{ fontSize: 12 }}>{fmtWorkload(run.workload_type)}</TableCell>
                            <TableCell sx={{ fontSize: 12 }}>{fmtEngine(run.engine)}</TableCell>
                            <TableCell sx={{ fontSize: 12 }}>{run.ad_mode}</TableCell>
                            <TableCell>
                                <Chip
                                    label={run.status}
                                    color={statusColor(run.status)}
                                    size="small"
                                    sx={full ? undefined : { fontSize: 10, height: 20 }}
                                />
                            </TableCell>
                            <TableCell sx={MONO} align="right">{fmtPrice(run.result_value)}</TableCell>
                            <TableCell sx={MONO} align="right">{fmtMs(run.mean_runtime_ms)}</TableCell>
                            <TableCell sx={{ ...MONO, color: "text.secondary" }} align="right">
                                {fmtMs(run.std_runtime_ms)}
                            </TableCell>
                            <Tooltip
                                title={run.ad_mode !== "none" ? "Ratio of AD runtime to no-AD baseline" : ""}
                                placement="left"
                            >
                                <TableCell
                                    sx={{
                                        ...MONO,
                                        fontWeight: run.ad_mode !== "none" && run.ad_overhead_ratio !== null ? 700 : 400,
                                        color:
                                            run.ad_mode !== "none" && run.ad_overhead_ratio !== null
                                                ? run.ad_overhead_ratio > 3
                                                    ? "error.main"
                                                    : "success.main"
                                                : "text.disabled",
                                    }}
                                    align="right"
                                >
                                    {fmtOverhead(run.ad_overhead_ratio, run.ad_mode)}
                                </TableCell>
                            </Tooltip>
                            {full && (
                                <TableCell>
                                    {run.error_message ? (
                                        <Typography
                                            variant="caption"
                                            color="error"
                                            noWrap
                                            sx={{ maxWidth: 150, display: "inline-block" }}
                                        >
                                            {run.error_message.slice(0, 60)}
                                        </Typography>
                                    ) : (
                                        "—"
                                    )}
                                </TableCell>
                            )}
                            {full && (
                                <TableCell>
                                    <Typography variant="caption" color="text.secondary">
                                        {fmtDatetime(run.created_at)}
                                    </Typography>
                                </TableCell>
                            )}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </>
    );
}
