/**
 * RunTable — shared run-results table used by RunHistory and SummaryPage.
 *
 * variant="full"    ID · Workload · Engine · AD · Status · Price · Runtime (ms) · Std (ms) · AD Overhead · Error · Submitted
 * variant="compact" Status · Workload · Engine · AD · Price · Runtime (ms) · Std (ms) · AD Overhead
 */
import React from "react";
import {
    Chip,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Tooltip,
    Typography,
} from "@mui/material";
import type { RunStatus } from "../api/client";
import {
    fmtWorkload,
    fmtEngine,
    fmtPrice,
    fmtMs,
    fmtOverhead,
    fmtDatetime,
} from "../utils/format";

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

    return (
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
                    <TableRow key={run.id} hover>
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
    );
}
