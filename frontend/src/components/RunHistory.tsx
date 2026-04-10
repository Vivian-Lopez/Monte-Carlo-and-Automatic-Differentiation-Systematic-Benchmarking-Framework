import React from "react";
import {
    Card,
    CardHeader,
    CardContent,
    Table,
    TableHead,
    TableBody,
    TableRow,
    TableCell,
    Chip,
    Typography,
    CircularProgress,
    Box,
} from "@mui/material";
import type { RunStatus } from "../api/client";

interface Props {
    runs: RunStatus[];
    loading: boolean;
}

function statusColor(s: string): "success" | "warning" | "error" | "default" {
    if (s === "completed") return "success";
    if (s === "failed") return "error";
    if (s === "running") return "warning";
    return "default";
}

export default function RunHistory({ runs, loading }: Props) {
    return (
        <Card variant="outlined">
            <CardHeader title="Run History" subheader="Most recent 50 runs" />
            <CardContent sx={{ p: 0 }}>
                {loading ? (
                    <Box display="flex" justifyContent="center" py={4}>
                        <CircularProgress size={28} />
                    </Box>
                ) : runs.length === 0 ? (
                    <Typography color="text.secondary" textAlign="center" py={4}>
                        No runs yet. Submit a simulation to see history.
                    </Typography>
                ) : (
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>ID</TableCell>
                                <TableCell>Workload</TableCell>
                                <TableCell>Engine</TableCell>
                                <TableCell>AD</TableCell>
                                <TableCell>Status</TableCell>
                                <TableCell align="right">Price</TableCell>
                                <TableCell align="right">Mean (ms)</TableCell>
                                <TableCell align="right">Std (ms)</TableCell>
                                <TableCell align="right">AD Overhead</TableCell>
                                <TableCell>Error</TableCell>
                                <TableCell>Submitted</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {runs.map((run) => (
                                <TableRow key={run.id} hover>
                                    <TableCell>
                                        <Typography variant="caption" fontFamily="monospace">
                                            {run.id.slice(0, 8)}
                                        </Typography>
                                    </TableCell>
                                    <TableCell>{run.workload_type}</TableCell>
                                    <TableCell>{run.engine}</TableCell>
                                    <TableCell>{run.ad_mode}</TableCell>
                                    <TableCell>
                                        <Chip
                                            label={run.status}
                                            color={statusColor(run.status)}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell align="right">
                                        {run.result_value !== null
                                            ? run.result_value.toFixed(4)
                                            : "—"}
                                    </TableCell>
                                    <TableCell align="right">
                                        {run.mean_runtime_ms !== null
                                            ? run.mean_runtime_ms.toFixed(1)
                                            : "—"}
                                    </TableCell>
                                    <TableCell align="right">
                                        {run.std_runtime_ms !== null
                                            ? run.std_runtime_ms.toFixed(1)
                                            : "—"}
                                    </TableCell>
                                    <TableCell align="right">
                                        {run.ad_overhead_ratio !== null && run.ad_mode !== "none"
                                            ? `${run.ad_overhead_ratio.toFixed(2)}×`
                                            : "—"}
                                    </TableCell>
                                    <TableCell>
                                        {run.error_message ? (
                                            <Typography variant="caption" color="error" noWrap sx={{ maxWidth: 150, display: "inline-block" }}>
                                                {run.error_message.slice(0, 60)}
                                            </Typography>
                                        ) : "—"}
                                    </TableCell>
                                    <TableCell>
                                        <Typography variant="caption" color="text.secondary">
                                            {run.created_at
                                                ? new Date(run.created_at).toLocaleString()
                                                : "—"}
                                        </Typography>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                )}
            </CardContent>
        </Card>
    );
}
