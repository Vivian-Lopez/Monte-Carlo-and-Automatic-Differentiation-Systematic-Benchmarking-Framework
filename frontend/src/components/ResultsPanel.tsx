import React from "react";
import {
    Card,
    CardContent,
    CardHeader,
    Typography,
    Divider,
    Box,
    Chip,
    CircularProgress,
    Alert,
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import type { RunStatus } from "../api/client";

interface Props {
    run: RunStatus | null;
    polling: boolean;
    error: string | null;
}

function MetricBox({
    label,
    value,
    unit,
}: {
    label: string;
    value: string | number | null;
    unit?: string;
}) {
    return (
        <Box
            sx={{
                p: 2,
                borderRadius: 2,
                bgcolor: "action.hover",
                minWidth: 140,
                flex: 1,
            }}
        >
            <Typography variant="caption" color="text.secondary">
                {label}
            </Typography>
            <Typography variant="h5" fontWeight={700} mt={0.5}>
                {value === null || value === undefined ? "—" : value}
                {unit && value !== null && (
                    <Typography component="span" variant="body2" color="text.secondary" ml={0.5}>
                        {unit}
                    </Typography>
                )}
            </Typography>
        </Box>
    );
}

function analyticalPrice(
    S0: number,
    K: number,
    r: number,
    sigma: number,
    T: number,
    optionType: string
): number {
    // Black-Scholes closed form for European options
    const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    const nc = (x: number) => {
        // Abramowitz & Stegun approximation
        const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        const sign = x < 0 ? -1 : 1;
        const t = 1 / (1 + p * Math.abs(x));
        const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
        return 0.5 * (1 + sign * (1 - poly * Math.exp(-Math.abs(x) * Math.abs(x) / 2)));
    };
    if (optionType === "call") {
        return S0 * nc(d1) - K * Math.exp(-r * T) * nc(d2);
    }
    return K * Math.exp(-r * T) * nc(-d2) - S0 * nc(-d1);
}

export default function ResultsPanel({ run, polling, error }: Props) {
    if (error) {
        return (
            <Alert severity="error" icon={<ErrorIcon />}>
                {error}
            </Alert>
        );
    }

    if (!run && !polling) {
        return (
            <Card variant="outlined">
                <CardContent>
                    <Typography color="text.secondary" textAlign="center" py={4}>
                        Submit a simulation to see results here.
                    </Typography>
                </CardContent>
            </Card>
        );
    }

    if (polling && (!run || run.status === "pending" || run.status === "running")) {
        return (
            <Card variant="outlined">
                <CardContent>
                    <Box display="flex" alignItems="center" gap={2} py={2}>
                        <CircularProgress size={24} />
                        <Typography>
                            {run?.status === "running" ? "Simulation running…" : "Waiting in queue…"}
                        </Typography>
                    </Box>
                </CardContent>
            </Card>
        );
    }

    if (!run) return null;

    const isFailed = run.status === "failed";
    const config = run.config as {
        S0?: number; K?: number; r?: number; sigma?: number;
        T?: number; option_type?: string;
    };

    // Compute Black-Scholes error only for European options with required fields
    let bsError: string | null = null;
    if (
        !isFailed &&
        run.workload_type === "european" &&
        run.result_value !== null &&
        config.S0 && config.K && config.r !== undefined && config.sigma && config.T
    ) {
        const bs = analyticalPrice(
            config.S0, config.K, config.r, config.sigma, config.T,
            config.option_type ?? "call"
        );
        const absErr = Math.abs(run.result_value - bs);
        const relErr = (absErr / bs) * 100;
        bsError = `${absErr.toFixed(4)} (${relErr.toFixed(2)}%)`;
    }

    return (
        <Card variant="outlined">
            <CardHeader
                title="Simulation Results"
                subheader={`Run ID: ${run.id.slice(0, 8)}…`}
                action={
                    isFailed ? (
                        <Chip label="Failed" color="error" size="small" icon={<ErrorIcon />} />
                    ) : (
                        <Chip label="Completed" color="success" size="small" icon={<CheckCircleIcon />} />
                    )
                }
            />
            <Box display="flex" gap={1} px={2} pb={1} flexWrap="wrap">
                <Chip label={run.workload_type} size="small" variant="outlined" />
                <Chip label={run.engine.toUpperCase()} size="small" variant="outlined" color="primary" />
                {run.ad_mode !== "none" && (
                    <Chip label={`AD: ${run.ad_mode}`} size="small" variant="outlined" color="secondary" />
                )}
            </Box>
            <Divider />
            <CardContent>
                {isFailed ? (
                    <Alert severity="error">{run.error_message ?? "Run failed with unknown error."}</Alert>
                ) : (
                    <>
                        <Box display="flex" flexWrap="wrap" gap={2} mb={2}>
                            <MetricBox
                                label="Option Price"
                                value={run.result_value !== null ? run.result_value.toFixed(6) : null}
                            />
                            <MetricBox
                                label="Mean Runtime"
                                value={
                                    run.mean_runtime_ms !== null
                                        ? run.mean_runtime_ms.toFixed(2)
                                        : null
                                }
                                unit="ms"
                            />
                            <MetricBox
                                label="Std Dev"
                                value={
                                    run.std_runtime_ms !== null
                                        ? run.std_runtime_ms.toFixed(2)
                                        : null
                                }
                                unit="ms"
                            />
                            {run.ad_overhead_ratio !== null && run.ad_mode !== "none" && (
                                <MetricBox
                                    label="AD Overhead"
                                    value={run.ad_overhead_ratio.toFixed(2)}
                                    unit="×"
                                />
                            )}
                        </Box>

                        {run.greeks && Object.keys(run.greeks).length > 0 && (
                            <Box mb={2}>
                                <Typography variant="subtitle2" mb={1} color="text.secondary">
                                    Greeks ({run.ad_mode}-mode AD)
                                </Typography>
                                <Box display="flex" flexWrap="wrap" gap={2}>
                                    {Object.entries(run.greeks).map(([name, val]) => (
                                        <MetricBox
                                            key={name}
                                            label={name.charAt(0).toUpperCase() + name.slice(1)}
                                            value={val.toFixed(6)}
                                        />
                                    ))}
                                </Box>
                            </Box>
                        )}

                        {bsError !== null && (
                            <Box
                                sx={{
                                    p: 1.5,
                                    bgcolor: "action.selected",
                                    borderRadius: 1,
                                    display: "inline-block",
                                }}
                            >
                                <Typography variant="body2" color="text.secondary">
                                    Error vs Black-Scholes:{" "}
                                    <Typography component="span" fontWeight={600} color="text.primary">
                                        {bsError}
                                    </Typography>
                                </Typography>
                            </Box>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
}
