import React, { useState, useEffect, useCallback, useRef } from "react";
import { Box, Typography, Alert } from "@mui/material";
import SimulationForm, { type FormValues } from "../components/SimulationForm";
import ResultsPanel from "../components/ResultsPanel";
import {
    fetchWorkloads,
    fetchEngines,
    submitRun,
    pollRun,
    type WorkloadInfo,
    type EngineInfo,
    type RunStatus,
    type SimulationRequest,
} from "../api/client";

const POLL_INTERVAL_MS = 1500;
const MAX_POLLS = 120; // 3 minutes tops

interface Props {
    onBackendStatus: (s: "online" | "offline") => void;
}

export default function SimulatePage({ onBackendStatus }: Props) {
    const [workloads, setWorkloads] = useState<Record<string, WorkloadInfo>>({});
    const [engines, setEngines] = useState<Record<string, EngineInfo>>({});
    const [submitting, setSubmitting] = useState(false);
    const [polling, setPolling] = useState(false);
    const [currentRun, setCurrentRun] = useState<RunStatus | null>(null);
    const [runError, setRunError] = useState<string | null>(null);
    const pollCount = useRef(0);
    const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        async function init() {
            try {
                const [wl, eng] = await Promise.all([fetchWorkloads(), fetchEngines()]);
                setWorkloads(wl);
                setEngines(eng);
                onBackendStatus("online");
            } catch {
                onBackendStatus("offline");
            }
        }
        init();
        return () => {
            if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        };
    }, [onBackendStatus]);

    const startPolling = useCallback((runId: string) => {
        pollCount.current = 0;
        setPolling(true);

        async function doPoll() {
            try {
                const run = await pollRun(runId);
                setCurrentRun(run);
                pollCount.current += 1;

                if (run.status === "completed" || run.status === "failed") {
                    setPolling(false);
                    setSubmitting(false);
                    return;
                }
                if (pollCount.current >= MAX_POLLS) {
                    setPolling(false);
                    setSubmitting(false);
                    setRunError("Polling timed out — check the server.");
                    return;
                }
                pollTimerRef.current = setTimeout(doPoll, POLL_INTERVAL_MS);
            } catch (err) {
                setPolling(false);
                setSubmitting(false);
                setRunError("Lost connection while polling for results.");
            }
        }

        doPoll();
    }, []);

    async function handleSubmit(values: FormValues) {
        setRunError(null);
        setCurrentRun(null);
        setSubmitting(true);

        const config: SimulationRequest["config"] = {
            S0: values.S0,
            K: values.K,
            r: values.r,
            sigma: values.sigma,
            T: values.T,
            N: values.N,
            M: values.M,
            seed: values.seed,
            option_type: values.option_type,
        };

        // Attach workload-specific fields
        if (values.workload_type === "asian") {
            (config as Record<string, unknown>)["averaging"] = values.averaging;
        }
        if (values.workload_type === "barrier") {
            (config as Record<string, unknown>)["B"] = values.B;
            (config as Record<string, unknown>)["barrier_type"] = values.barrier_type;
            (config as Record<string, unknown>)["barrier_side"] = values.barrier_side;
        }
        if (values.workload_type === "basket") {
            (config as Record<string, unknown>)["n_assets"] = values.n_assets;
            (config as Record<string, unknown>)["rho"] = values.rho;
        }

        const payload: SimulationRequest = {
            workload_type: values.workload_type,
            engine: values.engine,
            ad_mode: values.ad_mode,
            config,
        };

        try {
            const { id } = await submitRun(payload);
            startPolling(id);
        } catch (err: unknown) {
            setSubmitting(false);
            if (err && typeof err === "object" && "response" in err) {
                const axiosErr = err as { response?: { data?: { description?: string } } };
                setRunError(
                    axiosErr.response?.data?.description ??
                    "Failed to submit run. Is the backend running?"
                );
            } else {
                setRunError("Cannot reach backend. Make sure the Flask server is running on port 5050.");
            }
        }
    }

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Typography variant="h5" fontWeight={600}>
                Run Simulation
            </Typography>

            {runError && (
                <Alert severity="error" onClose={() => setRunError(null)}>
                    {runError}
                </Alert>
            )}

            <SimulationForm
                workloads={workloads}
                engines={engines}
                loading={submitting}
                onSubmit={handleSubmit}
            />

            <ResultsPanel
                run={currentRun}
                polling={polling}
                error={null}
            />
        </Box>
    );
}
