import React, { useState, useEffect } from "react";
import {
    Box,
    Typography,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import RunHistory from "../components/RunHistory";
import { fetchRuns, type RunStatus } from "../api/client";

const WORKLOAD_OPTIONS = ["", "european", "asian", "barrier", "basket"];
const ENGINE_OPTIONS = ["", "cpu", "jax", "cpp"];
const STATUS_OPTIONS = ["", "pending", "running", "completed", "failed"];

export default function HistoryPage() {
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [workload, setWorkload] = useState("");
    const [engine, setEngine] = useState("");
    const [status, setStatus] = useState("");

    async function load() {
        setLoading(true);
        setError(null);
        try {
            const filters: Record<string, string> = {};
            if (workload) filters.workload_type = workload;
            if (engine) filters.engine = engine;
            if (status) filters.status = status;
            const data = await fetchRuns(50, Object.keys(filters).length ? filters : undefined);
            setRuns(data);
        } catch {
            setError("Could not load run history. Is the backend running?");
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        load();
    }, [workload, engine, status]);

    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h5" fontWeight={600}>
                    Run History
                </Typography>
                <Button
                    startIcon={<RefreshIcon />}
                    variant="outlined"
                    size="small"
                    onClick={load}
                    disabled={loading}
                >
                    Refresh
                </Button>
            </Box>

            <Box display="flex" gap={2} flexWrap="wrap">
                <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel>Workload</InputLabel>
                    <Select
                        value={workload}
                        label="Workload"
                        onChange={(e) => setWorkload(e.target.value)}
                    >
                        <MenuItem value="">All</MenuItem>
                        {WORKLOAD_OPTIONS.filter(Boolean).map((w) => (
                            <MenuItem key={w} value={w}>{w}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                    <InputLabel>Engine</InputLabel>
                    <Select
                        value={engine}
                        label="Engine"
                        onChange={(e) => setEngine(e.target.value)}
                    >
                        <MenuItem value="">All</MenuItem>
                        {ENGINE_OPTIONS.filter(Boolean).map((e) => (
                            <MenuItem key={e} value={e}>{e.toUpperCase()}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                    <InputLabel>Status</InputLabel>
                    <Select
                        value={status}
                        label="Status"
                        onChange={(e) => setStatus(e.target.value)}
                    >
                        <MenuItem value="">All</MenuItem>
                        {STATUS_OPTIONS.filter(Boolean).map((s) => (
                            <MenuItem key={s} value={s}>{s}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Box>

            {error && (
                <Typography color="error">{error}</Typography>
            )}
            <RunHistory runs={runs} loading={loading} />
        </Box>
    );
}
