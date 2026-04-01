import React, { useState, useEffect } from "react";
import { Box, Typography, Button } from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import RunHistory from "../components/RunHistory";
import { fetchRuns, type RunStatus } from "../api/client";

export default function HistoryPage() {
    const [runs, setRuns] = useState<RunStatus[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function load() {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchRuns(50);
            setRuns(data);
        } catch {
            setError("Could not load run history. Is the backend running?");
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        load();
    }, []);

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
            {error && (
                <Typography color="error">{error}</Typography>
            )}
            <RunHistory runs={runs} loading={loading} />
        </Box>
    );
}
