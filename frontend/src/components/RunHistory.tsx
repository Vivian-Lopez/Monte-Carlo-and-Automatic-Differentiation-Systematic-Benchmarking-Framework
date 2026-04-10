import React from "react";
import {
    Card,
    CardHeader,
    CardContent,
    Typography,
    CircularProgress,
    Box,
} from "@mui/material";
import type { RunStatus } from "../api/client";
import RunTable from "./RunTable";

interface Props {
    runs: RunStatus[];
    loading: boolean;
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
                    <RunTable runs={runs} variant="full" />
                )}
            </CardContent>
        </Card>
    );
}
