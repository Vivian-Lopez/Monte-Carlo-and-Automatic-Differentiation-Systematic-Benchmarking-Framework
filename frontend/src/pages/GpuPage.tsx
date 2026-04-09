import React from "react";
import {
    Box,
    Card,
    CardContent,
    Chip,
    Divider,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Typography,
} from "@mui/material";
import MemoryIcon from "@mui/icons-material/Memory";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";

const MILESTONES = [
    { label: "CUDA kernel for path simulation (parallel paths)", done: false },
    { label: "Forward-mode AD gradient via finite difference", done: false },
    { label: "PyCUDA wrapper integration with engine interface", done: false },
    { label: "Numerical equivalence test vs CPU (< 0.5% error @ M=100k)", done: false },
    { label: "Benchmark: GPU vs CPU runtime across M = 1k–1M", done: false },
    { label: "Dashboard integration (engine picker shows GPU)", done: false },
];

const NOTES = [
    "Each simulation path is independent → embarrassingly parallel on GPU.",
    "Target: NVIDIA T4/A100 on Google Cloud (Days 10–11 cloud integration).",
    "RNG strategy: cuRAND XORWOW per thread with per-seed offset.",
    "Memory layout: coalesced path arrays, shared memory for partial sums.",
    "Occupancy target: ≥ 50% on T4 (compute capability 7.5).",
];

export default function GpuPage() {
    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box display="flex" alignItems="center" gap={2}>
                <MemoryIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>
                        GPU Implementation
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        CUDA Monte Carlo engine — Days 4–5
                    </Typography>
                </Box>
                <Chip label="Coming Soon" color="warning" size="small" sx={{ ml: "auto" }} />
            </Box>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                        Milestone Checklist
                    </Typography>
                    <Divider sx={{ mb: 1 }} />
                    <List dense disablePadding>
                        {MILESTONES.map((m) => (
                            <ListItem key={m.label} disablePadding sx={{ py: 0.25 }}>
                                <ListItemIcon sx={{ minWidth: 32 }}>
                                    {m.done ? (
                                        <CheckCircleOutlineIcon color="success" fontSize="small" />
                                    ) : (
                                        <RadioButtonUncheckedIcon color="disabled" fontSize="small" />
                                    )}
                                </ListItemIcon>
                                <ListItemText
                                    primary={m.label}
                                    primaryTypographyProps={{
                                        variant: "body2",
                                        color: m.done ? "text.primary" : "text.secondary",
                                    }}
                                />
                            </ListItem>
                        ))}
                    </List>
                </CardContent>
            </Card>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                        Design Notes
                    </Typography>
                    <Divider sx={{ mb: 1 }} />
                    <List dense disablePadding>
                        {NOTES.map((n) => (
                            <ListItem key={n} disablePadding sx={{ py: 0.25 }}>
                                <ListItemText
                                    primary={`• ${n}`}
                                    primaryTypographyProps={{ variant: "body2" }}
                                />
                            </ListItem>
                        ))}
                    </List>
                </CardContent>
            </Card>

            <Card variant="outlined" sx={{ bgcolor: "grey.50" }}>
                <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Expected Speedup Targets
                    </Typography>
                    <Box display="flex" gap={4} flexWrap="wrap">
                        {[
                            { label: "M = 10k", target: "5–15×" },
                            { label: "M = 100k", target: "20–50×" },
                            { label: "M = 1M", target: "50–200×" },
                        ].map(({ label, target }) => (
                            <Box key={label} textAlign="center">
                                <Typography variant="caption" color="text.secondary">
                                    {label}
                                </Typography>
                                <Typography variant="h6" fontWeight={700} color="primary.main">
                                    {target}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    vs CPU
                                </Typography>
                            </Box>
                        ))}
                    </Box>
                </CardContent>
            </Card>
        </Box>
    );
}
