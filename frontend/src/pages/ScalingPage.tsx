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
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import ScaleIcon from "@mui/icons-material/Scale";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";

const MILESTONES = [
    { label: "Strong scaling: fix M=1M, vary threads 1→32", done: false },
    { label: "Weak scaling: scale M proportionally with threads", done: false },
    { label: "Amdahl efficiency curve across CPU/JAX/C++/GPU", done: false },
    { label: "Bottleneck identification (RNG-bound vs compute-bound)", done: false },
    { label: "Memory bandwidth profiling (Intel VTune / NVIDIA Nsight)", done: false },
];

const MATRIX_HEADERS = ["Config", "CPU (np)", "JAX", "C++ OMP", "GPU"];
const MATRIX_ROWS = [
    ["M=1k, T=1, 1 thread", "—", "—", "—", "—"],
    ["M=10k, T=1, 1 thread", "—", "—", "—", "—"],
    ["M=100k, T=1, 8 threads", "—", "—", "—", "—"],
    ["M=1M, T=1, 32 threads", "—", "—", "—", "—"],
];

export default function ScalingPage() {
    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box display="flex" alignItems="center" gap={2}>
                <ScaleIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>
                        Parallelism &amp; Scaling Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Strong/weak scaling experiments — Days 8–9
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
                                    <RadioButtonUncheckedIcon color="disabled" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText
                                    primary={m.label}
                                    primaryTypographyProps={{ variant: "body2", color: "text.secondary" }}
                                />
                            </ListItem>
                        ))}
                    </List>
                </CardContent>
            </Card>

            <Card variant="outlined">
                <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                        Benchmark Matrix Template
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                        Mean runtime (ms) — will be populated by <code>experiments/run_benchmark_suite.py</code>
                    </Typography>
                    <Divider sx={{ mb: 1 }} />
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                {MATRIX_HEADERS.map((h) => (
                                    <TableCell key={h} sx={{ fontWeight: 700 }}>{h}</TableCell>
                                ))}
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {MATRIX_ROWS.map((row) => (
                                <TableRow key={row[0]}>
                                    {row.map((cell, i) => (
                                        <TableCell
                                            key={i}
                                            sx={{ color: i === 0 ? "text.primary" : "text.disabled" }}
                                        >
                                            {cell}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card variant="outlined" sx={{ bgcolor: "grey.50" }}>
                <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Theory: Strong Scaling Efficiency
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Efficiency E(p) = T(1) / (p × T(p)). For embarrassingly parallel workloads
                        (independent paths), we expect near-linear strong scaling. Deviations indicate
                        synchronisation overhead or memory bandwidth saturation.
                    </Typography>
                </CardContent>
            </Card>
        </Box>
    );
}
