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
import CloudIcon from "@mui/icons-material/Cloud";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";

const MILESTONES = [
    { label: "Deploy benchmark runner on Google Cloud Run / GCE", done: false },
    { label: "Profile CPU, memory, and network with Cloud Monitoring", done: false },
    { label: "Cost/perf matrix: n1-standard, n2-highmem, a2-highgpu-1g", done: false },
    { label: "Recommendation engine: optimal instance for workload+M", done: false },
    { label: "Cost per 1M paths for each engine × instance type", done: false },
];

const INSTANCES = [
    { name: "n1-standard-4", vcpus: 4, ram: "15 GB", gpu: "—", cost: "$0.19/h", notes: "CPU baseline" },
    { name: "n2-highmem-8", vcpus: 8, ram: "64 GB", gpu: "—", cost: "$0.47/h", notes: "JAX JIT" },
    { name: "a2-highgpu-1g", vcpus: 12, ram: "85 GB", gpu: "A100 40GB", cost: "$3.67/h", notes: "GPU target" },
    { name: "t4 (Colab)", vcpus: 2, ram: "13 GB", gpu: "T4 16GB", cost: "Free", notes: "Dev / test" },
];

export default function CloudPage() {
    return (
        <Box display="flex" flexDirection="column" gap={3}>
            <Box display="flex" alignItems="center" gap={2}>
                <CloudIcon color="primary" fontSize="large" />
                <Box>
                    <Typography variant="h5" fontWeight={600}>
                        Cloud Profiling &amp; Cost Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Google Cloud resource profiler with recommendation engine — Days 10–11
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
                        Target Instance Types (Google Cloud)
                    </Typography>
                    <Divider sx={{ mb: 1 }} />
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                {["Instance", "vCPUs", "RAM", "GPU", "On-demand", "Use case"].map((h) => (
                                    <TableCell key={h} sx={{ fontWeight: 700 }}>{h}</TableCell>
                                ))}
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {INSTANCES.map((r) => (
                                <TableRow key={r.name}>
                                    <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{r.name}</TableCell>
                                    <TableCell>{r.vcpus}</TableCell>
                                    <TableCell>{r.ram}</TableCell>
                                    <TableCell>{r.gpu}</TableCell>
                                    <TableCell>{r.cost}</TableCell>
                                    <TableCell sx={{ color: "text.secondary" }}>{r.notes}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card variant="outlined" sx={{ bgcolor: "grey.50" }}>
                <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Recommendation Logic (Planned)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Given a target workload, M, and time budget: the recommender will query the
                        cost/perf matrix and return the cheapest instance that meets the latency SLA.
                        GPU is recommended when M ≥ 500k and the speedup amortises the instance premium.
                    </Typography>
                </CardContent>
            </Card>
        </Box>
    );
}
