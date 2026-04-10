import React, { useState, useEffect } from "react";
import {
    Card,
    CardContent,
    CardHeader,
    Grid,
    TextField,
    MenuItem,
    Button,
    Divider,
    CircularProgress,
    FormControl,
    InputLabel,
    Select,
    Box,
    Collapse,
    Alert,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import type { WorkloadInfo, EngineInfo, SchemaField } from "../api/client";

export interface FormValues {
    workload_type: string;
    engine: string;
    ad_mode: string;
    config: Record<string, number | string>;
}

function buildConfigDefaults(schema: SchemaField[]): Record<string, number | string> {
    const out: Record<string, number | string> = {};
    for (const f of schema) {
        out[f.key] = f.default;
    }
    return out;
}

function validateConfig(
    schema: SchemaField[],
    config: Record<string, number | string>
): string | null {
    for (const f of schema) {
        const val = config[f.key];
        if (typeof val === "number") {
            if (f.min !== undefined && val < f.min) return `${f.label} must be ≥ ${f.min}`;
            if (f.max !== undefined && val > f.max) return `${f.label} must be ≤ ${f.max}`;
        }
    }
    return null;
}

interface Props {
    workloads: Record<string, WorkloadInfo>;
    engines: Record<string, EngineInfo>;
    loading: boolean;
    onSubmit: (values: FormValues) => void;
}

export default function SimulationForm({ workloads, engines, loading, onSubmit }: Props) {
    const [workloadType, setWorkloadType] = useState("european");
    const [engine, setEngine] = useState("cpu");
    const [adMode, setAdMode] = useState("none");
    const [config, setConfig] = useState<Record<string, number | string>>({});
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [validationError, setValidationError] = useState<string | null>(null);

    // Reset config to schema defaults whenever workload type changes or schema first loads
    useEffect(() => {
        const schema = workloads[workloadType]?.schema ?? [];
        setConfig(buildConfigDefaults(schema));
        setValidationError(null);
    }, [workloadType, workloads]);

    // Auto-switch engine when the current one doesn't support the new workload
    useEffect(() => {
        if (Object.keys(engines).length === 0) return;
        const supported = Object.keys(engines).filter(
            (eng) => engines[eng].supported_workloads.includes(workloadType)
        );
        if (supported.length > 0 && !supported.includes(engine)) {
            setEngine(supported[0]);
        }
    }, [workloadType, engines, engine]);

    const schema = workloads[workloadType]?.schema ?? [];
    const mainFields = schema.filter((f) => f.key !== "seed");
    const advancedFields = schema.filter((f) => f.key === "seed");

    // Only show engines that support the current workload
    const availableEngines = Object.keys(engines).length
        ? Object.keys(engines).filter((eng) =>
            engines[eng].supported_workloads.includes(workloadType)
        )
        : ["cpu", "jax"];

    function setConfigField(key: string, val: number | string) {
        setConfig((prev) => ({ ...prev, [key]: val }));
        setValidationError(null);
    }

    function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        const err = validateConfig(schema, config);
        if (err) {
            setValidationError(err);
            return;
        }
        onSubmit({ workload_type: workloadType, engine, ad_mode: adMode, config });
    }

    const workloadOptions = Object.keys(workloads).length
        ? Object.keys(workloads)
        : ["european", "asian", "barrier", "basket"];

    function renderField(f: SchemaField) {
        const val = config[f.key];

        if (f.type === "select") {
            return (
                <Grid item xs={12} sm={4} key={f.key}>
                    <FormControl size="small" fullWidth>
                        <InputLabel>{f.label}</InputLabel>
                        <Select
                            label={f.label}
                            value={String(val ?? f.default)}
                            onChange={(e) => setConfigField(f.key, e.target.value)}
                        >
                            {(f.options ?? []).map((opt) => (
                                <MenuItem key={opt} value={opt}>
                                    {opt.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Grid>
            );
        }

        // "number" or "integer"
        const isInt = f.type === "integer";
        return (
            <Grid item xs={6} sm={3} key={f.key}>
                <TextField
                    label={f.label}
                    type="number"
                    size="small"
                    fullWidth
                    value={val ?? f.default}
                    inputProps={{ step: isInt ? 1 : "any", min: f.min, max: f.max }}
                    onChange={(e) => {
                        const raw = isInt
                            ? parseInt(e.target.value, 10)
                            : parseFloat(e.target.value);
                        if (!isNaN(raw)) setConfigField(f.key, raw);
                    }}
                />
            </Grid>
        );
    }

    return (
        <Card variant="outlined">
            <CardHeader title="Simulation Parameters" />
            <Divider />
            <CardContent>
                <form onSubmit={handleSubmit}>
                    <Grid container spacing={2}>
                        {/* Workload / Engine / AD mode */}
                        <Grid item xs={12} sm={4}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Workload</InputLabel>
                                <Select
                                    label="Workload"
                                    value={workloadType}
                                    onChange={(e) => setWorkloadType(e.target.value)}
                                >
                                    {workloadOptions.map((w) => (
                                        <MenuItem key={w} value={w}>
                                            {workloads[w]?.label ?? w.charAt(0).toUpperCase() + w.slice(1)}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Engine</InputLabel>
                                <Select
                                    label="Engine"
                                    value={availableEngines.includes(engine) ? engine : (availableEngines[0] ?? "")}
                                    onChange={(e) => setEngine(e.target.value)}
                                >
                                    {availableEngines.map((eng) => (
                                        <MenuItem key={eng} value={eng}>
                                            {eng.toUpperCase()}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>AD Mode</InputLabel>
                                <Select
                                    label="AD Mode"
                                    value={adMode}
                                    onChange={(e) => setAdMode(e.target.value)}
                                >
                                    <MenuItem value="none">None</MenuItem>
                                    <MenuItem value="forward">Forward</MenuItem>
                                    <MenuItem value="reverse">Reverse</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>

                        {/* Schema-driven workload parameters */}
                        {mainFields.map(renderField)}

                        {/* Advanced: seed */}
                        {advancedFields.length > 0 && (
                            <Grid item xs={12}>
                                <Button
                                    size="small"
                                    variant="text"
                                    color="inherit"
                                    endIcon={showAdvanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                                    onClick={() => setShowAdvanced((v) => !v)}
                                    sx={{ textTransform: "none", color: "text.secondary" }}
                                >
                                    Advanced
                                </Button>
                                <Collapse in={showAdvanced}>
                                    <Box mt={1}>
                                        <Grid container spacing={2}>
                                            {advancedFields.map(renderField)}
                                        </Grid>
                                    </Box>
                                </Collapse>
                            </Grid>
                        )}

                        {validationError && (
                            <Grid item xs={12}>
                                <Alert severity="error" onClose={() => setValidationError(null)}>
                                    {validationError}
                                </Alert>
                            </Grid>
                        )}

                        {/* Submit */}
                        <Grid item xs={12}>
                            <Button
                                type="submit"
                                variant="contained"
                                size="large"
                                fullWidth
                                disabled={loading}
                                startIcon={
                                    loading ? (
                                        <CircularProgress size={18} color="inherit" />
                                    ) : (
                                        <PlayArrowIcon />
                                    )
                                }
                            >
                                {loading ? "Running…" : "Run Simulation"}
                            </Button>
                        </Grid>
                    </Grid>
                </form>
            </CardContent>
        </Card>
    );
}
