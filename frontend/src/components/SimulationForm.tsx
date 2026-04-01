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
    Typography,
    Collapse,
    Alert,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import type { WorkloadInfo, EngineInfo } from "../api/client";

export interface FormValues {
    workload_type: string;
    engine: string;
    ad_mode: string;
    S0: number;
    K: number;
    r: number;
    sigma: number;
    T: number;
    N: number;
    M: number;
    seed: number;
    option_type: string;
    // asian-specific
    averaging: string;
    // barrier-specific
    B: number;
    barrier_type: string;
    barrier_side: string;
    // basket-specific
    n_assets: number;
    rho: number;
}

const DEFAULTS: FormValues = {
    workload_type: "european",
    engine: "cpu",
    ad_mode: "none",
    S0: 100,
    K: 100,
    r: 0.05,
    sigma: 0.2,
    T: 1.0,
    N: 252,
    M: 10000,
    seed: 42,
    option_type: "call",
    averaging: "arithmetic",
    B: 120,
    barrier_type: "knock_out",
    barrier_side: "up",
    n_assets: 3,
    rho: 0.5,
};

interface Props {
    workloads: Record<string, WorkloadInfo>;
    engines: Record<string, EngineInfo>;
    loading: boolean;
    onSubmit: (values: FormValues) => void;
}

function NumField({
    label,
    name,
    value,
    onChange,
    step = "any",
    min,
    max,
    integer = false,
}: {
    label: string;
    name: keyof FormValues;
    value: number;
    onChange: (name: keyof FormValues, val: number) => void;
    step?: string | number;
    min?: number;
    max?: number;
    integer?: boolean;
}) {
    return (
        <TextField
            label={label}
            type="number"
            size="small"
            fullWidth
            value={value}
            inputProps={{ step: integer ? 1 : step, min, max }}
            onChange={(e) => {
                const parsed = integer ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
                if (!isNaN(parsed)) onChange(name, parsed);
            }}
        />
    );
}

export default function SimulationForm({ workloads, engines, loading, onSubmit }: Props) {
    const [values, setValues] = useState<FormValues>(DEFAULTS);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [validationError, setValidationError] = useState<string | null>(null);

    // When workload changes, reset N to a sensible default
    useEffect(() => {
        if (values.workload_type === "european") {
            setValues((v) => ({ ...v, N: 1 }));
        } else if (values.workload_type === "basket") {
            setValues((v) => ({ ...v, N: 52 }));
        } else {
            setValues((v) => ({ ...v, N: 252 }));
        }
    }, [values.workload_type]);

    function set<K extends keyof FormValues>(name: K, val: FormValues[K]) {
        setValues((prev) => ({ ...prev, [name]: val }));
        setValidationError(null);
    }

    function validate(): string | null {
        if (values.S0 <= 0) return "S0 must be positive";
        if (values.K <= 0) return "K must be positive";
        if (values.sigma <= 0) return "σ must be positive";
        if (values.T <= 0) return "T must be positive";
        if (values.M < 100) return "M (paths) must be at least 100";
        if (values.workload_type === "barrier" && values.B <= 0) return "Barrier B must be positive";
        return null;
    }

    function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        const err = validate();
        if (err) {
            setValidationError(err);
            return;
        }
        onSubmit(values);
    }

    const workloadOptions = Object.keys(workloads).length
        ? Object.keys(workloads)
        : ["european", "asian", "barrier", "basket"];

    const engineOptions = Object.keys(engines).length
        ? Object.keys(engines)
        : ["cpu", "jax"];

    const isPathDependent = values.workload_type !== "european";

    return (
        <Card variant="outlined">
            <CardHeader title="Simulation Parameters" />
            <Divider />
            <CardContent>
                <form onSubmit={handleSubmit}>
                    <Grid container spacing={2}>
                        {/* Row 1: Workload + Engine + AD mode */}
                        <Grid item xs={12} sm={4}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Workload</InputLabel>
                                <Select
                                    label="Workload"
                                    value={values.workload_type}
                                    onChange={(e) => set("workload_type", e.target.value)}
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
                                    value={values.engine}
                                    onChange={(e) => set("engine", e.target.value)}
                                >
                                    {engineOptions.map((eng) => (
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
                                    value={values.ad_mode}
                                    onChange={(e) => set("ad_mode", e.target.value)}
                                >
                                    <MenuItem value="none">None</MenuItem>
                                    <MenuItem value="forward">Forward</MenuItem>
                                    <MenuItem value="reverse">Reverse</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>

                        {/* Row 2: Core numerical params */}
                        <Grid item xs={6} sm={3}>
                            <NumField label="S₀ (Initial Price)" name="S0" value={values.S0} onChange={set} min={0.01} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                            <NumField label="K (Strike)" name="K" value={values.K} onChange={set} min={0.01} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                            <NumField label="r (Rate)" name="r" value={values.r} onChange={set} step={0.01} min={0} max={1} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                            <NumField label="σ (Volatility)" name="sigma" value={values.sigma} onChange={set} step={0.01} min={0} max={1} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                            <NumField label="T (Maturity, yr)" name="T" value={values.T} onChange={set} step={0.25} min={0.01} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                            <NumField label="M (Paths)" name="M" value={values.M} onChange={set} integer min={100} />
                        </Grid>
                        {isPathDependent && (
                            <Grid item xs={6} sm={3}>
                                <NumField label="N (Time Steps)" name="N" value={values.N} onChange={set} integer min={2} />
                            </Grid>
                        )}
                        <Grid item xs={6} sm={3}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Option Type</InputLabel>
                                <Select
                                    label="Option Type"
                                    value={values.option_type}
                                    onChange={(e) => set("option_type", e.target.value)}
                                >
                                    <MenuItem value="call">Call</MenuItem>
                                    <MenuItem value="put">Put</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>

                        {/* Workload-specific fields */}
                        {values.workload_type === "asian" && (
                            <Grid item xs={12} sm={4}>
                                <FormControl size="small" fullWidth>
                                    <InputLabel>Averaging</InputLabel>
                                    <Select
                                        label="Averaging"
                                        value={values.averaging}
                                        onChange={(e) => set("averaging", e.target.value)}
                                    >
                                        <MenuItem value="arithmetic">Arithmetic</MenuItem>
                                        <MenuItem value="geometric">Geometric</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                        )}

                        {values.workload_type === "barrier" && (
                            <>
                                <Grid item xs={6} sm={3}>
                                    <NumField label="B (Barrier)" name="B" value={values.B} onChange={set} min={0.01} />
                                </Grid>
                                <Grid item xs={6} sm={3}>
                                    <FormControl size="small" fullWidth>
                                        <InputLabel>Barrier Type</InputLabel>
                                        <Select
                                            label="Barrier Type"
                                            value={values.barrier_type}
                                            onChange={(e) => set("barrier_type", e.target.value)}
                                        >
                                            <MenuItem value="knock_out">Knock-Out</MenuItem>
                                            <MenuItem value="knock_in">Knock-In</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid item xs={6} sm={3}>
                                    <FormControl size="small" fullWidth>
                                        <InputLabel>Barrier Side</InputLabel>
                                        <Select
                                            label="Barrier Side"
                                            value={values.barrier_side}
                                            onChange={(e) => set("barrier_side", e.target.value)}
                                        >
                                            <MenuItem value="up">Up</MenuItem>
                                            <MenuItem value="down">Down</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </>
                        )}

                        {values.workload_type === "basket" && (
                            <>
                                <Grid item xs={6} sm={3}>
                                    <NumField label="Assets (n)" name="n_assets" value={values.n_assets} onChange={set} integer min={2} max={10} />
                                </Grid>
                                <Grid item xs={6} sm={3}>
                                    <NumField label="ρ (Correlation)" name="rho" value={values.rho} onChange={set} step={0.05} min={-1} max={1} />
                                </Grid>
                            </>
                        )}

                        {/* Advanced: seed */}
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
                                        <Grid item xs={6} sm={3}>
                                            <NumField label="Seed" name="seed" value={values.seed} onChange={set} integer min={0} />
                                        </Grid>
                                    </Grid>
                                </Box>
                            </Collapse>
                        </Grid>

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
                                startIcon={loading ? <CircularProgress size={18} color="inherit" /> : <PlayArrowIcon />}
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
