import axios from "axios";

const client = axios.create({
    baseURL: "/api",
    headers: { "Content-Type": "application/json" },
    timeout: 120_000, // runs can take a while
});

export interface SimulationRequest {
    workload_type: string;
    engine: string;
    ad_mode: string;
    config: Record<string, number | string>;
}

export interface RunStatus {
    id: string;
    status: "pending" | "running" | "completed" | "failed";
    workload_type: string;
    engine: string;
    ad_mode: string;
    config: Record<string, unknown>;
    result_value: number | null;
    mean_runtime_ms: number | null;
    std_runtime_ms: number | null;
    ad_overhead_ratio: number | null;
    error_message: string | null;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
}

export interface WorkloadInfo {
    label: string;
    workload_type: string;
    schema: SchemaField[];
}

export interface SchemaField {
    key: string;
    label: string;
    type: "number" | "integer" | "select";
    default: number | string;
    min?: number;
    max?: number;
    options?: string[];
}

export interface EngineInfo {
    name: string;
    supported_workloads: string[];
}

export async function fetchWorkloads(): Promise<Record<string, WorkloadInfo>> {
    const res = await client.get<Record<string, WorkloadInfo>>("/workloads");
    return res.data;
}

export async function fetchEngines(): Promise<Record<string, EngineInfo>> {
    const res = await client.get<Record<string, EngineInfo>>("/engines");
    return res.data;
}

export async function submitRun(
    payload: SimulationRequest
): Promise<{ id: string; status: string }> {
    const res = await client.post<{ id: string; status: string }>("/runs", payload);
    return res.data;
}

export async function pollRun(runId: string): Promise<RunStatus> {
    const res = await client.get<RunStatus>(`/runs/${runId}`);
    return res.data;
}

export async function fetchRuns(limit = 50): Promise<RunStatus[]> {
    const res = await client.get<RunStatus[]>(`/runs?limit=${limit}`);
    return res.data;
}

export interface SummaryResponse {
    total: number;
    completed: number;
    pending: number;
    failed: number;
    fastest_ms: number | null;
    by_workload: Record<string, number>;
    by_engine: Record<string, number>;
}

export async function fetchSummary(): Promise<SummaryResponse> {
    const res = await client.get<SummaryResponse>("/summary");
    return res.data;
}
