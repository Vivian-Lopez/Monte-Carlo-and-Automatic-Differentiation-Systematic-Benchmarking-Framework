/**
 * Shared display-formatting utilities.
 * All run table cells, result panels, and charts should use these functions
 * so column values are consistent across the entire UI.
 */

export const WORKLOAD_LABELS: Record<string, string> = {
    european: "European",
    asian: "Asian",
    barrier: "Barrier",
    basket: "Basket",
};

/** "european" → "European" */
export function fmtWorkload(w: string): string {
    return WORKLOAD_LABELS[w] ?? w;
}

/** "cpu" → "CPU" */
export function fmtEngine(e: string): string {
    return e.toUpperCase();
}

/** 10.3452 → "$10.3452" */
export function fmtPrice(v: number | null | undefined): string {
    return v != null ? `$${v.toFixed(4)}` : "—";
}

/** 0.193 → "0.19" (always 2 d.p., never "0.0") */
export function fmtMs(v: number | null | undefined): string {
    return v != null ? v.toFixed(2) : "—";
}

/** 1.07 → "1.07×", but "—" when ad_mode is "none" or value is null */
export function fmtOverhead(v: number | null | undefined, adMode: string): string {
    return v != null && adMode !== "none" ? `${v.toFixed(2)}×` : "—";
}

/** ISO-8601 UTC string → locale datetime, e.g. "10/04/2026, 15:27:08" */
export function fmtDatetime(d: string | null | undefined): string {
    return d ? new Date(d).toLocaleString() : "—";
}
