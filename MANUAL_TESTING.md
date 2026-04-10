# Manual Testing Guide

This document provides a comprehensive manual testing checklist for the Monte Carlo Benchmarking Engine. Test both the backend API and frontend UI.

---

## Prerequisites

```bash
# Ensure environment is set up
cd /workspaces/Systematic_Monte_Carlo_Benchmarking_Engine
source venv/bin/activate
pip install -r requirements.txt

# Optional: rebuild C++ engine if testing it
cd benchmarking/cpp && pip install -e . && cd ../..
```

---

## Part 1: Backend Startup & API Testing

### 1.1 Start the Flask Server

```bash
# Terminal 1
cd /workspaces/Systematic_Monte_Carlo_Benchmarking_Engine
source venv/bin/activate
python -m benchmarking.api.server
```

**Expected**: Server starts on `http://localhost:5050`
```
* Running on http://127.0.0.1:5050 (Press CTRL+C to quit)
```

### 1.2 Test API Endpoints (use `curl` or Postman)

#### Test: GET /api/workloads
```bash
curl -s http://localhost:5050/api/workloads | jq .
```
**Expected**: Returns JSON with 4 workload types (european, asian, barrier, basket) plus their parameter schemas.

#### Test: GET /api/engines
```bash
curl -s http://localhost:5050/api/engines | jq .
```
**Expected**: Returns `{"cpu": {...}, "jax": {...}, "cpp": {...}}` (cpp may be missing if not built).

#### Test: GET /api/summary
```bash
curl -s http://localhost:5050/api/summary | jq .
```
**Expected**: Returns JSON with `total`, `completed`, `pending`, `failed`, `fastest_ms`, and breakdowns by workload/engine.

#### Test: POST /api/runs (Submit Simulation)
```bash
curl -X POST http://localhost:5050/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "european",
    "engine": "jax",
    "ad_mode": "reverse",
    "config": {
      "S0": 100,
      "K": 100,
      "r": 0.05,
      "sigma": 0.2,
      "T": 1.0,
      "M": 10000,
      "seed": 42,
      "option_type": "call"
    }
  }' | jq .
```
**Expected**: Returns `{"id": "<uuid>", "status": "pending"}`. **Save the `id` for next step.**

#### Test: GET /api/runs/<id> (Poll for Results)
```bash
# Replace <id> with the UUID from previous step
curl -s http://localhost:5050/api/runs/<id> | jq .
```

**Poll until `status == "completed"`** (should take 5-15 seconds for JAX). 

**When complete, verify response contains**:
- `result_value`: ~10.45 (European call, ATM)
- `mean_runtime_ms`: > 0
- `std_runtime_ms`: >= 0
- `ad_overhead_ratio`: >= 1.0 (should be ~5-20× for AD)
- `greeks`: An object with `delta`, `vega`, `rho` keys
  - `delta`: ~0.63 (for ATM call)
  - `vega`: > 0
  - `rho`: > 0

#### Test: GET /api/runs?limit=10 (List Runs)
```bash
curl -s "http://localhost:5050/api/runs?limit=10" | jq .
```
**Expected**: Array of runs (most recent first), each with all fields including `greeks`.

#### Test: GET /api/runs with Filters
```bash
# Filter by workload
curl -s "http://localhost:5050/api/runs?workload=european" | jq .

# Filter by engine  
curl -s "http://localhost:5050/api/runs?engine=jax" | jq .

# Filter by status
curl -s "http://localhost:5050/api/runs?status=completed" | jq .

# Combine filters
curl -s "http://localhost:5050/api/runs?workload=european&engine=jax&status=completed" | jq .
```
**Expected**: Arrays filtered by those criteria.

---

## Part 2: Frontend Startup & UI Testing

### 2.1 Start Frontend Dev Server

```bash
# Terminal 2
cd /workspaces/Systematic_Monte_Carlo_Benchmarking_Engine/frontend
npm install  # (if needed)
npm run dev
```

**Expected**: Vite dev server on `http://localhost:5173`
```
VITE v... ready in ... ms
➜  Local:   http://localhost:5173/
```

### 2.2 Test Run Simulation Page

1. Open `http://localhost:5173/` in browser
2. **Verify UI loads** — no console errors, all pages visible in sidebar
3. **Navigate to "Run Simulation"** tab
4. **Fill form**:
   - Workload: **European**
   - Engine: **JAX** (verify only JAX shown, not CPU; JAX supports AD)
   - AD Mode: **Reverse**
   - Parameters: Keep defaults or adjust (S0=100, K=100, r=0.05, sigma=0.2, T=1, M=10000, seed=42)
5. **Click Submit**
6. **Verify polling**:
   - Status updates "Waiting in queue..." → "Simulation running..." → "Completed"
   - Takes ~10-20 seconds
7. **Verify Results Panel** shows:
   - **Chips**: workload (european), engine (JAX), AD mode (reverse)
   - **Option Price**: ~10.45
   - **Mean Runtime**: > 0 (likely 1-5 ms)
   - **Std Dev**: > 0
   - **AD Overhead**: ~5-20× (ratio of AD time vs no-AD)
   - **Greeks Section**: Shows delta, rho, vega with numerical values
     - Delta: ~0.63
     - Vega: ~37-40
     - Rho: ~45-50

### 2.3 Test AD Modes

#### Forward-Mode AD
1. Submit run with **AD Mode: Forward**
2. When complete, verify:
   - Greeks display (delta, vega, rho)
   - Greeks values match reverse-mode (within 0.5% tolerance)
   - AD Overhead shown and > 1.0

#### No AD
1. Submit run with **AD Mode: None**
2. Verify:
   - **Greeks section NOT displayed** (only None mode computed no Greeks)
   - AD Overhead chip NOT shown
   - Result still shows price, runtime, std dev

### 2.4 Test Different Workloads

#### European
1. Select workload: **European**
2. Verify engines shown: CPU, JAX, (CPU)
3. Submit with JAX
4. Verify completes and shows Greeks

#### Asian
1. Select workload: **Asian**
2. Verify engine dropdown shows: CPU only (JAX doesn't support)
3. Submit with CPU
4. Verify completes (no Greeks displayed, workload doesn't support AD)

#### Barrier
1. Select workload: **Barrier**
2. Verify engine dropdown shows: CPU only
3. Form should show barrier parameters (H, barrier_type)
4. Submit and verify completes

#### Basket
1. Select workload: **Basket**
2. Form should show: n_assets, correlation
3. Submit and verify completes

### 2.5 Test Run History Page

1. Navigate to "Run History" tab
2. **Verify table displays**:
   - ID (first 8 chars)
   - Workload, Engine, AD Mode, Status
   - Price, Mean (ms), **Std (ms)** ← new
   - **AD Overhead** ← new
   - **Error** (if failed)
   - Submitted (full datetime, not time-only)

3. **Submit a few more runs** with different configs
4. **Verify refresh button** updates table
5. **Test filter dropdowns**:
   - Filter by Workload: european → should show only european runs
   - Filter by Engine: jax → should show only JAX runs
   - Filter by Status: completed → should show only completed
   - Combine: european + jax + completed
6. **Verify filtering is immediate** (no manual refresh needed)
7. **Verify table updates and old filter clears** when selecting new filter

### 2.6 Test Summary Page

1. Navigate to "Summary" tab
2. **Verify displays**:
   - Statistics cards (total, completed, pending, failed)
   - Charts render (Runtime by engine, Price by workload)
   - Recent runs table with all columns
3. Run a few simulations with different configs
4. **Refresh page** and verify stats update

---

## Part 3: AD Functionality Deep Dive

### 3.1 Verify Forward vs Reverse AD Agreement

1. **Submit European run with Forward mode**:
   - Config: S0=100, K=100, r=0.05, sigma=0.2, T=1, M=50000, seed=42
   - Note the Greeks values

2. **Submit same config with Reverse mode** (same seed)
   - Greeks should be **nearly identical** (within 0.5%)
   - Both using same seed → same random numbers → same Greeks

3. **In browser console** (F12 → Console):
   ```javascript
   // Manually compare last two runs
   console.log("Forward Greeks:", forwardGreeks);
   console.log("Reverse Greeks:", reverseGreeks);
   ```

### 3.2 Verify Greeks vs Black-Scholes

European option, ATM (S0=K=100, r=0.05, sigma=0.2, T=1):
- **Expected Delta**: ~0.63 (±0.02 for 50k samples)
- **Expected Vega**: ~37-40
- **Expected Rho**: ~45-50

Submit JAX+Reverse with M=100000 (high precision):
- Compare Greeks with expected values (should be within 5%)

### 3.3 Verify No Single Seed Repetition

1. Submit two runs with **same exact config** (same seed)
2. Both should produce **identical prices** and **identical Greeks**
3. Both should have **identical runtimes** (deterministic)

---

## Part 4: Database Verification

### 4.1 Inspect Database Schema

```bash
sqlite3 results/benchmarks.db ".schema runs"
```

**Expected columns** (including new ones):
```
id, workload_type, engine, ad_mode, status, config_json, result_value,
mean_runtime_ms, std_runtime_ms, ad_overhead_ratio, greeks_json,
error_message, created_at, started_at, completed_at
```

### 4.2 Verify Greeks Stored in DB

```bash
# Query a completed run with AD
sqlite3 results/benchmarks.db "SELECT id, ad_mode, greeks_json FROM runs WHERE status='completed' AND ad_mode != 'none' LIMIT 1;"
```

**Expected**: Returns a row with `greeks_json` column populated with JSON like:
```json
{"delta": 0.63, "vega": 37.8, "rho": 45.1}
```

### 4.3 Verify Filter Support

```bash
# Count by workload
sqlite3 results/benchmarks.db "SELECT workload_type, COUNT(*) FROM runs GROUP BY workload_type;"

# Count by AD mode
sqlite3 results/benchmarks.db "SELECT ad_mode, COUNT(*) FROM runs GROUP BY ad_mode;"

# Count completed
sqlite3 results/benchmarks.db "SELECT COUNT(*) FROM runs WHERE status='completed';"
```

---

## Part 5: Error Handling

### 5.1 Test Invalid Config

```bash
curl -X POST http://localhost:5050/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "european",
    "engine": "jax",
    "ad_mode": "reverse",
    "config": {
      "S0": -100,
      "K": 100,
      "r": 0.05,
      "sigma": 0.2,
      "T": 1.0,
      "M": 10000,
      "seed": 42
    }
  }'
```

**Expected**: Returns HTTP 400 with error message (S0 must be > 0).

### 5.2 Test Unsupported Engine/Workload Combo

```bash
curl -X POST http://localhost:5050/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "asian",
    "engine": "jax",
    "ad_mode": "none",
    "config": {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1, "M": 10000, "N": 10, "averaging": "arithmetic", "seed": 42}
  }'
```

**Expected**: Returns HTTP 422 (or similar) — JAX doesn't support Asian.

### 5.3 Test Missing Fields

```bash
curl -X POST http://localhost:5050/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "european",
    "engine": "jax"
  }'
```

**Expected**: HTTP 400 — missing `ad_mode` and `config`.

### 5.4 Frontend Error Display

1. Fill form partially (e.g., leave all workload/engine empty)
2. Try to submit
3. **Verify error alert displays** with helpful message
4. **Verify alert dismissible** (× button works)

---

## Part 6: Stress Test (Optional)

### 6.1 Submit Multiple Sequential Runs

```bash
for i in {1..5}; do
  curl -X POST http://localhost:5050/api/runs \
    -H "Content-Type: application/json" \
    -d '{
      "workload_type": "european",
      "engine": "jax",
      "ad_mode": "reverse",
      "config": {
        "S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0,
        "M": 5000, "seed": '$i'
      }
    }' | jq .id
done
```

**Expected**: Each returns a unique ID, all eventually complete.

### 6.2 Mix Workloads and Engines

```bash
# European + CPU None
curl -X POST http://localhost:5050/api/runs -H "Content-Type: application/json" \
  -d '{"workload_type":"european","engine":"cpu","ad_mode":"none","config":{"S0":100,"K":100,"r":0.05,"sigma":0.2,"T":1,"M":5000,"seed":1}}'

# European + JAX Forward
curl -X POST http://localhost:5050/api/runs -H "Content-Type: application/json" \
  -d '{"workload_type":"european","engine":"jax","ad_mode":"forward","config":{"S0":100,"K":100,"r":0.05,"sigma":0.2,"T":1,"M":5000,"seed":2}}'

# Asian + CPU
curl -X POST http://localhost:5050/api/runs -H "Content-Type: application/json" \
  -d '{"workload_type":"asian","engine":"cpu","ad_mode":"none","config":{"S0":100,"K":100,"r":0.05,"sigma":0.2,"T":1,"M":5000,"N":10,"averaging":"arithmetic","seed":3}}'
```

Monitor Frontend History page — all should appear and complete.

---

## Part 7: Checklist Summary

- [ ] Backend server starts without errors
- [ ] All API workloads/engines endpoints return correct data
- [ ] Submit run via API, poll results, verify greeks_json
- [ ] GET /api/runs filters work (workload, engine, status)
- [ ] Frontend loads at localhost:5173
- [ ] Submit European run with JAX+Reverse
- [ ] Results panel shows workload/engine/AD chips
- [ ] Results panel displays Greeks (delta, vega, rho)
- [ ] Results panel shows AD Overhead for AD runs
- [ ] Forward-mode and Reverse-mode agree (within ~0.5%)
- [ ] No-AD mode does NOT show Greeks
- [ ] Asian workload only shows CPU engine
- [ ] Barrier/Basket workloads work
- [ ] Run History table shows Std, Overhead, Error columns
- [ ] Run History full datetime (not time-only)
- [ ] Run History filter dropdowns filter immediately
- [ ] Summary page loads and displays charts
- [ ] Invalid configs rejected with error message
- [ ] Error alerts in frontend are dismissible
- [ ] Database greeks_json column populated correctly
- [ ] Multiple runs with different configs all work

---

## Troubleshooting

### Backend Won't Start
```bash
# Check port 5050 in use
lsof -i :5050

# Or use different port
python -m benchmarking.api.server --port 5051
```

### Frontend Can't Connect to Backend
1. Ensure Flask server is running on 5050
2. Check browser console for CORS errors—likely backend not running
3. Verify proxy in `frontend/vite.config.ts` points to `5050`

### Greeks Not Displaying
1. Verify `ad_mode` is not `"none"`
2. Check backend logs for errors during AD run
3. Inspect HTML: right-click → Inspect → verify `greeks` in JSON response

### Runs Stuck on "Running"
1. Check backend logs for exceptions
2. Restart Flask server
3. Check if JAX is properly installed: `python -c "import jax; print(jax.__version__)"`

### Database Issues
```bash
# Backup and reset DB
mv results/benchmarks.db results/benchmarks.db.bak
# Next server start will create fresh schema
python -m benchmarking.api.server
```

---

## Additional Notes

- **Determinism**: Same seed always produces same price and Greeks
- **AD Overhead**: Forward and reverse both incur 5-20× slowdown vs no-AD
- **Greeks Precision**: Improves with larger M (samples), ~±2-5% error at M=10000
- **Performance**: No-AD CPU is fastest (~0.1 ms), JAX no-AD ~0.3 ms, with AD ~3-5 ms
