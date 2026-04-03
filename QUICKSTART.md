# Quick Start

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run A Benchmark

```bash
python experiments/run_experiment.py
```

This writes JSON output to `results/`.

## 3. Generate Multiple Runs (optional)

```bash
python experiments/generate_runs.py
```

## 4. Run Frontend

React UI + API:

Terminal 1:

```bash
python -m benchmarking.api.server --port 5050
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## 5. Optional C++ Engine

```bash
cd benchmarking/cpp
pip install -e .
cd ../..
```

Then use engine `cpp` via the API/UI.

## 6. Optional: Run Tests

```bash
pytest -q
```

For full project details, see `README.md`.