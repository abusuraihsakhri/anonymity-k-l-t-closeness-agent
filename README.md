# Anonymity K L T Closeness Agent

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

**Anonymity K L T Closeness Agent** is an advanced analytical and computational platform implementing k-Anonymity, l-diversity & Earth Mover's Distance t-closeness privacy guard.

K-anonymity / l-diversity / t-closeness enrichment features for
anonymity-k-l-t-closeness-agent.

Implements the top three items from specifications as a working engine over a
record list with quasi-identifiers (age, zipcode) and sensitive attributes:

1. K-anonymity generalization and verification: age bucketing and zip prefix
   truncation produce equivalence classes; each class must hold >= k records.
2. L-diversity verification: every equivalence class must contain >= l
   distinct values of the chosen sensitive attribute.
3. T-closeness audit via Earth Mover's Distance between each class's
   sensitive-value distribution and the overall distribution (numeric EMD by
   CDF integration; categorical total-variation fallback), plus a greedy
   minimal-generalization search that measures information loss.

Author: Dr. Abu Suraih Sakhri
License: MIT

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Core Algorithmic & Evaluation Engines

- **`DatasetRecord`** — dedicated module for dataset record evaluation and state verification.
- **`EquivalenceClass`** — dedicated module for equivalence class evaluation and state verification.
- **`KAnonymityReport`** — dedicated module for k anonymity report evaluation and state verification.
- **`LDiversityReport`** — dedicated module for l diversity report evaluation and state verification.
- **`TClosenessReport`** — dedicated module for t closeness report evaluation and state verification.
- **`PrivacyRiskReport`** — dedicated module for privacy risk report evaluation and state verification.

---

## 📐 Mathematical Formulation & Logic

```text
  - Individual Risk = 1 / |E_i|
  - Average Dataset Risk = (1/N) * sum(1 / |E(r)|)
  - Maximum Risk = max(1 / |E(r)|)
  dist_emd = calculate_categorical_emd(p_dist, q_dist)
  dist_emd = calculate_numerical_emd(p_vals, global_vals)
```

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --audit <value> --interactive <value> --k <value> --l <value>
```

### Parameter Reference
- `--audit`: Specifies input measurement or parameter value.
- `--interactive`: Specifies input measurement or parameter value.
- `--k`: Specifies input measurement or parameter value.
- `--l`: Specifies input measurement or parameter value.
- `--t`: Specifies input measurement or parameter value.
- `--json`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `task_id` | Parameter / observation metric | Required |
| `target_identifier` | Parameter / observation metric | Required |
| `primary_metric` | Parameter / observation metric | Required |
| `secondary_metric` | Parameter / observation metric | Required |
| `is_critical_flag` | Parameter / observation metric | Required |
| `status_descriptor` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t anonymity-k-l-t-closeness-agent .
docker run -p 8000:8000 anonymity-k-l-t-closeness-agent
```
