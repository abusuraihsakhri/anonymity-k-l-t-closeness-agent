# Anonymity Guard: k-Anonymity, l-Diversity & t-Closeness Engine

A pure Python privacy-preserving data anonymization, re-identification risk auditing, and multidimensional generalization engine implementing:
- **$k$-Anonymity Verification:** Groups records into equivalence classes sharing identical quasi-identifiers (e.g. age bracket, ZIP code prefix, gender) ensuring each group size $|E| \ge k$.
- **Distinct & Entropy $l$-Diversity:** Guards against attribute disclosure by enforcing at least $l$ distinct sensitive values per equivalence class ($-\sum p_i \ln p_i \ge \ln l$).
- **$t$-Closeness via Earth Mover's Distance (EMD):** Measures distance between the intra-class sensitive distribution ($P$) and the overall population distribution ($Q$) such that $D[P, Q] \le t$.
  - Continuous/ordered attributes: Kolmogorov-Smirnov cumulative distribution metric:
    $$D[P, Q] = \int |F_P(x) - F_Q(x)| dx$$
  - Categorical attributes: Total variation distance / variation metric:
    $$D[P, Q] = \frac{1}{2} \sum_{s \in S} |p_s - q_s|$$
- **Re-identification Risk Profiling:** Calculates individual risk ($1/|E_i|$), prosecutor risk ($\max(1/|E_i|)$), marketer risk (sample average), and journalist risk percentage.
- **Top-Down Mondrian Anonymization:** Multidimensional greedy spatial partitioning minimizing Normalized Certainty Penalty (NCP) information loss.
- **High-Throughput Batch CSV Cohort Auditing:** Audits research and registry data pipelines for HIPAA Safe Harbor and Expert Determination privacy standards.

Requires Python standard library only (zero external runtime dependencies).

---

## Privacy Formulations & Mathematical Logic

### $k$-Anonymity
$$\forall E \in \mathcal{E}, \quad |E| \ge k$$

### Distinct $l$-Diversity
$$\forall E \in \mathcal{E}, \quad |\{s \in S : \exists r \in E, r.S = s\}| \ge l$$

### Earth Mover's Distance $t$-Closeness
$$D[P_E, Q] \le t \quad \forall E \in \mathcal{E}$$

### Normalized Certainty Penalty (Information Loss)
$$NCP(r) = \sum_{i=1}^d \frac{|v_i^{max} - v_i^{min}|}{|D_i^{max} - D_i^{min}|}$$

---

## Features

- **HIPAA Safe Harbor & Expert Determination:** Quantifies re-identification risks to satisfy statutory de-identification requirements.
- **Skewness & Similarity Attack Defense:** Combines $l$-diversity with $t$-closeness to eliminate probabilistic inference breaches.
- **Mondrian Anonymizer:** Built-in multidimensional recursive partitioning algorithm.
- **Batch CSV Processing:** High-throughput batch auditing for clinical and demographic data files.

---

## Installation & Requirements

- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- Zero external runtime dependencies. `pytest` is optional for running tests.

```bash
git clone https://github.com/abusuraihsakhri/anonymity-k-l-t-closeness-agent.git
cd anonymity-k-l-t-closeness-agent
```

---

## CLI Usage

### 1. Run Complete Privacy Audit on Benchmark Cohort
```bash
python cli.py --audit --json
```

### 2. Audit with Custom Privacy Thresholds ($k=5, l=3, t=0.25$)
```bash
python cli.py --audit --k 5 --l 3 --t 0.25
```

### 3. Batch CSV Processing
```bash
python cli.py --batch sample.csv results.csv
```

---

## Python API Quickstart

```python
from anonymity_klt import (
    AnonymityGuardPipeline,
    DatasetRecord,
    MondrianAnonymizer,
)

# 1. Initialize Pipeline
pipeline = AnonymityGuardPipeline()

# 2. Build Records with Generalized Attributes
records = [
    DatasetRecord(1, {"age_group": "20-29", "zip3": "100**", "gender": "Female", "diagnosis": "Asthma"}),
    DatasetRecord(2, {"age_group": "20-29", "zip3": "100**", "gender": "Female", "diagnosis": "Diabetes"}),
    DatasetRecord(3, {"age_group": "20-29", "zip3": "100**", "gender": "Female", "diagnosis": "Hypertension"}),
]

# 3. Audit Dataset
report = pipeline.audit_dataset(
    records=records,
    quasi_identifiers=["age_group", "zip3", "gender"],
    sensitive_attribute="diagnosis",
    k=3,
    l=2,
    t=0.35,
)

print(f"Dataset k-Anonymous: {report['k_anonymity']['is_k_anonymous']}")
print(f"Dataset l-Diverse: {report['distinct_l_diversity']['is_l_diverse']}")
print(f"Dataset t-Close: {report['t_closeness']['is_t_close']}")
print(f"Prosecutor Risk: {report['privacy_risks']['prosecutor_risk_max']}")
```

---

## Running Tests

Run the test suite using standard `unittest` or `pytest`:

```bash
pytest -v
```

