# Anonymity Guard
*k-Anonymity, l-Diversity, and t-Closeness Privacy Engineering Engine*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![Tests: 100% Pass](https://img.shields.io/badge/Tests-22%20Passed-success.svg)]()

Anonymity Guard is a production-grade privacy preservation and de-identification auditing suite for healthcare and sensitive tabular microdata. It implements exact mathematical formulations for **$k$-Anonymity**, **$l$-Diversity** (Distinct, Entropy, and Recursive $(c,l)$ models), **$t$-Closeness** (Earth Mover's Distance for categorical and numerical attributes), **Mondrian multidimensional generalization**, and formal **re-identification disclosure risk modeling** (Marketer, Prosecutor, and Journalist risk).

---

## Privacy Theory & Mathematical Formulations

### 1. $k$-Anonymity Model
Guarantees that each released record is indistinguishable from at least $k-1$ other records with respect to the Quasi-Identifiers ($\text{QIs}$).
$$\forall E_i \in \mathcal{P}(\mathcal{D}): |E_i| \ge k$$
Where $E_i$ is an equivalence class sharing identical quasi-identifier values.

---

### 2. $l$-Diversity Frameworks
Protects against attribute disclosure and homogeneity attacks within an equivalence class:

- **Distinct $l$-Diversity**: Each equivalence class contains at least $l$ distinct sensitive attribute values:
  $$|\text{Distinct}(S_i)| \ge l$$
- **Entropy $l$-Diversity**: The entropy of the sensitive attribute distribution within class $E_i$ is at least $\ln(l)$:
  $$H(E_i) = -\sum_{s \in S} p(s) \ln p(s) \ge \ln(l)$$
- **Recursive $(c, l)$-Diversity**: Prevents the most frequent sensitive state ($r_1$) from dominating the remaining $m - l + 1$ less frequent values:
  $$r_1 < c \sum_{j=l}^{m} r_j$$

---

### 3. $t$-Closeness Framework (Earth Mover's Distance)
Ensures the distance between the marginal distribution of the sensitive attribute in equivalence class $E_i$ ($P$) and the entire global dataset ($Q$) does not exceed $t$:
$$D(P, Q) \le t$$

- **Categorical Attributes (Total Variation Distance / Equal EMD)**:
  $$D(P, Q) = \frac{1}{2} \sum_{s \in \mathcal{S}} |P(s) - Q(s)| \le t$$
- **Numerical Attributes (1-D Wasserstein Distance via Normalized Integrated CDF)**:
  $$D(P, Q) = \int_0^1 |F_P(x) - F_Q(x)| \, dx \le t$$

---

### 4. Re-identification & Disclosure Risk Metrics
- **Prosecutor Risk (Maximum Individual Risk)**:
  $$\text{Risk}_{\text{prosecutor}} = \max_{r \in \mathcal{D}} \frac{1}{|E(r)|}$$
- **Marketer Risk (Average Population Risk)**:
  $$\text{Risk}_{\text{marketer}} = \frac{1}{|\mathcal{D}|} \sum_{r \in \mathcal{D}} \frac{1}{|E(r)|}$$
- **Journalist Risk**: Proportion of records with disclosure probability above threshold ($\ge 50\%$).
- **Normalized Certainty Penalty (NCP)**: Quantifies information loss across multidimensional generalization.

---

## CLI Usage

The command-line interface supports automated privacy audits, Mondrian anonymization runs, risk scoring, and interactive query modes.

### 1. Run Complete Privacy Audit on Benchmark Dataset
```bash
python cli.py --audit --k 3 --l 2 --t 0.35
```
Output:
```text
============================================================================
      ANONYMITY GUARD - PRIVACY PRESERVATION AUDIT REPORT            
============================================================================
 Overall Compliance Status : COMPLIANT
 Quasi-Identifiers         : age_group, zip3, gender
 Sensitive Attribute       : diagnosis
----------------------------------------------------------------------------
 1. k-ANONYMITY AUDIT (Target k=3):
----------------------------------------------------------------------------
  Satisfied              : True
  Total Equivalence Cls  : 4 (Class Sizes: Min=3, Avg=3.0, Max=3)
  Violating Classes      : 0
----------------------------------------------------------------------------
 2. l-DIVERSITY AUDIT (Target l=2):
----------------------------------------------------------------------------
  Distinct l-Diversity   : True (0 violations)
  Entropy l-Diversity    : True (0 violations)
----------------------------------------------------------------------------
 3. t-CLOSENESS AUDIT (Target t=0.35):
----------------------------------------------------------------------------
  Satisfied              : True
  Max EMD Distance       : 0.3333 (Avg: 0.3333)
  Violating Classes      : 0
----------------------------------------------------------------------------
 4. RE-IDENTIFICATION RISK & DISCLOSURE METRICS:
----------------------------------------------------------------------------
  Marketer Risk (Avg)    : 0.3333 (33.3%)
  Prosecutor Risk (Max)  : 0.3333 (33.3%)
  Journalist Risk (>50%) : 0.0% (0 records)
============================================================================
```

### 2. Export Audit Results in JSON Format
```bash
python cli.py --audit --json
```

### 3. Launch Interactive Privacy Terminal
```bash
python cli.py --interactive
```

---

## Test Suite Execution

Run the complete 22-test suite with pure Python standard library:

```bash
python -m unittest test_anonymity_klt.py
python -m unittest discover -s tests
```

---

## License
MIT License. Developed for privacy-preserving data science and HIPAA de-identification research.
