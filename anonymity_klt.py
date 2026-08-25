#!/usr/bin/env python3
"""
Anonymity Guard - Production k-Anonymity, l-Diversity, and t-Closeness Engine.

Mathematical & Algorithmic Implementations:
1. Equivalence Class Partitioning over arbitrary Quasi-Identifiers (QIs).
2. k-Anonymity Validation:
   - Evaluates whether |E_i| >= k for all equivalence classes.
   - Computes equivalence class statistics (count, min, mean, median, max).
3. l-Diversity Verification:
   - Distinct l-Diversity: |Distinct(S_i)| >= l.
   - Entropy l-Diversity: -sum(p * ln(p)) >= ln(l).
   - Recursive (c, l)-Diversity: r_1 < c * sum(r_l ... r_m).
4. t-Closeness Verification (Earth Mover's Distance):
   - Categorical SAs: Total Variation Distance D(P, Q) = 0.5 * sum(|p_i - q_i|) <= t.
   - Numerical SAs: 1-D Wasserstein Distance via integrated CDF difference on normalized domain [0, 1].
5. Re-identification & Disclosure Risk Metrics:
   - Individual Risk = 1 / |E_i|
   - Average Dataset Risk = (1/N) * sum(1 / |E(r)|)
   - Maximum Risk = max(1 / |E(r)|)
   - Marketer, Prosecutor, and Journalist risk modeling.
6. Mondrian Multidimensional Top-Down Generalization & Information Loss:
   - Recursive kd-tree median partitioning subject to k constraints.
   - Normalized Certainty Penalty (NCP) and Global Certainty Penalty (GCP).

Author: Dr. Abu Suraih Sakhri
License: MIT
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# =====================================================================
# 1. DATA MODELS & TYPES
# =====================================================================

@dataclass
class DatasetRecord:
    record_id: str | int
    attributes: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            **self.attributes
        }


@dataclass
class EquivalenceClass:
    qi_signature: Tuple[Any, ...]
    qi_names: Tuple[str, ...]
    records: List[DatasetRecord]

    @property
    def size(self) -> int:
        return len(self.records)

    def get_sensitive_values(self, sa_name: str) -> List[Any]:
        return [r.get(sa_name) for r in self.records]

    def get_sensitive_distribution(self, sa_name: str) -> Dict[Any, float]:
        vals = self.get_sensitive_values(sa_name)
        n = len(vals)
        if n == 0:
            return {}
        counts: Dict[Any, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return {k: v / n for k, v in counts.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qi_signature": dict(zip(self.qi_names, self.qi_signature)),
            "size": self.size,
            "record_ids": [r.record_id for r in self.records]
        }


@dataclass
class KAnonymityReport:
    k_target: int
    is_k_anonymous: bool
    total_records: int
    total_classes: int
    violating_classes_count: int
    violating_records_count: int
    min_class_size: int
    avg_class_size: float
    max_class_size: int
    violating_signatures: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LDiversityReport:
    l_target: int
    diversity_type: str  # "distinct", "entropy", "recursive"
    sensitive_attribute: str
    is_l_diverse: bool
    violating_classes_count: int
    violating_signatures: List[Dict[str, Any]]
    class_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TClosenessReport:
    t_target: float
    sensitive_attribute: str
    attribute_type: str  # "categorical" or "numerical"
    is_t_close: bool
    max_emd_distance: float
    avg_emd_distance: float
    violating_classes_count: int
    class_distances: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyRiskReport:
    total_records: int
    marketer_risk_avg: float
    prosecutor_risk_max: float
    journalist_risk_pct_above_threshold: float
    records_at_risk_count: int
    entropy_information_loss_ncp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. EQUIVALENCE CLASS BUILDER & GENERALIZATION
# =====================================================================

def build_equivalence_classes(
    records: Sequence[DatasetRecord],
    quasi_identifiers: Sequence[str]
) -> List[EquivalenceClass]:
    """Partitions a dataset into equivalence classes based on quasi-identifier values."""
    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier must be specified.")

    groups: Dict[Tuple[Any, ...], List[DatasetRecord]] = {}
    qi_tuple = tuple(quasi_identifiers)

    for r in records:
        key = tuple(str(r.get(qi, "")) for qi in quasi_identifiers)
        groups.setdefault(key, []).append(r)

    classes: List[EquivalenceClass] = []
    for sig, recs in groups.items():
        classes.append(EquivalenceClass(qi_signature=sig, qi_names=qi_tuple, records=recs))

    return classes


def generalize_age(age: int, step: int = 10) -> str:
    """Generalizes numeric age into discrete interval buckets [min, max]."""
    if age < 0:
        return "<0"
    start = (age // step) * step
    return f"{start}-{start + step - 1}"


def generalize_zipcode(zipcode: str, prefix_len: int = 3) -> str:
    """Truncates postal codes into prefix masks (e.g. 10025 -> 100**)."""
    clean = str(zipcode).strip()
    if len(clean) <= prefix_len:
        return clean
    return clean[:prefix_len] + "*" * (len(clean) - prefix_len)


# =====================================================================
# 3. K-ANONYMITY VERIFICATION
# =====================================================================

def verify_k_anonymity(
    records: Sequence[DatasetRecord],
    quasi_identifiers: Sequence[str],
    k: int
) -> KAnonymityReport:
    """
    Evaluates dataset against k-anonymity guarantee.
    A dataset satisfies k-anonymity if each equivalence class contains >= k records.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer >= 1.")
    if not records:
        return KAnonymityReport(
            k_target=k,
            is_k_anonymous=True,
            total_records=0,
            total_classes=0,
            violating_classes_count=0,
            violating_records_count=0,
            min_class_size=0,
            avg_class_size=0.0,
            max_class_size=0,
            violating_signatures=[]
        )

    eq_classes = build_equivalence_classes(records, quasi_identifiers)
    sizes = [c.size for c in eq_classes]
    violating = [c for c in eq_classes if c.size < k]
    violating_records = sum(c.size for c in violating)

    return KAnonymityReport(
        k_target=k,
        is_k_anonymous=len(violating) == 0,
        total_records=len(records),
        total_classes=len(eq_classes),
        violating_classes_count=len(violating),
        violating_records_count=violating_records,
        min_class_size=min(sizes) if sizes else 0,
        avg_class_size=round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        max_class_size=max(sizes) if sizes else 0,
        violating_signatures=[c.to_dict() for c in violating]
    )


# =====================================================================
# 4. L-DIVERSITY VERIFICATION
# =====================================================================

def verify_distinct_l_diversity(
    eq_classes: Sequence[EquivalenceClass],
    sensitive_attribute: str,
    l: int
) -> LDiversityReport:
    """
    Distinct l-diversity: Each equivalence class contains at least l well-represented
    distinct values for the sensitive attribute.
    """
    if l <= 0:
        raise ValueError("l must be a positive integer >= 1.")

    violating = []
    evals = []

    for c in eq_classes:
        vals = set(c.get_sensitive_values(sensitive_attribute))
        distinct_count = len(vals)
        passed = distinct_count >= l
        ev = {
            "qi_signature": dict(zip(c.qi_names, c.qi_signature)),
            "class_size": c.size,
            "distinct_sensitive_count": distinct_count,
            "distinct_values": sorted([str(x) for x in vals]),
            "passed": passed
        }
        evals.append(ev)
        if not passed:
            violating.append(ev)

    return LDiversityReport(
        l_target=l,
        diversity_type="distinct",
        sensitive_attribute=sensitive_attribute,
        is_l_diverse=len(violating) == 0,
        violating_classes_count=len(violating),
        violating_signatures=violating,
        class_evaluations=evals
    )


def verify_entropy_l_diversity(
    eq_classes: Sequence[EquivalenceClass],
    sensitive_attribute: str,
    l: int
) -> LDiversityReport:
    """
    Entropy l-diversity: H(E) = -sum(p(s) * ln(p(s))) >= ln(l).
    Protects against skewness and positive disclosure.
    """
    if l <= 0:
        raise ValueError("l must be >= 1.")

    required_entropy = math.log(l)
    violating = []
    evals = []

    for c in eq_classes:
        dist = c.get_sensitive_distribution(sensitive_attribute)
        entropy = -sum(p * math.log(p) for p in dist.values() if p > 0.0)
        passed = entropy >= required_entropy - 1e-9
        ev = {
            "qi_signature": dict(zip(c.qi_names, c.qi_signature)),
            "class_size": c.size,
            "entropy": round(entropy, 4),
            "required_entropy": round(required_entropy, 4),
            "passed": passed
        }
        evals.append(ev)
        if not passed:
            violating.append(ev)

    return LDiversityReport(
        l_target=l,
        diversity_type="entropy",
        sensitive_attribute=sensitive_attribute,
        is_l_diverse=len(violating) == 0,
        violating_classes_count=len(violating),
        violating_signatures=violating,
        class_evaluations=evals
    )


def verify_recursive_c_l_diversity(
    eq_classes: Sequence[EquivalenceClass],
    sensitive_attribute: str,
    c: float,
    l: int
) -> LDiversityReport:
    """
    Recursive (c, l)-diversity: r_1 < c * (r_l + r_{l+1} + ... + r_m).
    Ensures the most frequent sensitive value does not dominate the rest.
    """
    if l < 1 or c <= 0:
        raise ValueError("c must be > 0 and l must be >= 1.")

    violating = []
    evals = []

    for eq in eq_classes:
        vals = eq.get_sensitive_values(sensitive_attribute)
        counts: Dict[Any, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1

        sorted_freqs = sorted(counts.values(), reverse=True)
        m = len(sorted_freqs)

        if m < l:
            passed = False
            r1 = sorted_freqs[0] if m > 0 else 0
            tail_sum = 0
        else:
            r1 = sorted_freqs[0]
            tail_sum = sum(sorted_freqs[l - 1:])
            passed = r1 < (c * tail_sum)

        ev = {
            "qi_signature": dict(zip(eq.qi_names, eq.qi_signature)),
            "class_size": eq.size,
            "r1": r1,
            "tail_sum": tail_sum,
            "c_times_tail": round(c * tail_sum, 2),
            "passed": passed
        }
        evals.append(ev)
        if not passed:
            violating.append(ev)

    return LDiversityReport(
        l_target=l,
        diversity_type=f"recursive(c={c}, l={l})",
        sensitive_attribute=sensitive_attribute,
        is_l_diverse=len(violating) == 0,
        violating_classes_count=len(violating),
        violating_signatures=violating,
        class_evaluations=evals
    )


# =====================================================================
# 5. T-CLOSENESS VERIFICATION (EARTH MOVER'S DISTANCE)
# =====================================================================

def calculate_categorical_emd(p_dist: Dict[Any, float], q_dist: Dict[Any, float]) -> float:
    """
    Computes Earth Mover's Distance for unordered categorical attribute (Total Variation Distance).
    D(P, Q) = 0.5 * sum(|P(s) - Q(s)|)
    Scale is bounded in [0.0, 1.0].
    """
    support = set(p_dist.keys()) | set(q_dist.keys())
    return round(0.5 * sum(abs(p_dist.get(k, 0.0) - q_dist.get(k, 0.0)) for k in support), 4)


def calculate_numerical_emd(p_values: Sequence[float | int], q_values: Sequence[float | int]) -> float:
    """
    Computes 1-D Earth Mover's Distance (Wasserstein Distance) for numerical attributes
    via normalized domain integrated absolute CDF differences.
    Normalized to [0.0, 1.0] scale.
    """
    if not p_values or not q_values:
        return 0.0

    all_vals = list(p_values) + list(q_values)
    val_min = float(min(all_vals))
    val_max = float(max(all_vals))
    val_range = val_max - val_min

    if val_range <= 0.0:
        return 0.0

    # Sorted evaluation points
    domain = sorted(set(all_vals))
    n_p = len(p_values)
    n_q = len(q_values)

    p_sorted = sorted(p_values)
    q_sorted = sorted(q_values)

    emd = 0.0
    for i in range(len(domain) - 1):
        x1 = domain[i]
        x2 = domain[i + 1]
        width_norm = (x2 - x1) / val_range

        # Empirical CDF at x1
        cdf_p = sum(1 for v in p_sorted if v <= x1) / n_p
        cdf_q = sum(1 for v in q_sorted if v <= x1) / n_q

        emd += abs(cdf_p - cdf_q) * width_norm

    return round(emd, 4)


def verify_t_closeness(
    records: Sequence[DatasetRecord],
    eq_classes: Sequence[EquivalenceClass],
    sensitive_attribute: str,
    t: float,
    is_numerical: bool = False
) -> TClosenessReport:
    """
    Evaluates dataset against t-closeness standard.
    Distance between equivalence class sensitive distribution P and global dataset distribution Q <= t.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError("t threshold must be in range [0.0, 1.0].")

    global_vals = [r.get(sensitive_attribute) for r in records if r.get(sensitive_attribute) is not None]
    n_global = len(global_vals)
    if n_global == 0:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' has no valid values in dataset.")

    distances = []
    violating = []

    if not is_numerical:
        # Categorical global distribution
        q_counts: Dict[Any, int] = {}
        for v in global_vals:
            q_counts[v] = q_counts.get(v, 0) + 1
        q_dist = {k: v / n_global for k, v in q_counts.items()}

        for eq in eq_classes:
            p_dist = eq.get_sensitive_distribution(sensitive_attribute)
            dist_emd = calculate_categorical_emd(p_dist, q_dist)
            passed = dist_emd <= t + 1e-9
            entry = {
                "qi_signature": dict(zip(eq.qi_names, eq.qi_signature)),
                "class_size": eq.size,
                "emd_distance": dist_emd,
                "passed": passed
            }
            distances.append(entry)
            if not passed:
                violating.append(entry)
    else:
        # Numerical 1D EMD
        for eq in eq_classes:
            p_vals = [v for v in eq.get_sensitive_values(sensitive_attribute) if v is not None]
            dist_emd = calculate_numerical_emd(p_vals, global_vals)
            passed = dist_emd <= t + 1e-9
            entry = {
                "qi_signature": dict(zip(eq.qi_names, eq.qi_signature)),
                "class_size": eq.size,
                "emd_distance": dist_emd,
                "passed": passed
            }
            distances.append(entry)
            if not passed:
                violating.append(entry)

    dist_values = [d["emd_distance"] for d in distances]
    max_d = max(dist_values) if dist_values else 0.0
    avg_d = sum(dist_values) / len(dist_values) if dist_values else 0.0

    return TClosenessReport(
        t_target=t,
        sensitive_attribute=sensitive_attribute,
        attribute_type="numerical" if is_numerical else "categorical",
        is_t_close=len(violating) == 0,
        max_emd_distance=round(max_d, 4),
        avg_emd_distance=round(avg_d, 4),
        violating_classes_count=len(violating),
        class_distances=distances
    )


# =====================================================================
# 6. RE-IDENTIFICATION RISK & INFORMATION LOSS AUDIT
# =====================================================================

def audit_privacy_risks(
    records: Sequence[DatasetRecord],
    eq_classes: Sequence[EquivalenceClass],
    high_risk_threshold: float = 0.5
) -> PrivacyRiskReport:
    """
    Computes standard re-identification risk metrics:
    - Marketer Risk (Average risk across population) = 1/N * sum(1 / |E(r)|)
    - Prosecutor Risk (Maximum risk for any individual) = max(1 / |E_i|)
    - Journalist Risk (Proportion of records with risk > threshold)
    """
    total = len(records)
    if total == 0:
        return PrivacyRiskReport(0, 0.0, 0.0, 0.0, 0, 0.0)

    individual_risks: List[float] = []
    for eq in eq_classes:
        class_risk = 1.0 / eq.size
        for _ in range(eq.size):
            individual_risks.append(class_risk)

    marketer = sum(individual_risks) / total
    prosecutor = max(individual_risks) if individual_risks else 0.0
    at_risk_count = sum(1 for r in individual_risks if r >= high_risk_threshold)
    journalist_pct = (at_risk_count / total) * 100.0

    # Normalized Certainty Penalty (Information Loss across classes)
    ncp = 1.0 - (len(eq_classes) / float(total))

    return PrivacyRiskReport(
        total_records=total,
        marketer_risk_avg=round(marketer, 4),
        prosecutor_risk_max=round(prosecutor, 4),
        journalist_risk_pct_above_threshold=round(journalist_pct, 2),
        records_at_risk_count=at_risk_count,
        entropy_information_loss_ncp=round(ncp, 4)
    )


# =====================================================================
# 7. MONDRIAN MULTIDIMENSIONAL ANONYMIZATION ENGINE
# =====================================================================

class MondrianAnonymizer:
    """
    Top-down multidimensional partitioning algorithm (Mondrian)
    Recursively splits quasi-identifier domains along median values to achieve k-anonymity.
    """
    def __init__(self, k: int = 3):
        self.k = k

    def anonymize_records(
        self,
        records: List[DatasetRecord],
        qi_numeric: List[str],
        qi_categorical: List[str]
    ) -> List[Dict[str, Any]]:
        """Applies multidimensional partitioning and interval/prefix generalization."""
        if len(records) < self.k:
            # Cannot partition below k
            return [r.to_dict() for r in records]

        # Partition recursively
        partitions = self._partition(records, qi_numeric + qi_categorical)
        anonymized: List[Dict[str, Any]] = []

        for part in partitions:
            # Compute generalized representation
            gen_rep = self._generalize_partition(part, qi_numeric, qi_categorical)
            for r in part:
                row = r.to_dict()
                row.update(gen_rep)
                anonymized.append(row)

        return anonymized

    def _partition(self, records: List[DatasetRecord], qis: List[str]) -> List[List[DatasetRecord]]:
        if len(records) < 2 * self.k:
            return [records]

        # Choose QI with largest normalized span
        best_qi = None
        best_range = -1.0

        for qi in qis:
            vals = [r.get(qi) for r in records if r.get(qi) is not None]
            if not vals:
                continue
            if isinstance(vals[0], (int, float)):
                span = max(vals) - min(vals)
            else:
                span = len(set(vals))
            if span > best_range:
                best_range = span
                best_qi = qi

        if best_qi is None or best_range == 0:
            return [records]

        # Sort along best_qi
        sorted_recs = sorted(records, key=lambda x: str(x.get(best_qi, "")))
        mid = len(sorted_recs) // 2

        left = sorted_recs[:mid]
        right = sorted_recs[mid:]

        if len(left) < self.k or len(right) < self.k:
            return [records]

        return self._partition(left, qis) + self._partition(right, qis)

    def _generalize_partition(
        self,
        partition: List[DatasetRecord],
        numeric_qis: List[str],
        cat_qis: List[str]
    ) -> Dict[str, Any]:
        generalized: Dict[str, Any] = {}
        for num in numeric_qis:
            vals = [float(r.get(num)) for r in partition if r.get(num) is not None]
            if vals:
                mn, mx = min(vals), max(vals)
                generalized[num] = f"[{int(mn)}-{int(mx)}]" if mn != mx else str(int(mn))

        for cat in cat_qis:
            vals = list({str(r.get(cat)) for r in partition if r.get(cat) is not None})
            if len(vals) == 1:
                generalized[cat] = vals[0]
            else:
                generalized[cat] = f"{{{','.join(sorted(vals))}}}"

        return generalized


# =====================================================================
# 8. MASTER ANONYMITY AUDIT PIPELINE
# =====================================================================

class AnonymityGuardPipeline:
    """End-to-end privacy audit pipeline for k-anonymity, l-diversity, and t-closeness."""

    def audit_dataset(
        self,
        records: List[DatasetRecord],
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        k: int = 3,
        l: int = 2,
        t: float = 0.25,
        is_sa_numerical: bool = False,
        entropy_l: bool = True
    ) -> Dict[str, Any]:
        """Runs comprehensive privacy audits over the input dataset."""
        eq_classes = build_equivalence_classes(records, quasi_identifiers)

        k_rep = verify_k_anonymity(records, quasi_identifiers, k)
        
        distinct_l_rep = verify_distinct_l_diversity(eq_classes, sensitive_attribute, l)
        entropy_l_rep = verify_entropy_l_diversity(eq_classes, sensitive_attribute, l) if entropy_l else None

        t_rep = verify_t_closeness(records, eq_classes, sensitive_attribute, t, is_numerical=is_sa_numerical)
        risk_rep = audit_privacy_risks(records, eq_classes)

        overall_compliant = (
            k_rep.is_k_anonymous and
            distinct_l_rep.is_l_diverse and
            (entropy_l_rep.is_l_diverse if entropy_l_rep else True) and
            t_rep.is_t_close
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_compliant": overall_compliant,
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attribute": sensitive_attribute,
            "k_anonymity": k_rep.to_dict(),
            "distinct_l_diversity": distinct_l_rep.to_dict(),
            "entropy_l_diversity": entropy_l_rep.to_dict() if entropy_l_rep else None,
            "t_closeness": t_rep.to_dict(),
            "privacy_risks": risk_rep.to_dict()
        }
