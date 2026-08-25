#!/usr/bin/env python3
"""
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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Record:
    record_id: int
    age: int
    zipcode: str
    diagnosis: str
    los_days: int


Generalizer = Callable[[Record], Tuple[str, ...]]


def generalize_standard(age_bucket_years: int = 5,
                        zip_prefix_digits: int = 3) -> Generalizer:
    def _g(rec: Record) -> Tuple[str, ...]:
        bucket_start = (rec.age // age_bucket_years) * age_bucket_years
        return (
            f"{bucket_start}-{bucket_start + age_bucket_years - 1}",
            rec.zipcode[:zip_prefix_digits],
        )
    return _g


def equivalence_classes(records: Sequence[Record],
                        generalizer: Generalizer) -> Dict[Tuple[str, ...], List[Record]]:
    classes: Dict[Tuple[str, ...], List[Record]] = {}
    for rec in records:
        classes.setdefault(generalizer(rec), []).append(rec)
    return classes


def k_anonymity_ok(classes: Dict[Tuple[str, ...], List[Record]], k: int) -> bool:
    return all(len(members) >= k for members in classes.values())


def l_diversity_violations(classes: Dict[Tuple[str, ...], List[Record]],
                           l: int, field_name: str = "diagnosis") -> List[Tuple[str, ...]]:
    bad = []
    for qi, members in classes.items():
        distinct = {getattr(r, field_name) for r in members}
        if len(distinct) < l:
            bad.append(qi)
    return bad


def emd_categorical(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    """Total-variation distance / 1 for categorical distributions."""
    support = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(s, 0.0) - dist_b.get(s, 0.0)) for s in support)


def emd_numeric(values_a: Sequence[int], values_b: Sequence[int]) -> float:
    """EMD between 1-D empirical distributions via integrated |CDF difference|.

    Values are min-max normalized to [0, 1] so the result shares the same
    [0, 1] scale as the categorical total-variation distance and can be
    audited against one common t threshold.
    """
    if not values_a or not values_b:
        raise ValueError("empty value list")
    lo = min(min(values_a), min(values_b))
    hi = max(max(values_a), max(values_b))
    if hi == lo:
        return 0.0
    points = sorted(set(values_a) | set(values_b))
    emd = 0.0
    prev: Optional[float] = None

    def cdf_frac(values: Sequence[int], v: int) -> float:
        return sum(1 for x in values if x <= v) / len(values)

    for v in points:
        fa = cdf_frac(values_a, v)
        fb = cdf_frac(values_b, v)
        scaled = (v - lo) / (hi - lo)
        if prev is not None:
            emd += abs(fa - fb) * (scaled - prev)
        prev = scaled
    return emd


def global_distribution(records: Sequence[Record], field_name: str) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for rec in records:
        key = str(getattr(rec, field_name))
        counts[key] = counts.get(key, 0) + 1
    n = len(records)
    return {key: c / n for key, c in counts.items()}


def t_closeness_audit(classes: Dict[Tuple[str, ...], List[Record]],
                      records: Sequence[Record], t: float,
                      field_name: str = "diagnosis",
                      numeric_field: Optional[str] = None) -> Dict[str, object]:
    """Max EMD across classes against the global distribution."""
    overall = global_distribution(records, field_name)
    max_emd = 0.0
    worst_qi: Optional[Tuple[str, ...]] = None

    for qi, members in classes.items():
        if numeric_field is not None:
            emd = emd_numeric([getattr(r, numeric_field) for r in members],
                              [getattr(r, numeric_field) for r in records])
        else:
            local = global_distribution(members, field_name)
            emd = emd_categorical(local, overall)
        if emd > max_emd:
            max_emd, worst_qi = emd, qi

    return {
        "max_emd": round(max_emd, 4),
        "worst_equivalence_class": worst_qi,
        "t_threshold": t,
        "t_closeness_satisfied": bool(max_emd <= t),
    }


GENERALIZATION_LADDER: List[Dict[str, int]] = [
    {"age_bucket": 5, "zip_prefix": 4},
    {"age_bucket": 5, "zip_prefix": 3},
    {"age_bucket": 10, "zip_prefix": 3},
    {"age_bucket": 10, "zip_prefix": 2},
    {"age_bucket": 20, "zip_prefix": 2},
]
MAX_LEVELS = {"age_bucket": 20, "zip_prefix": 2}


def information_loss(level: Dict[str, int]) -> float:
    parts = [
        math.log2(level["age_bucket"] / GENERALIZATION_LADDER[0]["age_bucket"]) /
        math.log2(MAX_LEVELS["age_bucket"] / GENERALIZATION_LADDER[0]["age_bucket"]),
        math.log2(GENERALIZATION_LADDER[0]["zip_prefix"] / level["zip_prefix"]) /
        math.log2(GENERALIZATION_LADDER[0]["zip_prefix"] / MAX_LEVELS["zip_prefix"]),
    ]
    return round(sum(parts) / len(parts), 4)


def anonymize(records: Sequence[Record], k: int, l: int, t: float,
              numeric_sensitive: Optional[str] = None) -> Dict[str, object]:
    """Greedy minimal-generalization search up the ladder until k/l/t hold."""
    for level in GENERALIZATION_LADDER:
        gen = generalize_standard(level["age_bucket"], level["zip_prefix"])
        classes = equivalence_classes(records, gen)
        if not k_anonymity_ok(classes, k):
            continue
        if l_diversity_violations(classes, l):
            continue
        audit = t_closeness_audit(classes, records, t, numeric_field=numeric_sensitive)
        if audit["t_closeness_satisfied"]:
            return {
                "satisfied": True,
                "generalization_level": level,
                "information_loss": information_loss(level),
                "num_equivalence_classes": len(classes),
                "min_class_size": min(len(v) for v in classes.values()),
                "t_audit": audit,
                "anonymized_records": [
                    {"record_id": r.record_id, **dict(zip(
                        ("age_bucket", "zipcode_prefix"), gen(r))),
                     "diagnosis": r.diagnosis}
                    for r in records
                ],
            }
    return {"satisfied": False}


def _demo() -> None:
    diagnoses = ["diabetes", "hypertension", "asthma", "depression", "copd"]
    rng_values = [3, 7, 12, 21, 30, 44, 60, 90]
    records = [
        Record(i, 23 + (i * 13) % 55, f"021{chr(52 + i % 6)}{i % 10}{i % 5}",
               diagnoses[i % len(diagnoses)], rng_values[i % len(rng_values)])
        for i in range(40)
    ]

    result = anonymize(records, k=4, l=2, t=0.45, numeric_sensitive="los_days")
    print({"satisfied": result["satisfied"],
           "level": result.get("generalization_level"),
           "info_loss": result.get("information_loss"),
           "min_class_size": result.get("min_class_size")})

    gen = generalize_standard(10, 2)
    classes = equivalence_classes(records, gen)
    print({"k4_anonymous_at_10y_2digit": k_anonymity_ok(classes, 4)})
    print(t_closeness_audit(classes, records, t=0.45, numeric_field="los_days"))


if __name__ == "__main__":
    _demo()
