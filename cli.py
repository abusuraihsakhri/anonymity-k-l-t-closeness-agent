#!/usr/bin/env python3
"""
Anonymity Guard - Command Line Interface (CLI)

Performs k-anonymity, l-diversity, and t-closeness audits on health/tabular datasets,
computes re-identification risk metrics, and runs top-down Mondrian anonymization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

from anonymity_klt import (
    AnonymityGuardPipeline,
    DatasetRecord,
    MondrianAnonymizer,
    build_equivalence_classes,
    generalize_age,
    generalize_zipcode,
    verify_k_anonymity,
    verify_distinct_l_diversity,
    verify_t_closeness,
    audit_privacy_risks,
)


def load_benchmark_dataset() -> List[DatasetRecord]:
    """Constructs realistic synthetic clinical research dataset for privacy auditing."""
    raw_data = [
        {"id": 1, "age": 28, "zipcode": "10025", "gender": "Female", "diagnosis": "Hypertension", "los_days": 2},
        {"id": 2, "age": 29, "zipcode": "10028", "gender": "Female", "diagnosis": "Asthma", "los_days": 3},
        {"id": 3, "age": 27, "zipcode": "10021", "gender": "Female", "diagnosis": "Diabetes", "los_days": 4},
        {"id": 4, "age": 42, "zipcode": "90210", "gender": "Male", "diagnosis": "Hypertension", "los_days": 5},
        {"id": 5, "age": 45, "zipcode": "90212", "gender": "Male", "diagnosis": "Pneumonia", "los_days": 7},
        {"id": 6, "age": 44, "zipcode": "90215", "gender": "Male", "diagnosis": "COVID-19", "los_days": 6},
        {"id": 7, "age": 63, "zipcode": "60601", "gender": "Female", "diagnosis": "Heart Failure", "los_days": 10},
        {"id": 8, "age": 68, "zipcode": "60605", "gender": "Female", "diagnosis": "Diabetes", "los_days": 8},
        {"id": 9, "age": 65, "zipcode": "60611", "gender": "Female", "diagnosis": "Stroke", "los_days": 14},
        {"id": 10, "age": 72, "zipcode": "33101", "gender": "Male", "diagnosis": "Pneumonia", "los_days": 9},
        {"id": 11, "age": 75, "zipcode": "33109", "gender": "Male", "diagnosis": "Hypertension", "los_days": 4},
        {"id": 12, "age": 70, "zipcode": "33139", "gender": "Male", "diagnosis": "Heart Failure", "los_days": 12},
    ]

    records: List[DatasetRecord] = []
    for row in raw_data:
        # Pre-apply 10-year age generalization and 3-digit zip prefix
        gen_age = generalize_age(row["age"], step=10)
        gen_zip = generalize_zipcode(row["zipcode"], prefix_len=3)
        records.append(DatasetRecord(
            record_id=row["id"],
            attributes={
                "raw_age": row["age"],
                "age_group": gen_age,
                "raw_zip": row["zipcode"],
                "zip3": gen_zip,
                "gender": row["gender"],
                "diagnosis": row["diagnosis"],
                "los_days": row["los_days"]
            }
        ))
    return records


def run_benchmark_audit(k: int = 3, l: int = 2, t: float = 0.35, json_out: bool = False) -> None:
    records = load_benchmark_dataset()
    pipeline = AnonymityGuardPipeline()
    res = pipeline.audit_dataset(
        records=records,
        quasi_identifiers=["age_group", "zip3", "gender"],
        sensitive_attribute="diagnosis",
        k=k,
        l=l,
        t=t,
        is_sa_numerical=False,
        entropy_l=True
    )

    if json_out:
        print(json.dumps(res, indent=2))
        return

    print("=" * 76)
    print("      ANONYMITY GUARD - PRIVACY PRESERVATION AUDIT REPORT            ")
    print("=" * 76)
    print(f" Overall Compliance Status : {'COMPLIANT' if res['overall_compliant'] else 'NON-COMPLIANT'}")
    print(f" Quasi-Identifiers         : {', '.join(res['quasi_identifiers'])}")
    print(f" Sensitive Attribute       : {res['sensitive_attribute']}")
    print("-" * 76)
    print(f" 1. k-ANONYMITY AUDIT (Target k={k}):")
    print("-" * 76)
    k_res = res["k_anonymity"]
    print(f"  Satisfied              : {k_res['is_k_anonymous']}")
    print(f"  Total Equivalence Cls  : {k_res['total_classes']} (Class Sizes: Min={k_res['min_class_size']}, Avg={k_res['avg_class_size']:.1f}, Max={k_res['max_class_size']})")
    print(f"  Violating Classes      : {k_res['violating_classes_count']}")
    print("-" * 76)
    print(f" 2. l-DIVERSITY AUDIT (Target l={l}):")
    print("-" * 76)
    l_res = res["distinct_l_diversity"]
    print(f"  Distinct l-Diversity   : {l_res['is_l_diverse']} ({l_res['violating_classes_count']} violations)")
    ent_res = res["entropy_l_diversity"]
    if ent_res:
        print(f"  Entropy l-Diversity    : {ent_res['is_l_diverse']} ({ent_res['violating_classes_count']} violations)")
    print("-" * 76)
    print(f" 3. t-CLOSENESS AUDIT (Target t={t:.2f}):")
    print("-" * 76)
    t_res = res["t_closeness"]
    print(f"  Satisfied              : {t_res['is_t_close']}")
    print(f"  Max EMD Distance       : {t_res['max_emd_distance']:.4f} (Avg: {t_res['avg_emd_distance']:.4f})")
    print(f"  Violating Classes      : {t_res['violating_classes_count']}")
    print("-" * 76)
    print(" 4. RE-IDENTIFICATION RISK & DISCLOSURE METRICS:")
    print("-" * 76)
    r_res = res["privacy_risks"]
    print(f"  Marketer Risk (Avg)    : {r_res['marketer_risk_avg']:.4f} ({r_res['marketer_risk_avg']*100:.1f}%)")
    print(f"  Prosecutor Risk (Max)  : {r_res['prosecutor_risk_max']:.4f} ({r_res['prosecutor_risk_max']*100:.1f}%)")
    print(f"  Journalist Risk (>50%) : {r_res['journalist_risk_pct_above_threshold']:.1f}% ({r_res['records_at_risk_count']} records)")
    print("=" * 76)


def interactive_terminal() -> None:
    print("Anonymity Guard Interactive CLI. Type 'help' for commands, 'exit' to quit.\n")
    while True:
        try:
            line = input("anonymity-guard> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        elif line.lower() == "help":
            print("Commands:")
            print("  audit                     - Run full k, l, t audit on benchmark dataset")
            print("  mondrian [k]              - Run Mondrian multidimensional partition")
            print("  eval <k> <l> <t>          - Audit benchmark dataset with specific parameters")
            print("  exit                      - Quit CLI")
        elif line.lower() == "audit":
            run_benchmark_audit(k=3, l=2, t=0.35)
        elif line.lower().startswith("eval "):
            parts = line.split()[1:]
            if len(parts) < 3:
                print("Usage: eval <k> <l> <t>")
                continue
            try:
                k_val = int(parts[0])
                l_val = int(parts[1])
                t_val = float(parts[2])
                run_benchmark_audit(k=k_val, l=l_val, t=t_val)
            except Exception as ex:
                print(f"Error: {ex}")
        elif line.lower().startswith("mondrian"):
            parts = line.split()[1:]
            k_val = int(parts[0]) if parts else 3
            recs = load_benchmark_dataset()
            anon = MondrianAnonymizer(k=k_val)
            res = anon.anonymize_records(recs, qi_numeric=["raw_age", "los_days"], qi_categorical=["gender"])
            print(f"Mondrian Anonymization Result (k={k_val}, {len(res)} records):")
            for r in res[:4]:
                print(f"  Record {r['record_id']}: Age={r['raw_age']}, LOS={r['los_days']}, Gender={r['gender']}, Diagnosis={r['diagnosis']}")
        else:
            print(f"Unknown command: {line}. Type 'help'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anonymity Guard - k-Anonymity, l-Diversity, and t-Closeness Verification System"
    )
    parser.add_argument("--audit", action="store_true", help="Run full privacy audit on benchmark dataset")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI terminal")
    parser.add_argument("--k", type=int, default=3, help="k-anonymity threshold (default: 3)")
    parser.add_argument("--l", type=int, default=2, help="l-diversity threshold (default: 2)")
    parser.add_argument("--t", type=float, default=0.35, help="t-closeness threshold (default: 0.35)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.interactive:
        interactive_terminal()
    else:
        run_benchmark_audit(k=args.k, l=args.l, t=args.t, json_out=args.json)


if __name__ == "__main__":
    main()
