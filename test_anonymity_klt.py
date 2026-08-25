#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for Anonymity Guard (k-Anonymity, l-Diversity, t-Closeness).
"""

import math
import unittest
from anonymity_klt import (
    AnonymityGuardPipeline,
    DatasetRecord,
    EquivalenceClass,
    MondrianAnonymizer,
    audit_privacy_risks,
    build_equivalence_classes,
    calculate_categorical_emd,
    calculate_numerical_emd,
    generalize_age,
    generalize_zipcode,
    verify_distinct_l_diversity,
    verify_entropy_l_diversity,
    verify_k_anonymity,
    verify_recursive_c_l_diversity,
    verify_t_closeness,
)


class TestGeneralizationHelpers(unittest.TestCase):
    def test_age_generalization_standard(self):
        self.assertEqual(generalize_age(28, step=10), "20-29")
        self.assertEqual(generalize_age(40, step=10), "40-49")
        self.assertEqual(generalize_age(5, step=10), "0-9")

    def test_age_generalization_custom_step(self):
        self.assertEqual(generalize_age(23, step=5), "20-24")
        self.assertEqual(generalize_age(25, step=5), "25-29")

    def test_age_generalization_negative(self):
        self.assertEqual(generalize_age(-3), "<0")

    def test_zipcode_generalization(self):
        self.assertEqual(generalize_zipcode("10025", prefix_len=3), "100**")
        self.assertEqual(generalize_zipcode("90210", prefix_len=2), "90***")
        self.assertEqual(generalize_zipcode("100", prefix_len=3), "100")


class TestEquivalenceClasses(unittest.TestCase):
    def test_build_equivalence_classes(self):
        records = [
            DatasetRecord(1, {"age_group": "20-29", "gender": "F", "diagnosis": "Flu"}),
            DatasetRecord(2, {"age_group": "20-29", "gender": "F", "diagnosis": "Asthma"}),
            DatasetRecord(3, {"age_group": "40-49", "gender": "M", "diagnosis": "Diabetes"}),
        ]
        classes = build_equivalence_classes(records, ["age_group", "gender"])
        self.assertEqual(len(classes), 2)
        sizes = sorted([c.size for c in classes])
        self.assertEqual(sizes, [1, 2])

    def test_empty_qi_raises_error(self):
        records = [DatasetRecord(1, {"age": 20})]
        with self.assertRaises(ValueError):
            build_equivalence_classes(records, [])


class TestKAnonymity(unittest.TestCase):
    def test_k_anonymity_satisfied(self):
        records = [
            DatasetRecord(1, {"zip": "100**", "age": "20-29"}),
            DatasetRecord(2, {"zip": "100**", "age": "20-29"}),
            DatasetRecord(3, {"zip": "902**", "age": "40-49"}),
            DatasetRecord(4, {"zip": "902**", "age": "40-49"}),
        ]
        rep = verify_k_anonymity(records, ["zip", "age"], k=2)
        self.assertTrue(rep.is_k_anonymous)
        self.assertEqual(rep.violating_classes_count, 0)
        self.assertEqual(rep.min_class_size, 2)

    def test_k_anonymity_violation(self):
        records = [
            DatasetRecord(1, {"zip": "100**", "age": "20-29"}),
            DatasetRecord(2, {"zip": "100**", "age": "20-29"}),
            DatasetRecord(3, {"zip": "902**", "age": "40-49"}),  # singleton class
        ]
        rep = verify_k_anonymity(records, ["zip", "age"], k=2)
        self.assertFalse(rep.is_k_anonymous)
        self.assertEqual(rep.violating_classes_count, 1)
        self.assertEqual(rep.violating_records_count, 1)

    def test_k_anonymity_invalid_k(self):
        with self.assertRaises(ValueError):
            verify_k_anonymity([], ["age"], k=0)


class TestLDiversity(unittest.TestCase):
    def setUp(self):
        self.eq1 = EquivalenceClass(
            qi_signature=("20-29", "100**"),
            qi_names=("age", "zip"),
            records=[
                DatasetRecord(1, {"diagnosis": "Diabetes"}),
                DatasetRecord(2, {"diagnosis": "Flu"}),
                DatasetRecord(3, {"diagnosis": "Asthma"}),
            ]
        )
        self.eq_homogeneous = EquivalenceClass(
            qi_signature=("40-49", "902**"),
            qi_names=("age", "zip"),
            records=[
                DatasetRecord(4, {"diagnosis": "Cancer"}),
                DatasetRecord(5, {"diagnosis": "Cancer"}),
                DatasetRecord(6, {"diagnosis": "Cancer"}),
            ]
        )

    def test_distinct_l_diversity_passes(self):
        rep = verify_distinct_l_diversity([self.eq1], "diagnosis", l=3)
        self.assertTrue(rep.is_l_diverse)
        self.assertEqual(rep.violating_classes_count, 0)

    def test_distinct_l_diversity_homogeneity_failure(self):
        rep = verify_distinct_l_diversity([self.eq_homogeneous], "diagnosis", l=2)
        self.assertFalse(rep.is_l_diverse)
        self.assertEqual(rep.violating_classes_count, 1)

    def test_entropy_l_diversity(self):
        # eq1 has 3 distinct uniform values -> H = ln(3) = 1.0986 >= ln(2)
        rep = verify_entropy_l_diversity([self.eq1], "diagnosis", l=2)
        self.assertTrue(rep.is_l_diverse)

        # eq_homogeneous has 1 value -> H = 0.0 < ln(2)
        rep_hom = verify_entropy_l_diversity([self.eq_homogeneous], "diagnosis", l=2)
        self.assertFalse(rep_hom.is_l_diverse)

    def test_recursive_c_l_diversity(self):
        # 3 values each frequency 1: r1=1, tail (l=2) = 2. r1 < 2 * 2 = 4 -> passes
        rep = verify_recursive_c_l_diversity([self.eq1], "diagnosis", c=2.0, l=2)
        self.assertTrue(rep.is_l_diverse)

        # Homogeneous class has only 1 distinct value (m < l) -> fails
        rep_hom = verify_recursive_c_l_diversity([self.eq_homogeneous], "diagnosis", c=2.0, l=2)
        self.assertFalse(rep_hom.is_l_diverse)


class TestTCloseness(unittest.TestCase):
    def test_categorical_emd_identical(self):
        dist_a = {"Flu": 0.5, "Asthma": 0.5}
        dist_b = {"Flu": 0.5, "Asthma": 0.5}
        self.assertEqual(calculate_categorical_emd(dist_a, dist_b), 0.0)

    def test_categorical_emd_disjoint(self):
        dist_a = {"Flu": 1.0}
        dist_b = {"Asthma": 1.0}
        self.assertEqual(calculate_categorical_emd(dist_a, dist_b), 1.0)

    def test_numerical_emd_identical(self):
        vals = [2, 4, 6, 8]
        self.assertEqual(calculate_numerical_emd(vals, vals), 0.0)

    def test_numerical_emd_shifted(self):
        vals_a = [1, 2, 3]
        vals_b = [8, 9, 10]
        emd = calculate_numerical_emd(vals_a, vals_b)
        self.assertTrue(0.5 < emd <= 1.0)

    def test_t_closeness_audit_categorical(self):
        recs = [
            DatasetRecord(1, {"grp": "A", "diag": "Flu"}),
            DatasetRecord(2, {"grp": "A", "diag": "Asthma"}),
            DatasetRecord(3, {"grp": "B", "diag": "Flu"}),
            DatasetRecord(4, {"grp": "B", "diag": "Asthma"}),
        ]
        eqs = build_equivalence_classes(recs, ["grp"])
        rep = verify_t_closeness(recs, eqs, "diag", t=0.1)
        self.assertTrue(rep.is_t_close)
        self.assertEqual(rep.max_emd_distance, 0.0)

    def test_t_closeness_audit_failure_on_skew(self):
        recs = [
            DatasetRecord(1, {"grp": "A", "diag": "Flu"}),
            DatasetRecord(2, {"grp": "A", "diag": "Flu"}),
            DatasetRecord(3, {"grp": "B", "diag": "Cancer"}),
            DatasetRecord(4, {"grp": "B", "diag": "Cancer"}),
        ]
        eqs = build_equivalence_classes(recs, ["grp"])
        rep = verify_t_closeness(recs, eqs, "diag", t=0.2)
        self.assertFalse(rep.is_t_close)
        self.assertTrue(rep.violating_classes_count > 0)


class TestPrivacyRiskAudit(unittest.TestCase):
    def test_risk_metrics_calculation(self):
        eq1 = EquivalenceClass(("A",), ("grp",), [DatasetRecord(1, {}), DatasetRecord(2, {})])  # size 2 -> risk 0.5
        eq2 = EquivalenceClass(("B",), ("grp",), [DatasetRecord(3, {}), DatasetRecord(4, {}), DatasetRecord(5, {}), DatasetRecord(6, {})])  # size 4 -> risk 0.25
        recs = eq1.records + eq2.records

        rep = audit_privacy_risks(recs, [eq1, eq2], high_risk_threshold=0.5)
        self.assertEqual(rep.total_records, 6)
        # Average risk = (2*0.5 + 4*0.25) / 6 = 2.0 / 6 = 0.3333
        self.assertAlmostEqual(rep.marketer_risk_avg, 0.3333, delta=0.001)
        self.assertEqual(rep.prosecutor_risk_max, 0.5)
        self.assertEqual(rep.records_at_risk_count, 2)


class TestMondrianAnonymizer(unittest.TestCase):
    def test_mondrian_partitioning_enforces_k(self):
        records = [
            DatasetRecord(i, {"age": 20 + i * 2, "los": i % 5, "gender": "M" if i % 2 == 0 else "F"})
            for i in range(12)
        ]
        anon = MondrianAnonymizer(k=3)
        anonymized = anon.anonymize_records(records, qi_numeric=["age", "los"], qi_categorical=["gender"])
        self.assertEqual(len(anonymized), 12)
        # All records should receive generalized intervals
        for r in anonymized:
            self.assertIn("age", r)
            self.assertTrue(isinstance(r["age"], str))


class TestAnonymityGuardPipeline(unittest.TestCase):
    def test_full_pipeline_audit(self):
        records = [
            DatasetRecord(1, {"age_grp": "20-29", "zip": "100**", "diag": "Flu"}),
            DatasetRecord(2, {"age_grp": "20-29", "zip": "100**", "diag": "Asthma"}),
            DatasetRecord(3, {"age_grp": "20-29", "zip": "100**", "diag": "Diabetes"}),
            DatasetRecord(4, {"age_grp": "40-49", "zip": "902**", "diag": "Flu"}),
            DatasetRecord(5, {"age_grp": "40-49", "zip": "902**", "diag": "Asthma"}),
            DatasetRecord(6, {"age_grp": "40-49", "zip": "902**", "diag": "Diabetes"}),
        ]
        pipeline = AnonymityGuardPipeline()
        report = pipeline.audit_dataset(
            records=records,
            quasi_identifiers=["age_grp", "zip"],
            sensitive_attribute="diag",
            k=3,
            l=3,
            t=0.1
        )
        self.assertTrue(report["overall_compliant"])
        self.assertTrue(report["k_anonymity"]["is_k_anonymous"])
        self.assertTrue(report["distinct_l_diversity"]["is_l_diverse"])
        self.assertTrue(report["t_closeness"]["is_t_close"])


if __name__ == "__main__":
    unittest.main()
