#!/usr/bin/env python3
"""
Test Suite Wrapper for Anonymity Guard.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from test_anonymity_klt import (
    TestGeneralizationHelpers,
    TestEquivalenceClasses,
    TestKAnonymity,
    TestLDiversity,
    TestTCloseness,
    TestPrivacyRiskAudit,
    TestMondrianAnonymizer,
    TestAnonymityGuardPipeline,
)

if __name__ == "__main__":
    unittest.main()
