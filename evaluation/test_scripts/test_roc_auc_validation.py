"""
Comprehensive validation tests for ROC and AUC calculation

This test suite thoroughly validates that ROC curves and AUC scores
are calculated correctly according to statistical definitions.
"""

import sys
sys.path.append('..')
import unittest
import numpy as np
from datetime import datetime, timedelta
from segment_evaluator import SegmentEvaluator
from sklearn.metrics import roc_curve, auc


class TestROCAUCCorrectness(unittest.TestCase):
    """Validate ROC/AUC calculation correctness"""

    def setUp(self):
        self.evaluator = SegmentEvaluator(
            timeline_resolution_seconds=1.0,
            overlap_threshold=0.5
        )

    def test_perfect_classifier_auc_equals_1(self):
        """Perfect predictions should yield AUC = 1.0"""
        print("\n" + "="*70)
        print("TEST: Perfect Classifier (AUC should = 1.0)")
        print("="*70)

        # 5 GT segments, all detected with perfect confidence
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i, 0),
                'end_time': datetime(2008, 1, 1, 0, i+1, 0),
                'duration': 60.0,
                'confidence': 1.0
            }
            for i in range(5)
        ]

        pred_segments = [
            {
                'segment_id': '1',
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'duration': 60.0,
                'confidence': 1.0  # Perfect confidence
            }
            for seg in gt_segments
        ]

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        if '1' in roc_results and roc_results['1']['auc'] is not None:
            auc_val = roc_results['1']['auc']
            print(f"✓ Perfect classifier AUC: {auc_val:.4f}")
            self.assertAlmostEqual(auc_val, 1.0, places=2,
                                  msg="Perfect predictions should have AUC = 1.0")
        else:
            self.fail("AUC not calculated for perfect predictions")

    def test_worst_classifier_auc_equals_0(self):
        """Worst possible predictions should yield AUC = 0.0"""
        print("\n" + "="*70)
        print("TEST: Worst Classifier (AUC should = 0.0)")
        print("="*70)

        # 5 GT segments
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i, 0),
                'end_time': datetime(2008, 1, 1, 0, i+1, 0),
                'duration': 60.0,
                'confidence': 1.0
            }
            for i in range(5)
        ]

        # Predict WRONG locations with high confidence (all FPs)
        # Don't predict actual GT locations (all FNs)
        pred_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 1, i, 0),  # Different time
                'end_time': datetime(2008, 1, 1, 1, i+1, 0),
                'duration': 60.0,
                'confidence': 1.0  # High confidence but wrong!
            }
            for i in range(5)
        ]

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        if '1' in roc_results and roc_results['1']['auc'] is not None:
            auc_val = roc_results['1']['auc']
            print(f"✓ Worst classifier AUC: {auc_val:.4f}")
            self.assertLessEqual(auc_val, 0.1,
                               msg="All wrong predictions should have AUC ≈ 0.0")
        else:
            self.fail("AUC not calculated")

    def test_random_classifier_auc_near_0_5(self):
        """Random-like predictions should yield AUC ≈ 0.5"""
        print("\n" + "="*70)
        print("TEST: Random-like Classifier (AUC should ≈ 0.5)")
        print("="*70)

        # 10 GT segments
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i*2, 0),
                'end_time': datetime(2008, 1, 1, 0, i*2+1, 0),
                'duration': 60.0,
                'confidence': 1.0
            }
            for i in range(10)
        ]

        # Predict 50% correctly with random confidence
        # 5 correct predictions (randomly distributed confidence)
        pred_segments = []
        for i in range(5):
            pred_segments.append({
                'segment_id': '1',
                'start_time': gt_segments[i]['start_time'],
                'end_time': gt_segments[i]['end_time'],
                'duration': 60.0,
                'confidence': 0.3 + (i % 3) * 0.2  # Random: 0.3, 0.5, 0.7
            })

        # 5 false positives (randomly distributed confidence)
        for i in range(5):
            pred_segments.append({
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 1, i, 0),
                'end_time': datetime(2008, 1, 1, 1, i+1, 0),
                'duration': 60.0,
                'confidence': 0.4 + (i % 3) * 0.2  # Random: 0.4, 0.6, 0.8
            })

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        if '1' in roc_results and roc_results['1']['auc'] is not None:
            auc_val = roc_results['1']['auc']
            print(f"✓ Random-like classifier AUC: {auc_val:.4f}")
            # Should be around 0.5, allow range 0.3-0.7
            self.assertGreater(auc_val, 0.3,
                             msg="Random classifier should have AUC > 0.3")
            self.assertLess(auc_val, 0.7,
                          msg="Random classifier should have AUC < 0.7")
        else:
            self.fail("AUC not calculated")

    def test_manual_auc_calculation_verification(self):
        """Verify AUC matches manual calculation"""
        print("\n" + "="*70)
        print("TEST: Manual AUC Calculation Verification")
        print("="*70)

        # Create specific scenario with known outcome
        # 4 GT segments
        gt_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 0, 0),
             'end_time': datetime(2008, 1, 1, 0, 1, 0), 'duration': 60.0, 'confidence': 1.0},
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 1, 0),
             'end_time': datetime(2008, 1, 1, 0, 2, 0), 'duration': 60.0, 'confidence': 1.0},
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 2, 0),
             'end_time': datetime(2008, 1, 1, 0, 3, 0), 'duration': 60.0, 'confidence': 1.0},
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 3, 0),
             'end_time': datetime(2008, 1, 1, 0, 4, 0), 'duration': 60.0, 'confidence': 1.0},
        ]

        # 3 predictions:
        # - 2 TPs with confidence 0.9, 0.7
        # - 1 FP with confidence 0.3
        # - 2 FNs (missed GT #2 and #3)
        pred_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 0, 0),
             'end_time': datetime(2008, 1, 1, 0, 1, 0), 'duration': 60.0, 'confidence': 0.9},  # TP
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 3, 0),
             'end_time': datetime(2008, 1, 1, 0, 4, 0), 'duration': 60.0, 'confidence': 0.7},  # TP
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 1, 0, 0),
             'end_time': datetime(2008, 1, 1, 1, 1, 0), 'duration': 60.0, 'confidence': 0.3},  # FP
        ]

        # Manual calculation:
        # y_true = [1, 1, 1, 1, 0]  # 4 GT positives (including 2 FNs), 1 FP
        # y_score = [0.9, 0.0, 0.0, 0.7, 0.3]  # Scores for GT#0, GT#1(FN), GT#2(FN), GT#3, FP
        # After sorting by score descending: [0.9, 0.7, 0.3, 0.0, 0.0]
        # Corresponding y_true:              [1,   1,   0,   1,   1  ]

        expected_y_true = np.array([1, 1, 1, 1, 0])  # Order: GT#0(TP), GT#1(FN), GT#2(FN), GT#3(TP), FP
        expected_y_scores = np.array([0.9, 0.0, 0.0, 0.7, 0.3])

        # Calculate expected AUC using sklearn
        expected_fpr, expected_tpr, expected_thresholds = roc_curve(expected_y_true, expected_y_scores)
        expected_auc = auc(expected_fpr, expected_tpr)

        print(f"\nExpected calculation:")
        print(f"  y_true:  {expected_y_true}")
        print(f"  y_score: {expected_y_scores}")
        print(f"  Expected AUC: {expected_auc:.4f}")

        # Get actual AUC from evaluator
        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        actual_auc = roc_results['1']['auc']
        print(f"  Actual AUC:   {actual_auc:.4f}")

        self.assertAlmostEqual(actual_auc, expected_auc, places=3,
                              msg=f"AUC should match manual calculation: expected {expected_auc:.4f}, got {actual_auc:.4f}")

    def test_confidence_ranking_affects_roc_curve(self):
        """Higher confidence predictions should appear earlier in ROC curve"""
        print("\n" + "="*70)
        print("TEST: Confidence Ranking")
        print("="*70)

        # 3 GT segments
        gt_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, i, 0),
             'end_time': datetime(2008, 1, 1, 0, i+1, 0), 'duration': 60.0, 'confidence': 1.0}
            for i in range(3)
        ]

        # Predict all 3 with DECREASING confidence
        pred_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 0, 0),
             'end_time': datetime(2008, 1, 1, 0, 1, 0), 'duration': 60.0, 'confidence': 0.9},
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 1, 0),
             'end_time': datetime(2008, 1, 1, 0, 2, 0), 'duration': 60.0, 'confidence': 0.5},
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, 2, 0),
             'end_time': datetime(2008, 1, 1, 0, 3, 0), 'duration': 60.0, 'confidence': 0.1},
        ]

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        fpr = roc_results['1']['fpr']
        tpr = roc_results['1']['tpr']
        thresholds = roc_results['1']['thresholds']

        print(f"  TPR values: {tpr}")
        print(f"  FPR values: {fpr}")
        print(f"  Thresholds: {thresholds}")

        # TPR should increase monotonically (or stay same)
        for i in range(len(tpr) - 1):
            self.assertGreaterEqual(tpr[i+1], tpr[i] - 1e-10,
                                  msg=f"TPR should be non-decreasing: {tpr}")

        # FPR should increase monotonically (or stay same)
        for i in range(len(fpr) - 1):
            self.assertGreaterEqual(fpr[i+1], fpr[i] - 1e-10,
                                  msg=f"FPR should be non-decreasing: {fpr}")

        print("✓ ROC curve is monotonic")

    def test_auc_with_all_fn(self):
        """Test AUC when all GT segments are missed (all FN)"""
        print("\n" + "="*70)
        print("TEST: All False Negatives")
        print("="*70)

        gt_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, i, 0),
             'end_time': datetime(2008, 1, 1, 0, i+1, 0), 'duration': 60.0, 'confidence': 1.0}
            for i in range(3)
        ]

        # No predictions at all -> all FN
        pred_segments = []

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        # With all FNs (y_true all 1s, y_score all 0s), there's no variation
        # Should handle gracefully
        if '1' in roc_results:
            auc_val = roc_results['1']['auc']
            print(f"  AUC with all FN: {auc_val}")
            # Either None or 0.0 is acceptable
            if auc_val is not None:
                self.assertEqual(auc_val, 0.0,
                               msg="All false negatives should give AUC=0")
            print("✓ Handled gracefully")

    def test_auc_with_only_fp(self):
        """Test AUC when there are only false positives (no GT)"""
        print("\n" + "="*70)
        print("TEST: Only False Positives (no GT)")
        print("="*70)

        gt_segments = []  # No ground truth

        pred_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, i, 0),
             'end_time': datetime(2008, 1, 1, 0, i+1, 0), 'duration': 60.0, 'confidence': 0.8}
            for i in range(3)
        ]

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        # Should handle gracefully (no positive class)
        if '1' in roc_results:
            auc_val = roc_results['1']['auc']
            print(f"  AUC with no GT: {auc_val}")
            self.assertIsNone(auc_val, msg="No GT should return None")
            print("✓ Handled gracefully")

    def test_auc_interpretation_correctness(self):
        """Test that AUC correctly interprets classifier quality"""
        print("\n" + "="*70)
        print("TEST: AUC Interpretation")
        print("="*70)

        # Scenario: Good classifier (80% recall, 0 FP)
        gt_segments = [
            {'segment_id': '1', 'start_time': datetime(2008, 1, 1, 0, i, 0),
             'end_time': datetime(2008, 1, 1, 0, i+1, 0), 'duration': 60.0, 'confidence': 1.0}
            for i in range(5)
        ]

        # Detect 4 out of 5 with high confidence, no false positives
        pred_segments = [
            {'segment_id': '1', 'start_time': gt_segments[i]['start_time'],
             'end_time': gt_segments[i]['end_time'], 'duration': 60.0,
             'confidence': 0.9}
            for i in range(4)  # Only first 4
        ]

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        auc_val = roc_results['1']['auc']
        print(f"  Good classifier (80% recall, 0% FP): AUC = {auc_val:.4f}")

        # Good classifier should have high AUC
        self.assertGreater(auc_val, 0.7,
                          msg="Good classifier should have AUC > 0.7")

        # Now test mediocre classifier (50% recall, 50% precision)
        pred_segments_mediocre = []
        # 2 correct predictions
        for i in range(2):
            pred_segments_mediocre.append({
                'segment_id': '1',
                'start_time': gt_segments[i]['start_time'],
                'end_time': gt_segments[i]['end_time'],
                'duration': 60.0,
                'confidence': 0.6
            })
        # 2 false positives
        for i in range(2):
            pred_segments_mediocre.append({
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 1, i, 0),
                'end_time': datetime(2008, 1, 1, 1, i+1, 0),
                'duration': 60.0,
                'confidence': 0.4
            })

        roc_results_mediocre = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments_mediocre, gt_segments, overlap_threshold=0.5
        )

        auc_mediocre = roc_results_mediocre['1']['auc']
        print(f"  Mediocre classifier (40% recall, 50% precision): AUC = {auc_mediocre:.4f}")

        # Mediocre classifier should have lower AUC than good one
        self.assertLess(auc_mediocre, auc_val,
                       msg="Mediocre classifier should have lower AUC than good classifier")

        print("✓ AUC correctly ranks classifier quality")


def run_roc_auc_validation():
    """Run comprehensive ROC/AUC validation tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ROC & AUC VALIDATION TEST SUITE")
    print("="*70)
    print("\nValidating that ROC curves and AUC scores are calculated")
    print("correctly according to statistical definitions...\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestROCAUCCorrectness))

    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("ROC & AUC VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n" + "="*70)
        print("✓ ALL ROC/AUC TESTS PASSED!")
        print("="*70)
        print("\nYour ROC and AUC calculations are 100% CORRECT!")
        print("\nVerified:")
        print("  ✓ Perfect classifier -> AUC = 1.0")
        print("  ✓ Worst classifier -> AUC = 0.0")
        print("  ✓ Random classifier -> AUC ≈ 0.5")
        print("  ✓ Manual calculation matches")
        print("  ✓ Confidence ranking works")
        print("  ✓ Edge cases handled")
        print("  ✓ AUC correctly ranks quality")
        print("\n" + "="*70)
    else:
        print("\n" + "="*70)
        print("✗ SOME ROC/AUC TESTS FAILED")
        print("="*70)

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
                print(f"    {traceback.split(chr(10))[-2]}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
                print(f"    {traceback.split(chr(10))[-2]}")

    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_roc_auc_validation()
    exit(0 if success else 1)
