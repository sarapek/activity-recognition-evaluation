"""
Comprehensive unit tests for evaluation metrics

This test suite validates that all evaluation metrics work correctly according to
the specifications in README.md. It's designed to be highly critical and catch
edge cases, corner cases, and potential bugs.
"""

import sys
sys.path.append('..')
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from segment_evaluator import SegmentEvaluator
import tempfile
import os


class TestSegmentEvaluator(unittest.TestCase):
    """Test core SegmentEvaluator functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = SegmentEvaluator(
            timeline_resolution_seconds=1.0,
            start_time_threshold_seconds=0,
            overlap_threshold=0.5
        )

    def test_perfect_match_should_yield_perfect_metrics(self):
        """Test that perfect predictions yield F1=1.0, Precision=1.0, Recall=1.0"""
        # Create identical segments
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            },
            {
                'segment_id': '2',
                'start_time': datetime(2008, 1, 1, 0, 10, 0),
                'end_time': datetime(2008, 1, 1, 0, 20, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        # Evaluate against itself
        results = self.evaluator.evaluate_segments(segments, segments)

        # All metrics should be perfect
        self.assertAlmostEqual(results['micro_avg']['f1_score'], 1.0, places=4,
                              msg="Perfect match should have F1=1.0")
        self.assertAlmostEqual(results['micro_avg']['precision'], 1.0, places=4,
                              msg="Perfect match should have Precision=1.0")
        self.assertAlmostEqual(results['micro_avg']['recall'], 1.0, places=4,
                              msg="Perfect match should have Recall=1.0")
        self.assertAlmostEqual(results['macro_avg']['f1_score'], 1.0, places=4,
                              msg="Perfect match should have Macro F1=1.0")

        # Per-class metrics should also be perfect
        for activity_id in ['1', '2']:
            metrics = results['per_class'][activity_id]
            self.assertAlmostEqual(metrics['precision'], 1.0, places=4,
                                 msg=f"Activity {activity_id} should have perfect precision")
            self.assertAlmostEqual(metrics['recall'], 1.0, places=4,
                                 msg=f"Activity {activity_id} should have perfect recall")
            self.assertAlmostEqual(metrics['f1_score'], 1.0, places=4,
                                 msg=f"Activity {activity_id} should have perfect F1")
            self.assertAlmostEqual(metrics['iou'], 1.0, places=4,
                                 msg=f"Activity {activity_id} should have perfect IoU")

    def test_missing_segments_should_show_low_recall_high_precision(self):
        """Test that missing 50% of segments shows Recall=0.5, Precision=1.0"""
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            },
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 20, 0),
                'end_time': datetime(2008, 1, 1, 0, 30, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        # Only predict first segment
        pred_segments = [gt_segments[0].copy()]

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)

        # Should have perfect precision (no false positives)
        # But 50% recall (missed half the segments)
        metrics = results['per_class']['1']
        self.assertAlmostEqual(metrics['precision'], 1.0, places=4,
                              msg="Missing segments should still have perfect precision")
        self.assertAlmostEqual(metrics['recall'], 0.5, places=4,
                              msg="Missing 50% of segments should give Recall=0.5")

        # F1 should be harmonic mean
        expected_f1 = 2 * (1.0 * 0.5) / (1.0 + 0.5)
        self.assertAlmostEqual(metrics['f1_score'], expected_f1, places=4,
                              msg=f"F1 should be {expected_f1:.4f}")

    def test_false_positives_should_show_low_precision_high_recall(self):
        """Test that extra false detections show low precision, high recall"""
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        # Predict correctly + add false positive
        pred_segments = [
            gt_segments[0].copy(),
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 20, 0),
                'end_time': datetime(2008, 1, 1, 0, 30, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)

        # Should have perfect recall (found all GT segments)
        # But 50% precision (half predictions are false)
        metrics = results['per_class']['1']
        self.assertAlmostEqual(metrics['recall'], 1.0, places=4,
                              msg="False positives should still have perfect recall")
        self.assertAlmostEqual(metrics['precision'], 0.5, places=4,
                              msg="50% false positives should give Precision=0.5")

    def test_empty_predictions_should_give_zero_metrics(self):
        """Test that no predictions yields Precision=0, Recall=0"""
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        pred_segments = []

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)

        metrics = results['per_class']['1']
        self.assertEqual(metrics['precision'], 0.0,
                        msg="Empty predictions should have Precision=0")
        self.assertEqual(metrics['recall'], 0.0,
                        msg="Empty predictions should have Recall=0")
        self.assertEqual(metrics['f1_score'], 0.0,
                        msg="Empty predictions should have F1=0")

    def test_empty_ground_truth_should_handle_gracefully(self):
        """Test that empty GT with predictions handles gracefully"""
        pred_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        gt_segments = []

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)

        # Should not crash and should handle gracefully
        self.assertIsInstance(results, dict,
                            msg="Empty GT should return valid results dict")

    def test_timeline_resolution_affects_metrics(self):
        """Test that timeline resolution is properly applied"""
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 0, 10),  # 10 seconds
                'duration': 10.0,
                'confidence': 1.0
            }
        ]

        # Test with 1-second resolution
        evaluator_1s = SegmentEvaluator(timeline_resolution_seconds=1.0)
        start = datetime(2008, 1, 1, 0, 0, 0)
        end = datetime(2008, 1, 1, 0, 0, 10)
        timeline, _ = evaluator_1s.create_timeline(segments, start, end)

        # Should have 10 time units
        filled_frames = np.sum(timeline == '1')
        self.assertEqual(filled_frames, 10,
                        msg="1-second resolution should create 10 frames for 10-second segment")

        # Test with 5-second resolution
        evaluator_5s = SegmentEvaluator(timeline_resolution_seconds=5.0)
        timeline, _ = evaluator_5s.create_timeline(segments, start, end)

        # Should have 2 time units
        filled_frames = np.sum(timeline == '1')
        self.assertEqual(filled_frames, 2,
                        msg="5-second resolution should create 2 frames for 10-second segment")

    def test_overlap_threshold_affects_matching(self):
        """Test that overlap threshold correctly filters matches"""
        gt_seg = {
            'segment_id': '1',
            'start_time': datetime(2008, 1, 1, 0, 0, 0),
            'end_time': datetime(2008, 1, 1, 0, 0, 10),  # 10 seconds
            'duration': 10.0,
            'confidence': 1.0
        }

        # Prediction with 6 seconds overlap (60%)
        pred_seg_60 = {
            'segment_id': '1',
            'start_time': datetime(2008, 1, 1, 0, 0, 4),  # Starts 4s late
            'end_time': datetime(2008, 1, 1, 0, 0, 10),
            'duration': 6.0,
            'confidence': 1.0
        }

        # With 50% threshold: should match (60% > 50%)
        evaluator_50 = SegmentEvaluator(overlap_threshold=0.5)
        results_50 = evaluator_50.evaluate_segments([pred_seg_60], [gt_seg])
        metrics_50 = results_50['per_class']['1']

        # Should detect the segment
        self.assertGreater(metrics_50['recall'], 0.0,
                          msg="60% overlap should be detected with 50% threshold")

        # With 70% threshold: should NOT match (60% < 70%)
        evaluator_70 = SegmentEvaluator(overlap_threshold=0.7)
        results_70 = evaluator_70.evaluate_segments([pred_seg_60], [gt_seg])
        metrics_70 = results_70['per_class']['1']

        # Should NOT detect the segment
        self.assertEqual(metrics_70['recall'], 0.0,
                        msg="60% overlap should NOT be detected with 70% threshold")

    def test_macro_vs_micro_averaging(self):
        """Test that macro and micro averaging differ correctly"""
        # Create imbalanced data:
        # Activity 1: 1000 seconds, perfect prediction
        # Activity 2: 100 seconds, 50% recall

        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 16, 40),  # 1000 seconds
                'duration': 1000.0,
                'confidence': 1.0
            },
            {
                'segment_id': '2',
                'start_time': datetime(2008, 1, 1, 1, 0, 0),
                'end_time': datetime(2008, 1, 1, 1, 0, 50),  # 50 seconds
                'duration': 50.0,
                'confidence': 1.0
            },
            {
                'segment_id': '2',
                'start_time': datetime(2008, 1, 1, 2, 0, 0),
                'end_time': datetime(2008, 1, 1, 2, 0, 50),  # 50 seconds
                'duration': 50.0,
                'confidence': 1.0
            }
        ]

        pred_segments = [
            gt_segments[0].copy(),  # Perfect for activity 1
            gt_segments[1].copy()   # Only one of activity 2 (50% recall)
        ]

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)

        # Macro: (1.0 + 0.5) / 2 = 0.75
        # Micro: weighted by duration, closer to 1.0
        macro_recall = results['macro_avg']['recall']
        micro_recall = results['micro_avg']['recall']

        self.assertAlmostEqual(macro_recall, 0.75, places=2,
                              msg="Macro recall should be 0.75 (average of 1.0 and 0.5)")
        self.assertGreater(micro_recall, macro_recall,
                          msg="Micro recall should be higher due to larger weight of activity 1")

    def test_iou_calculation(self):
        """Test that IoU is correctly calculated as TP / (TP + FP + FN)"""
        # Create segment with 80% overlap
        gt_seg = {
            'segment_id': '1',
            'start_time': datetime(2008, 1, 1, 0, 0, 0),
            'end_time': datetime(2008, 1, 1, 0, 0, 100),  # 100 seconds
            'duration': 100.0,
            'confidence': 1.0
        }

        pred_seg = {
            'segment_id': '1',
            'start_time': datetime(2008, 1, 1, 0, 0, 10),  # 10s late
            'end_time': datetime(2008, 1, 1, 0, 0, 110),   # 10s extra
            'duration': 100.0,
            'confidence': 1.0
        }

        results = self.evaluator.evaluate_segments([pred_seg], [gt_seg])
        metrics = results['per_class']['1']

        # TP = 80s (overlap)
        # FP = 20s (predicted but not in GT: 100-110 and extra from shift)
        # FN = 20s (in GT but not predicted: 0-10 and missing from shift)
        # Wait, let me recalculate with timeline approach:
        # Timeline from 0 to 110
        # GT: 0-100 (100 units)
        # Pred: 10-110 (100 units)
        # Overlap: 10-100 (90 units) = TP
        # FP: 100-110 (10 units)
        # FN: 0-10 (10 units)
        # IoU = 90 / (90 + 10 + 10) = 90/110 = 0.818...

        expected_iou = 90.0 / 110.0
        self.assertAlmostEqual(metrics['iou'], expected_iou, places=2,
                              msg=f"IoU should be {expected_iou:.4f}")

    def test_label_confusion_affects_per_class_metrics(self):
        """Test that swapped labels result in low metrics for affected classes"""
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            },
            {
                'segment_id': '2',
                'start_time': datetime(2008, 1, 1, 0, 10, 0),
                'end_time': datetime(2008, 1, 1, 0, 20, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        # Swap labels
        pred_segments = [
            {
                'segment_id': '2',  # Should be '1'
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            },
            {
                'segment_id': '1',  # Should be '2'
                'start_time': datetime(2008, 1, 1, 0, 10, 0),
                'end_time': datetime(2008, 1, 1, 0, 20, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)

        # Both activities should have 0% metrics
        for activity_id in ['1', '2']:
            metrics = results['per_class'][activity_id]
            self.assertEqual(metrics['recall'], 0.0,
                           msg=f"Swapped labels should give 0% recall for activity {activity_id}")
            self.assertEqual(metrics['precision'], 0.0,
                           msg=f"Swapped labels should give 0% precision for activity {activity_id}")


class TestFileParsingAndSegmentExtraction(unittest.TestCase):
    """Test file parsing and segment extraction"""

    def setUp(self):
        self.evaluator = SegmentEvaluator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_ground_truth_format(self):
        """Test parsing ground truth format: date time sensor status activity"""
        content = """2008-02-27 00:07:46.898610 M003 ON 1
2008-02-27 00:10:32.539722 M004 OFF 2
2008-02-27 00:12:15.268912 M005 ON 1"""

        filepath = os.path.join(self.temp_dir, 'test_gt.txt')
        with open(filepath, 'w') as f:
            f.write(content)

        df = self.evaluator.parse_log_file(filepath)

        self.assertEqual(len(df), 3, msg="Should parse 3 events")
        self.assertEqual(df.iloc[0]['annotation'], '1',
                        msg="First event should be activity 1")
        self.assertEqual(df.iloc[1]['annotation'], '2',
                        msg="Second event should be activity 2")
        self.assertIn('timestamp', df.columns,
                     msg="Should have timestamp column")

    def test_parse_prediction_format_with_confidence(self):
        """Test parsing prediction format with confidence scores"""
        content = """2008-02-27 00:07:46.898610 M003 M003 ON 1 0.945
2008-02-27 00:10:32.539722 M004 M004 OFF 2 0.876"""

        filepath = os.path.join(self.temp_dir, 'test_pred.txt')
        with open(filepath, 'w') as f:
            f.write(content)

        df = self.evaluator.parse_log_file(filepath)

        self.assertEqual(len(df), 2, msg="Should parse 2 events")
        self.assertIn('confidence', df.columns,
                     msg="Should have confidence column")
        self.assertAlmostEqual(df.iloc[0]['confidence'], 0.945, places=3,
                              msg="First event should have confidence 0.945")

    def test_extract_segments_with_continuous_labels(self):
        """Test segment extraction with continuous activity labels"""
        data = pd.DataFrame({
            'timestamp': [
                datetime(2008, 1, 1, 0, 0, 0),
                datetime(2008, 1, 1, 0, 5, 0),
                datetime(2008, 1, 1, 0, 10, 0),  # Activity change
                datetime(2008, 1, 1, 0, 15, 0),
            ],
            'event_id': ['M1', 'M2', 'M3', 'M4'],
            'state': ['ON', 'OFF', 'ON', 'OFF'],
            'annotation': ['1', '1', '2', '2']
        })

        segments = self.evaluator.extract_segments(data)

        self.assertEqual(len(segments), 2,
                        msg="Should extract 2 segments (one per activity)")
        self.assertEqual(segments[0]['segment_id'], '1',
                        msg="First segment should be activity 1")
        self.assertEqual(segments[1]['segment_id'], '2',
                        msg="Second segment should be activity 2")

    def test_extract_segments_with_start_end_markers(self):
        """Test segment extraction with start/end markers"""
        data = pd.DataFrame({
            'timestamp': [
                datetime(2008, 1, 1, 0, 0, 0),
                datetime(2008, 1, 1, 0, 10, 0),
            ],
            'event_id': ['M1', 'M2'],
            'state': ['ON', 'OFF'],
            'annotation': ['1-start', '1-end']
        })

        segments = self.evaluator.extract_segments(data)

        self.assertEqual(len(segments), 1,
                        msg="Should extract 1 segment from start/end markers")
        self.assertEqual(segments[0]['segment_id'], '1',
                        msg="Segment should be activity 1")
        self.assertAlmostEqual(segments[0]['duration'], 600.0, places=1,
                              msg="Duration should be 10 minutes = 600 seconds")

    def test_extract_segments_filters_other_activity(self):
        """Test that Other_Activity is filtered out"""
        data = pd.DataFrame({
            'timestamp': [
                datetime(2008, 1, 1, 0, 0, 0),
                datetime(2008, 1, 1, 0, 5, 0),
                datetime(2008, 1, 1, 0, 10, 0),
            ],
            'event_id': ['M1', 'M2', 'M3'],
            'state': ['ON', 'OFF', 'ON'],
            'annotation': ['1', 'Other_Activity', '2']
        })

        segments = self.evaluator.extract_segments(data)

        # Should have segments for 1 and 2, but Other_Activity should be gap
        activity_ids = [seg['segment_id'] for seg in segments]
        self.assertNotIn('Other_Activity', activity_ids,
                        msg="Other_Activity should not appear in segments")


class TestROCCalculation(unittest.TestCase):
    """Test ROC curve and AUC calculation"""

    def setUp(self):
        self.evaluator = SegmentEvaluator()

    def test_perfect_predictions_yield_auc_1(self):
        """Test that perfect predictions with perfect confidence yield AUC=1.0"""
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        # Test segment-level ROC
        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            segments, segments, overlap_threshold=0.5
        )

        if '1' in roc_results and roc_results['1']['auc'] is not None:
            self.assertAlmostEqual(roc_results['1']['auc'], 1.0, places=2,
                                  msg="Perfect predictions should yield AUC=1.0")

    def test_varied_confidence_affects_auc(self):
        """Test that confidence score variation affects AUC calculation"""
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i, 0),
                'end_time': datetime(2008, 1, 1, 0, i+1, 0),
                'duration': 60.0,
                'confidence': 1.0
            }
            for i in range(5)  # 5 GT segments
        ]

        # Predict with varying confidence
        pred_segments = [
            {
                'segment_id': '1',
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'duration': 60.0,
                'confidence': 0.9 - i * 0.1  # Decreasing confidence
            }
            for i, seg in enumerate(gt_segments[:3])  # Only predict first 3
        ]

        # Add false positives with low confidence
        pred_segments.append({
            'segment_id': '1',
            'start_time': datetime(2008, 1, 1, 1, 0, 0),
            'end_time': datetime(2008, 1, 1, 1, 1, 0),
            'duration': 60.0,
            'confidence': 0.2
        })

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            pred_segments, gt_segments, overlap_threshold=0.5
        )

        if '1' in roc_results and roc_results['1']['auc'] is not None:
            auc_val = roc_results['1']['auc']
            # AUC should be less than 1.0 but reasonable
            self.assertLess(auc_val, 1.0,
                           msg="Imperfect predictions should have AUC < 1.0")
            self.assertGreater(auc_val, 0.5,
                              msg="Reasonable predictions should have AUC > 0.5")

    def test_no_variation_in_scores_handles_gracefully(self):
        """Test that constant confidence scores are handled gracefully"""
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i, 0),
                'end_time': datetime(2008, 1, 1, 0, i+1, 0),
                'duration': 60.0,
                'confidence': 0.5  # All same confidence
            }
            for i in range(3)
        ]

        roc_results = self.evaluator.calculate_segment_roc_auc_per_class(
            segments, segments, overlap_threshold=0.5
        )

        # Should return None or handle gracefully without crashing
        self.assertIsInstance(roc_results, dict,
                            msg="Should return dict even with constant confidence")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and corner cases"""

    def setUp(self):
        self.evaluator = SegmentEvaluator()

    def test_zero_duration_segments(self):
        """Test handling of zero-duration segments"""
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 0, 0),  # Same time
                'duration': 0.0,
                'confidence': 1.0
            }
        ]

        # Should not crash
        try:
            results = self.evaluator.evaluate_segments(segments, segments)
            self.assertIsInstance(results, dict,
                                msg="Zero duration should be handled gracefully")
        except Exception as e:
            self.fail(f"Zero duration segments caused crash: {e}")

    def test_very_small_segments(self):
        """Test handling of sub-second segments"""
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 0, 0, 100000),  # 0.1 second
                'duration': 0.1,
                'confidence': 1.0
            }
        ]

        results = self.evaluator.evaluate_segments(segments, segments)
        # Should still work, though metrics might be affected by resolution
        self.assertIsInstance(results, dict)

    def test_segments_spanning_days(self):
        """Test handling of long segments spanning multiple days"""
        segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 3, 0, 0, 0),  # 2 days
                'duration': 172800.0,
                'confidence': 1.0
            }
        ]

        results = self.evaluator.evaluate_segments(segments, segments)
        metrics = results['per_class']['1']

        self.assertAlmostEqual(metrics['precision'], 1.0, places=4,
                              msg="Long segments should work correctly")
        self.assertAlmostEqual(metrics['recall'], 1.0, places=4,
                              msg="Long segments should work correctly")

    def test_many_small_overlapping_segments(self):
        """Test handling of many overlapping segments"""
        # Create 100 overlapping 10-second segments
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, 0, i),
                'end_time': datetime(2008, 1, 1, 0, 0, i+10),
                'duration': 10.0,
                'confidence': 1.0
            }
            for i in range(0, 100, 5)  # Overlapping every 5 seconds
        ]

        pred_segments = gt_segments.copy()

        # Should handle without issues
        try:
            results = self.evaluator.evaluate_segments(pred_segments, gt_segments)
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Many overlapping segments caused issues: {e}")

    def test_non_sequential_activity_ids(self):
        """Test handling of non-sequential activity IDs"""
        segments = [
            {
                'segment_id': '15',  # Non-sequential
                'start_time': datetime(2008, 1, 1, 0, 0, 0),
                'end_time': datetime(2008, 1, 1, 0, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            },
            {
                'segment_id': '3',
                'start_time': datetime(2008, 1, 1, 0, 10, 0),
                'end_time': datetime(2008, 1, 1, 0, 20, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
        ]

        results = self.evaluator.evaluate_segments(segments, segments)

        # Should have metrics for both activities
        self.assertIn('15', results['per_class'],
                     msg="Should handle activity ID 15")
        self.assertIn('3', results['per_class'],
                     msg="Should handle activity ID 3")


class TestIntegrationWithFiles(unittest.TestCase):
    """Integration tests with actual file reading"""

    def setUp(self):
        self.evaluator = SegmentEvaluator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluate_with_dual_roc_integration(self):
        """Test full integration of evaluate_with_dual_roc"""
        # Create test files
        gt_content = """2008-02-27 00:00:00.000000 M001 ON 1
2008-02-27 00:10:00.000000 M002 ON 2
2008-02-27 00:20:00.000000 M003 ON 1"""

        pred_content = """2008-02-27 00:00:00.000000 M001 M001 ON 1 0.95
2008-02-27 00:10:00.000000 M002 M002 ON 2 0.87
2008-02-27 00:20:00.000000 M003 M003 ON 1 0.92"""

        gt_path = os.path.join(self.temp_dir, 'gt.txt')
        pred_path = os.path.join(self.temp_dir, 'pred.txt')

        with open(gt_path, 'w') as f:
            f.write(gt_content)
        with open(pred_path, 'w') as f:
            f.write(pred_content)

        # Should not crash
        try:
            results = self.evaluator.evaluate_with_dual_roc(
                pred_path, gt_path, aggregation='average'
            )

            # Check structure
            self.assertIn('per_class', results,
                         msg="Results should have per_class metrics")
            self.assertIn('frame_level', results,
                         msg="Results should have frame_level ROC")
            self.assertIn('segment_level', results,
                         msg="Results should have segment_level ROC")
            self.assertIn('metadata', results,
                         msg="Results should have metadata")

        except Exception as e:
            self.fail(f"evaluate_with_dual_roc failed: {e}")


class TestSanityCheckExpectations(unittest.TestCase):
    """
    Test expected behaviors from README.md sanity checks
    These tests validate what SHOULD happen with known error patterns
    """

    def setUp(self):
        self.evaluator = SegmentEvaluator(
            timeline_resolution_seconds=1.0,
            overlap_threshold=0.5
        )

    def test_sanity_perfect_expectations(self):
        """Perfect variant should have F1=1.0, all metrics perfect"""
        segments = [
            {
                'segment_id': str(i),
                'start_time': datetime(2008, 1, 1, i, 0, 0),
                'end_time': datetime(2008, 1, 1, i, 10, 0),
                'duration': 600.0,
                'confidence': 1.0
            }
            for i in range(1, 9)  # Activities 1-8
        ]

        results = self.evaluator.evaluate_segments(segments, segments)

        # All aggregate metrics should be 1.0
        self.assertAlmostEqual(results['micro_avg']['f1_score'], 1.0, places=4)
        self.assertAlmostEqual(results['micro_avg']['precision'], 1.0, places=4)
        self.assertAlmostEqual(results['micro_avg']['recall'], 1.0, places=4)
        self.assertAlmostEqual(results['macro_avg']['f1_score'], 1.0, places=4)

        # All per-class metrics should be 1.0
        for i in range(1, 9):
            metrics = results['per_class'][str(i)]
            self.assertAlmostEqual(metrics['f1_score'], 1.0, places=4,
                                 msg=f"Activity {i} should have perfect F1")

    def test_sanity_missing_20_percent_expectations(self):
        """Missing 20% of segments should show Precision≈1.0, Recall≈0.8"""
        # Create 10 GT segments
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i*10, 0),
                'end_time': datetime(2008, 1, 1, 0, i*10+5, 0),
                'duration': 300.0,
                'confidence': 1.0
            }
            for i in range(10)
        ]

        # Predict only 8 (missing 20%)
        pred_segments = gt_segments[:8]

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)
        metrics = results['per_class']['1']

        # Should have perfect precision (no false positives)
        self.assertAlmostEqual(metrics['precision'], 1.0, places=2,
                              msg="Missing segments should not create false positives")

        # Should have 80% recall (found 8 out of 10)
        self.assertAlmostEqual(metrics['recall'], 0.8, places=2,
                              msg="Missing 20% should give 80% recall")

        # F1 should be between precision and recall
        expected_f1 = 2 * (1.0 * 0.8) / (1.0 + 0.8)
        self.assertAlmostEqual(metrics['f1_score'], expected_f1, places=2)

    def test_sanity_false_positives_expectations(self):
        """False positives should show low precision, high recall"""
        # 5 GT segments
        gt_segments = [
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 0, i*10, 0),
                'end_time': datetime(2008, 1, 1, 0, i*10+5, 0),
                'duration': 300.0,
                'confidence': 1.0
            }
            for i in range(5)
        ]

        # Predict all 5 + 5 false positives
        pred_segments = gt_segments.copy()
        pred_segments.extend([
            {
                'segment_id': '1',
                'start_time': datetime(2008, 1, 1, 1, i*10, 0),
                'end_time': datetime(2008, 1, 1, 1, i*10+5, 0),
                'duration': 300.0,
                'confidence': 1.0
            }
            for i in range(5)
        ])

        results = self.evaluator.evaluate_segments(pred_segments, gt_segments)
        metrics = results['per_class']['1']

        # Should have perfect recall (found all GT)
        self.assertAlmostEqual(metrics['recall'], 1.0, places=2,
                              msg="Should find all GT segments")

        # Should have 50% precision (5 correct out of 10 predictions)
        self.assertAlmostEqual(metrics['precision'], 0.5, places=2,
                              msg="50% false positives should give 50% precision")


def run_comprehensive_tests():
    """Run all comprehensive tests and report results"""
    print("="*70)
    print("COMPREHENSIVE EVALUATION METRICS TEST SUITE")
    print("="*70)
    print("\nRunning all tests...\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSegmentEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestFileParsingAndSegmentExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestROCCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestSanityCheckExpectations))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
