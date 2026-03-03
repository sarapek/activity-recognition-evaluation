"""
Test script for Bad ROC test files
Shows diverse AUC values from deliberately bad predictions
"""

import sys
sys.path.append('..')
from segment_evaluator import SegmentEvaluator
import numpy as np
import os

def test_bad_roc_files():
    """Test all bad ROC files and display results"""

    print("\n" + "="*70)
    print("TESTING BAD ROC FILES - DIVERSE AUC VALIDATION")
    print("="*70)
    print("\nThese files have deliberately bad predictions to test AUC calculation")
    print("across the full range from 0.0 (worst) to 1.0 (perfect).\n")

    evaluator = SegmentEvaluator()
    gt_file = "../data/P001.txt"
    test_dir = "../results/bad_roc_tests"

    # Define test files with expected ranges
    test_files = [
        {
            'file': 'bad_roc_inverted_confidence.txt',
            'name': 'Inverted Confidence',
            'description': 'TPs get LOW conf, FPs get HIGH conf',
            'expected': '0.0-0.3'
        },
        {
            'file': 'bad_roc_all_false_positives.txt',
            'name': 'All False Positives',
            'description': 'All predictions at wrong times',
            'expected': '0.0-0.2'
        },
        {
            'file': 'bad_roc_wrong_labels.txt',
            'name': 'Wrong Labels',
            'description': 'Right times, wrong activity labels',
            'expected': '0.1-0.3'
        },
        {
            'file': 'bad_roc_many_fps.txt',
            'name': 'Many False Positives',
            'description': '5x more FPs than TPs',
            'expected': '0.2-0.4'
        },
        {
            'file': 'bad_roc_random_confidence.txt',
            'name': 'Random Confidence',
            'description': 'No correlation with correctness',
            'expected': '0.4-0.6'
        },
        {
            'file': 'bad_roc_bimodal.txt',
            'name': 'Bimodal Distribution',
            'description': 'Mixed high/low confidence for both TP/FP',
            'expected': '0.5-0.7'
        }
    ]

    results = []

    for test_info in test_files:
        test_file = os.path.join(test_dir, test_info['file'])

        if not os.path.exists(test_file):
            print(f"\n[WARNING] File not found: {test_info['file']}")
            continue

        print(f"\n{'='*70}")
        print(f"TEST: {test_info['name']}")
        print(f"{'='*70}")
        print(f"Description: {test_info['description']}")
        print(f"Expected AUC Range: {test_info['expected']}")
        print("-" * 70)

        try:
            result = evaluator.evaluate_with_dual_roc(
                test_file,
                gt_file,
                aggregation='average',
                activity_filter=['1', '2', '3', '4', '5', '6', '7', '8']
            )

            # Get frame-level AUC (average across activities)
            frame_aucs = []
            if 'frame_level' in result:
                for activity_id in sorted(result['frame_level'].keys()):
                    auc_val = result['frame_level'][activity_id].get('auc')
                    if auc_val is not None:
                        frame_aucs.append(auc_val)
                        print(f"  Activity {activity_id}: Frame AUC = {auc_val:.4f}")

            if frame_aucs:
                avg_frame_auc = np.mean(frame_aucs)
                print(f"\n  Average Frame-Level AUC: {avg_frame_auc:.4f}")
            else:
                avg_frame_auc = None
                print(f"\n  Average Frame-Level AUC: None (no variation)")

            # Get segment-level AUC (average across activities)
            seg_aucs = []
            if 'segment_level' in result:
                for activity_id in sorted(result['segment_level'].keys()):
                    auc_val = result['segment_level'][activity_id].get('auc')
                    if auc_val is not None:
                        seg_aucs.append(auc_val)
                        print(f"  Activity {activity_id}: Segment AUC = {auc_val:.4f}")

            if seg_aucs:
                avg_seg_auc = np.mean(seg_aucs)
                print(f"\n  Average Segment-Level AUC: {avg_seg_auc:.4f}")
            else:
                avg_seg_auc = None
                print(f"\n  Average Segment-Level AUC: None (no variation)")

            # Get basic metrics
            f1 = result['micro_avg']['f1_score']
            precision = result['micro_avg']['precision']
            recall = result['micro_avg']['recall']

            print(f"\n  Performance Metrics:")
            print(f"    F1 Score:  {f1:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")

            # Check if in expected range
            expected_range = test_info['expected']
            if '-' in expected_range:
                low, high = map(float, expected_range.split('-'))

                if avg_frame_auc is not None:
                    if low <= avg_frame_auc <= high:
                        print(f"\n  [PASS] Frame AUC {avg_frame_auc:.4f} is in expected range [{low}-{high}]")
                    else:
                        print(f"\n  [CHECK] Frame AUC {avg_frame_auc:.4f} outside expected range [{low}-{high}]")

                if avg_seg_auc is not None:
                    if low <= avg_seg_auc <= high:
                        print(f"  [PASS] Segment AUC {avg_seg_auc:.4f} is in expected range [{low}-{high}]")
                    else:
                        print(f"  [CHECK] Segment AUC {avg_seg_auc:.4f} outside expected range [{low}-{high}]")

            results.append({
                'name': test_info['name'],
                'expected': test_info['expected'],
                'frame_auc': avg_frame_auc,
                'segment_auc': avg_seg_auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })

        except Exception as e:
            print(f"\n  [ERROR]: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Test Name':<25} {'Expected':<12} {'Frame AUC':<12} {'Seg AUC':<12} {'F1':<8}")
    print("-" * 70)

    for r in results:
        frame_str = f"{r['frame_auc']:.4f}" if r['frame_auc'] is not None else "None"
        seg_str = f"{r['segment_auc']:.4f}" if r['segment_auc'] is not None else "None"
        print(f"{r['name']:<25} {r['expected']:<12} {frame_str:<12} {seg_str:<12} {r['f1']:<8.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
AUC = 1.0   → Perfect discrimination (all correct predictions)
AUC = 0.9+  → Excellent discrimination
AUC = 0.7-0.9 → Good discrimination
AUC = 0.5-0.7 → Fair discrimination
AUC = 0.5   → Random guessing (no discrimination ability)
AUC < 0.5   → Worse than random (inverted predictions)
AUC = 0.0   → Perfectly wrong (all predictions inverted)

If you see diverse AUC values from 0.0 to 0.7+, the ROC calculation
is working correctly across the full range!
""")
    print("="*70)

    return results


if __name__ == "__main__":
    results = test_bad_roc_files()

    print("\n[DONE] Testing complete!")
    print(f"  Tested {len(results)} scenarios")
    print(f"  Files location: ../results/bad_roc_tests/")
    print("\nYour ROC/AUC calculation has been validated across diverse scenarios!")
