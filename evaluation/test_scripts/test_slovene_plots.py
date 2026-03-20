#!/usr/bin/env python3
"""
Test script to generate plots with Slovene translations
"""

import sys
sys.path.append('./evaluation')

from segment_evaluator import SegmentEvaluator
from visualize_segments_timeline import visualize_segments_timeline
import matplotlib.pyplot as plt

def test_plots():
    """Generate test plots with Slovene translations"""

    # Use P002.txt as a test file (using it as both GT and prediction for demo)
    gt_file = './data_filtered_normal/P002.txt'
    pred_file = './data_filtered_normal/P002.txt'  # Same file for testing

    print("Initializing evaluator...")
    evaluator = SegmentEvaluator(
        timeline_resolution_seconds=1.0,
        start_time_threshold_seconds=30,
        overlap_threshold=0.5,
        confidence_threshold=0.0,
        max_gap_seconds=60.0
    )

    # Filter to activities 1-8 (excluding activity 6 as in the original code)
    activity_filter = [str(i) for i in range(1, 9) if i != 6]

    print("\nEvaluating with dual ROC curves...")
    results = evaluator.evaluate_with_dual_roc(
        pred_file,
        gt_file,
        aggregation='average',
        activity_filter=activity_filter
    )

    # Generate ROC curves
    print("\nGenerating ROC curves with Slovene labels...")
    evaluator.plot_dual_roc_curves(
        results,
        save_path='roc_curves_slovene.png',
        activity_filter=activity_filter
    )
    print("[OK] ROC curves saved to: roc_curves_slovene.png")

    # Extract segments for timeline visualization
    print("\nExtracting segments for timeline visualization...")
    pred_df = evaluator.parse_log_file(pred_file)
    gt_df = evaluator.parse_log_file(gt_file)

    pred_segments = evaluator.extract_segments(pred_df)
    gt_segments = evaluator.extract_segments(gt_df)

    print(f"  Ground truth segments: {len(gt_segments)}")
    print(f"  Predicted segments: {len(pred_segments)}")

    # Generate timeline visualization
    print("\nGenerating timeline visualization with Slovene labels...")
    fig, ax = visualize_segments_timeline(
        predicted_segments=pred_segments,
        ground_truth_segments=gt_segments,
        segment_ids=activity_filter,
        title=""  # No title as requested
    )

    plt.savefig('timeline_slovene.png', dpi=150, bbox_inches='tight')
    print("[OK] Timeline saved to: timeline_slovene.png")

    print("\n" + "="*70)
    print("SUCCESS! Plots generated with Slovene translations:")
    print("  - roc_curves_slovene.png")
    print("  - timeline_slovene.png")
    print("="*70)

    # Display summary metrics
    print("\nSummary Metrics:")
    if 'macro_avg' in results:
        if 'frame_auc' in results['macro_avg'] and results['macro_avg']['frame_auc'] is not None:
            print(f"  Frame-Level AUC:   {results['macro_avg']['frame_auc']:.4f}".replace('.', ','))
        if 'segment_auc' in results['macro_avg'] and results['macro_avg']['segment_auc'] is not None:
            print(f"  Segment-Level AUC: {results['macro_avg']['segment_auc']:.4f}".replace('.', ','))

if __name__ == '__main__':
    test_plots()
