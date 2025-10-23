import os
import random
import subprocess
import shutil
import sys
from datetime import datetime
import json
import numpy as np
from collections import defaultdict

# Add evaluation folder to path
sys.path.append('./evaluation')

from segment_evaluator import SegmentEvaluator
from visualize_segments_timeline import visualize_segments_timeline
import matplotlib.pyplot as plt

def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)

def filter_results_by_activity(results, activity_filter):
    """Filter evaluation results to only include specified activities AND remove ROC arrays"""
    filtered_results = {}
    
    # Copy and filter metadata
    if 'metadata' in results:
        filtered_results['metadata'] = results['metadata'].copy()
        # Filter activity_classes in metadata
        if 'activity_classes' in filtered_results['metadata']:
            filtered_results['metadata']['activity_classes'] = [
                ac for ac in filtered_results['metadata']['activity_classes']
                if ac in activity_filter
            ]
    
    # Copy top-level metrics (these are already aggregated across filtered activities)
    for key in ['macro_avg', 'micro_avg', 'weighted_avg']:
        if key in results:
            filtered_results[key] = results[key]
    
    # Filter per_class metrics
    if 'per_class' in results:
        filtered_results['per_class'] = {
            activity_id: metrics 
            for activity_id, metrics in results['per_class'].items()
            if activity_id in activity_filter
        }
    
    # Filter frame_level ROC - KEEP ONLY AUC, DROP FPR/TPR/THRESHOLDS
    if 'frame_level' in results:
        filtered_results['frame_level'] = {}
        for activity_id, roc_data in results['frame_level'].items():
            if activity_id in activity_filter:
                # Only keep AUC value, drop the arrays
                filtered_results['frame_level'][activity_id] = {
                    'auc': roc_data.get('auc')
                }
    
    # Filter segment_level ROC - KEEP ONLY AUC, DROP FPR/TPR/THRESHOLDS
    if 'segment_level' in results:
        filtered_results['segment_level'] = {}
        for activity_id, roc_data in results['segment_level'].items():
            if activity_id in activity_filter:
                # Only keep AUC value, drop the arrays
                filtered_results['segment_level'][activity_id] = {
                    'auc': roc_data.get('auc')
                }
    
    # Filter per_file_results if they exist
    if 'per_file_results' in results:
        filtered_per_file = []
        for file_result in results['per_file_results']:
            filtered_file_result = {
                'pred_file': file_result['pred_file'],
                'gt_file': file_result['gt_file'],
                'num_pred_segments': file_result['num_pred_segments'],
                'num_gt_segments': file_result['num_gt_segments']
            }
            
            # Filter frame_level in per-file results - AUC ONLY
            if 'frame_level' in file_result:
                filtered_file_result['frame_level'] = {
                    activity_id: {'auc': roc_data.get('auc')}
                    for activity_id, roc_data in file_result['frame_level'].items()
                    if activity_id in activity_filter
                }
            
            # Filter segment_level in per-file results - AUC ONLY
            if 'segment_level' in file_result:
                filtered_file_result['segment_level'] = {
                    activity_id: {'auc': roc_data.get('auc')}
                    for activity_id, roc_data in file_result['segment_level'].items()
                    if activity_id in activity_filter
                }
            
            # Filter metrics in per-file results
            if 'metrics' in file_result:
                filtered_file_result['metrics'] = {}
                for key in ['macro_avg', 'micro_avg', 'weighted_avg']:
                    if key in file_result['metrics']:
                        filtered_file_result['metrics'][key] = file_result['metrics'][key]
                
                if 'per_class' in file_result['metrics']:
                    filtered_file_result['metrics']['per_class'] = {
                        activity_id: metrics
                        for activity_id, metrics in file_result['metrics']['per_class'].items()
                        if activity_id in activity_filter
                    }
            
            filtered_per_file.append(filtered_file_result)
        
        filtered_results['per_file_results'] = filtered_per_file
    
    return filtered_results

def clear_annotations(input_file, output_file):
    """Remove the last column (activity label) from each line and strip BOM"""
    with open(input_file, 'r', encoding='utf-8-sig') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            words = line.split()
            if len(words) >= 5:
                # Keep only first 4 fields: date, time, sensor, status
                new_line = ' '.join(words[:4]) + '\n'
                f_out.write(new_line)
            elif len(words) >= 4:
                # Already unannotated
                new_line = ' '.join(words[:4]) + '\n'
                f_out.write(new_line)

def check_existing_model(model_name='quick_test_model'):
    """Check if a trained model already exists"""
    model_path = os.path.join('./model', f'{model_name}.pkl.gz')
    return os.path.exists(model_path)

def count_activity_instances(file_path, evaluator, activity_ids=['1','2','3','4','5','6','7','8']):
    """Count how many instances of each activity are in a file"""
    try:
        df = evaluator.parse_log_file(file_path)
        segments = evaluator.extract_segments(df)
        
        counts = {aid: 0 for aid in activity_ids}
        for seg in segments:
            if seg['segment_id'] in activity_ids:
                counts[seg['segment_id']] += 1
        
        return counts
    except Exception as e:
        print(f"Warning: Could not count activities in {file_path}: {e}")
        return {aid: 0 for aid in activity_ids}

def create_stratified_folds(files, evaluator, n_folds=5, activity_ids=['1','2','3','4','5','6','7','8'], random_seed=42):
    """
    Create stratified k-folds based on activity presence
    
    Strategy: Ensure each fold has roughly equal representation of files containing each activity
    """
    print(f"\n{'='*70}")
    print(f"CREATING {n_folds}-FOLD STRATIFIED CROSS-VALIDATION SPLITS")
    print(f"{'='*70}")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Count activities in each file
    print("\nCounting activity instances in all files...")
    file_activity_counts = {}
    for file_path in files:
        counts = count_activity_instances(file_path, evaluator, activity_ids)
        file_activity_counts[file_path] = counts
    
    # Create activity presence matrix (file x activity binary matrix)
    # 1 if activity present in file, 0 otherwise
    file_list = list(file_activity_counts.keys())
    n_files = len(file_list)
    
    # Calculate which activities each file contains
    file_activity_presence = []
    for file_path in file_list:
        presence = tuple(1 if file_activity_counts[file_path][aid] > 0 else 0 
                        for aid in activity_ids)
        file_activity_presence.append(presence)
    
    # Group files by their activity presence pattern
    pattern_groups = defaultdict(list)
    for idx, pattern in enumerate(file_activity_presence):
        pattern_groups[pattern].append(idx)
    
    print(f"\nFound {len(pattern_groups)} unique activity presence patterns")
    print(f"Pattern examples:")
    for i, (pattern, indices) in enumerate(list(pattern_groups.items())[:5]):
        pattern_str = ''.join(str(p) for p in pattern)
        print(f"  Pattern {pattern_str}: {len(indices)} files")
    
    # Initialize folds
    folds = [[] for _ in range(n_folds)]
    
    # Distribute each pattern group across folds
    for pattern, indices in pattern_groups.items():
        # Shuffle this group
        random.shuffle(indices)
        
        # Distribute across folds in round-robin fashion
        for i, idx in enumerate(indices):
            fold_num = i % n_folds
            folds[fold_num].append(file_list[idx])
    
    # Print fold statistics
    print(f"\n{'='*70}")
    print("FOLD STATISTICS")
    print(f"{'='*70}")
    
    for fold_idx in range(n_folds):
        print(f"\n--- Fold {fold_idx + 1} ---")
        print(f"  Number of files: {len(folds[fold_idx])}")
        
        # Count total activity instances in this fold
        fold_activity_counts = {aid: 0 for aid in activity_ids}
        for file_path in folds[fold_idx]:
            for aid in activity_ids:
                fold_activity_counts[aid] += file_activity_counts[file_path][aid]
        
        # Print per-activity counts
        for aid in activity_ids:
            print(f"  Activity {aid}: {fold_activity_counts[aid]} instances")
    
    # Print overall distribution
    print(f"\n{'='*70}")
    print("OVERALL ACTIVITY DISTRIBUTION CHECK")
    print(f"{'='*70}")
    
    total_counts = {aid: 0 for aid in activity_ids}
    for file_path in file_list:
        for aid in activity_ids:
            total_counts[aid] += file_activity_counts[file_path][aid]
    
    print(f"\n{'Activity':<12} {'Total':>8} {'Per Fold Avg':>15} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-"*70)
    
    for aid in activity_ids:
        fold_counts = []
        for fold_idx in range(n_folds):
            count = sum(file_activity_counts[f][aid] for f in folds[fold_idx])
            fold_counts.append(count)
        
        total = total_counts[aid]
        avg = np.mean(fold_counts)
        min_count = np.min(fold_counts)
        max_count = np.max(fold_counts)
        std = np.std(fold_counts)
        
        print(f"{aid:<12} {total:>8} {avg:>15.1f} {min_count:>8} {max_count:>8} {std:>8.2f}")
    
    print(f"{'='*70}\n")
    
    return folds, file_activity_counts

def add_realistic_confidence_scores(gt_file, output_file, 
                                              error_rate=0.10):
    """
    Add uniformly distributed confidence scores (0.0-1.0) with errors for ROC validation
    
    This creates smooth ROC curves with many points to properly test ROC calculation.
    
    Args:
        error_rate: Fraction of predictions to make incorrect (0.10 = 10% errors)
    """
    import random
    random.seed(42)
        
    # Possible activities for wrong predictions
    activities = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    lines_read = 0
    lines_written = 0
    confidence_values = []
    errors_introduced = 0
    
    try:
        with open(gt_file, 'r', encoding='utf-8-sig') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                lines_read += 1
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) >= 5:
                    # Decide if this should be an error
                    is_error = random.random() < error_rate
                    
                    if is_error:
                        # Change activity to wrong one
                        correct_activity = parts[4]
                        wrong_activities = [a for a in activities if a != correct_activity]
                        if wrong_activities:
                            parts[4] = random.choice(wrong_activities)
                            errors_introduced += 1
                        
                        # Errors get LOW uniform confidence (0.0-0.5)
                        confidence = random.uniform(0.0, 0.5)
                    else:
                        # Correct predictions get HIGH uniform confidence (0.5-1.0)
                        confidence = random.uniform(0.5, 1.0)
                    
                    confidence_values.append(confidence)
                    
                    # Output format: date time sensor sensor status activity confidence
                    new_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[2]} {parts[3]} {parts[4]} {confidence:.4f}\n"
                    f_out.write(new_line)
                    lines_written += 1                
                    
    except Exception as e:
        print(f"    ERROR in add_uniform_confidence_scores_with_noise: {e}")
        import traceback
        traceback.print_exc()

def run_fold_validation_only(fold_idx, train_files, test_files, output_dir, 
                              evaluator, activity_filter):
    """
    Run evaluation WITHOUT training - with synthetic confidence scores
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1} - VALIDATION MODE (WITH CONFIDENCE SCORES)")
    print(f"{'='*70}")
    print(f"Test files: {len(test_files)}")
    
    # Create temp directory for files with confidence scores
    temp_dir = './temp_validation'
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temp directory: {temp_dir}")
    
    predicted_files = []
    ground_truth_files = []
    
    for idx, test_file in enumerate(test_files):
        print(f"\n  Processing file {idx + 1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        # Create prediction file with varying confidence scores
        pred_file = os.path.join(temp_dir, f'pred_with_conf_{idx}.txt')
        add_realistic_confidence_scores(test_file, pred_file, error_rate=0.50)
        
        # Verify the file was created and has the right format
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                print(f"    First line parts: {len(parts)}")
                if len(parts) >= 6:
                    print(f"    Has confidence column: YES (value: {parts[-1]})")
                else:
                    print(f"    Has confidence column: NO")
        
        predicted_files.append(pred_file)
        ground_truth_files.append(test_file)
    
    print(f"\n[VALIDATION MODE] Created {len(predicted_files)} prediction files")
    print("Confidence scores: 0.85-1.0 (varying)")
    print("Expected: Precision ~= 1.0, Recall ~= 1.0, AUC ~= 0.95-1.0")
    
    # Evaluate
    print(f"\nEvaluating fold {fold_idx + 1}...")
    
    multi_results = evaluator.evaluate_multiple_files(predicted_files, ground_truth_files)
    
    # Check what we got back
    print("\nDEBUG: Evaluation completed")
    print(f"  Results keys: {multi_results.keys()}")
    
    if 'frame_level' in multi_results:
        print(f"  Frame-level activities: {list(multi_results['frame_level'].keys())}")
        for activity_id in activity_filter:
            if activity_id in multi_results['frame_level']:
                auc = multi_results['frame_level'][activity_id].get('auc')
                print(f"    Activity {activity_id} frame AUC: {auc}")
    
    if 'segment_level' in multi_results:
        print(f"  Segment-level activities: {list(multi_results['segment_level'].keys())}")
        for activity_id in activity_filter:
            if activity_id in multi_results['segment_level']:
                auc = multi_results['segment_level'][activity_id].get('auc')
                print(f"    Activity {activity_id} segment AUC: {auc}")
    
    # Generate ROC curves
    print(f"\nGenerating ROC curves for fold {fold_idx + 1}...")
    roc_plot_path = os.path.join(output_dir, f'roc_fold{fold_idx + 1}_validation.png')
    print(f"  Output path: {roc_plot_path}")
    
    try:
        evaluator.plot_dual_roc_curves(
            multi_results,
            save_path=roc_plot_path,
            activity_filter=activity_filter
        )
        
        # Verify file was created
        if os.path.exists(roc_plot_path):
            file_size = os.path.getsize(roc_plot_path)
            print(f"[OK] ROC curves saved: {roc_plot_path} ({file_size} bytes)")
        else:
            print(f"[ERROR] ROC file not created: {roc_plot_path}")
            
    except Exception as e:
        print(f"[ERROR] Failed to generate ROC curves: {e}")
        import traceback
        traceback.print_exc()
    
    # Don't cleanup temp directory yet so we can inspect files
    print(f"\nTemp files kept for inspection: {temp_dir}")
    # shutil.rmtree(temp_dir)
    
    return multi_results

def test_file_format(filepath):
    """Test if a file has the expected format"""
    print(f"\nTesting file format: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ERROR: File does not exist!")
        return False
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()[:10]  # First 10 lines
    
    print(f"  Total lines to check: {len(lines)}")
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        print(f"  Line {i+1}: {len(parts)} parts")
        if len(parts) >= 5:
            print(f"    Format: {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        else:
            print(f"    WARNING: Not enough parts! Content: {line.strip()}")
    
    return True


def run_cross_validation_validation_mode(files, output_dir, start_error_threshold, n_folds=5):
    """
    Run k-fold validation WITHOUT training - for validating the evaluation pipeline
    """
    evaluator = SegmentEvaluator(
        timeline_resolution_seconds=1.0,
        start_time_threshold_seconds=start_error_threshold
    )
    
    activity_filter = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    # Create stratified folds
    folds, file_activity_counts = create_stratified_folds(
        files, evaluator, n_folds=n_folds, activity_ids=activity_filter
    )
    
    # Run each fold
    all_fold_results = []
    
    for fold_idx in range(n_folds):
        # Get test files for this fold
        test_file_paths = folds[fold_idx]
        
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds} - VALIDATION MODE")
        print(f"{'='*70}")
        print(f"Files to validate: {len(test_file_paths)}")
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        print(f"Fold output directory: {fold_output_dir}")
        
        # Verify directory was created
        if not os.path.exists(fold_output_dir):
            print(f"[ERROR] Could not create fold directory: {fold_output_dir}")
            continue
        
        # Run validation (no training)
        fold_results = run_fold_validation_only(
            fold_idx,
            [],  # No training files needed
            test_file_paths,
            fold_output_dir,
            evaluator,
            activity_filter
        )
        
        if fold_results is None:
            print(f"[ERROR] Fold {fold_idx + 1} failed")
            continue
        
        all_fold_results.append(fold_results)
        
        # Print fold report
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1} VALIDATION RESULTS")
        print(f"{'='*70}")
        evaluator.print_report(fold_results, activity_filter=activity_filter)
        
        # Check if results are close to perfect
        micro_prec = fold_results['micro_avg']['precision']
        micro_rec = fold_results['micro_avg']['recall']
        micro_f1 = fold_results['micro_avg']['f1_score']
        
        print(f"\n{'='*70}")
        print("VALIDATION CHECK")
        print(f"{'='*70}")
        
        tolerance = 0.05  # Allow 5% deviation
        checks = []
        
        if micro_prec >= 0.80:
            print(f"[PASS] Precision: {micro_prec:.4f} (expected >= 0.80)")
            checks.append(True)
        else:
            print(f"[FAIL] Precision: {micro_prec:.4f} (expected >= 0.80)")
            checks.append(False)
        
        if micro_rec >= 0.75:
            print(f"[PASS] Recall: {micro_rec:.4f} (expected >= 0.75)")
            checks.append(True)
        else:
            print(f"[FAIL] Recall: {micro_rec:.4f} (expected >= 0.75)")
            checks.append(False)
        
        if micro_f1 >= 0.77:
            print(f"[PASS] F1 Score: {micro_f1:.4f} (expected >= 0.77)")
            checks.append(True)
        else:
            print(f"[FAIL] F1 Score: {micro_f1:.4f} (expected >= 0.77)")
            checks.append(False)
        
        # Check AUC values
        frame_aucs = []
        segment_aucs = []
        
        for activity_id in activity_filter:
            if 'frame_level' in fold_results and activity_id in fold_results['frame_level']:
                auc_val = fold_results['frame_level'][activity_id].get('auc')
                if auc_val is not None and auc_val > 0:
                    frame_aucs.append(auc_val)
            
            if 'segment_level' in fold_results and activity_id in fold_results['segment_level']:
                auc_val = fold_results['segment_level'][activity_id].get('auc')
                if auc_val is not None and auc_val > 0:
                    segment_aucs.append(auc_val)
        
        avg_frame_auc = np.mean(frame_aucs) if frame_aucs else None
        avg_segment_auc = np.mean(segment_aucs) if segment_aucs else None
        
        if avg_frame_auc and avg_frame_auc >= 0.80:
            print(f"[PASS] Frame AUC: {avg_frame_auc:.4f} (expected >= 0.80)")
            checks.append(True)
        elif avg_frame_auc:
            print(f"[WARN] Frame AUC: {avg_frame_auc:.4f} (expected >= 0.80)")
            checks.append(False)
        else:
            print(f"[FAIL] Frame AUC: Not calculated")
            checks.append(False)
        
        if avg_segment_auc and avg_segment_auc >= 0.85:
            print(f"[PASS] Segment AUC: {avg_segment_auc:.4f} (expected >= 0.85)")
            checks.append(True)
        elif avg_segment_auc:
            print(f"[WARN] Segment AUC: {avg_segment_auc:.4f} (expected >= 0.85)")
            checks.append(False)
        else:
            print(f"[FAIL] Segment AUC: Not calculated")
            checks.append(False)
        
        if all(checks):
            print(f"\n[PASS] FOLD {fold_idx + 1}: VALIDATION PASSED")
        else:
            print(f"\n[FAIL] FOLD {fold_idx + 1}: VALIDATION FAILED - Check evaluation logic!")
    
    # Print summary
    if len(all_fold_results) > 0:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY ACROSS ALL FOLDS")
        print(f"{'='*70}")
        
        all_precision = [r['micro_avg']['precision'] for r in all_fold_results]
        all_recall = [r['micro_avg']['recall'] for r in all_fold_results]
        all_f1 = [r['micro_avg']['f1_score'] for r in all_fold_results]
        
        print(f"\nMicro-averaged metrics across {len(all_fold_results)} folds:")
        print(f"  Precision: {np.mean(all_precision):.4f} +/- {np.std(all_precision):.4f}")
        print(f"  Recall:    {np.mean(all_recall):.4f} +/- {np.std(all_recall):.4f}")
        print(f"  F1 Score:  {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")
        
        # Collect AUC values
        all_frame_aucs = []
        all_segment_aucs = []
        
        for fold_result in all_fold_results:
            for activity_id in activity_filter:
                if 'frame_level' in fold_result and activity_id in fold_result['frame_level']:
                    auc_val = fold_result['frame_level'][activity_id].get('auc')
                    if auc_val is not None and auc_val > 0:
                        all_frame_aucs.append(auc_val)
                
                if 'segment_level' in fold_result and activity_id in fold_result['segment_level']:
                    auc_val = fold_result['segment_level'][activity_id].get('auc')
                    if auc_val is not None and auc_val > 0:
                        all_segment_aucs.append(auc_val)
        
        if all_frame_aucs:
            print(f"  Frame AUC: {np.mean(all_frame_aucs):.4f} +/- {np.std(all_frame_aucs):.4f}")
        if all_segment_aucs:
            print(f"  Segment AUC: {np.mean(all_segment_aucs):.4f} +/- {np.std(all_segment_aucs):.4f}")
        
        if (abs(np.mean(all_precision) - 1.0) < 0.05 and 
            abs(np.mean(all_recall) - 1.0) < 0.05 and
            abs(np.mean(all_f1) - 1.0) < 0.05):
            print(f"\n[PASS] OVERALL VALIDATION: PASSED")
            print("Your evaluation pipeline is working correctly!")
        else:
            print(f"\n[FAIL] OVERALL VALIDATION: FAILED")
            print("There may be issues with your evaluation logic.")
            print("\nPossible causes:")
            print("  - Time alignment issues")
            print("  - Activity ID mismatch")
            print("  - Segment parsing errors")
            print("  - Incorrect TP/FP/FN calculations")
    else:
        print("\n[ERROR] No folds completed successfully")
    
        return all_fold_results
    
    # Print summary
    if len(all_fold_results) > 0:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY ACROSS ALL FOLDS")
        print(f"{'='*70}")
        
        all_precision = [r['micro_avg']['precision'] for r in all_fold_results]
        all_recall = [r['micro_avg']['recall'] for r in all_fold_results]
        all_f1 = [r['micro_avg']['f1_score'] for r in all_fold_results]
        
        print(f"\nMicro-averaged metrics across {len(all_fold_results)} folds:")
        print(f"  Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
        print(f"  Recall:    {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
        print(f"  F1 Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
        
        if (abs(np.mean(all_precision) - 1.0) < 0.05 and 
            abs(np.mean(all_recall) - 1.0) < 0.05 and
            abs(np.mean(all_f1) - 1.0) < 0.05):
            print(f"\n[OK] OVERALL VALIDATION: PASSED")
            print("Your evaluation pipeline is working correctly!")
        else:
            print(f"\n[NOK] OVERALL VALIDATION: FAILED")
            print("There may be issues with your evaluation logic.")
            print("\nPossible causes:")
            print("  - Time alignment issues")
            print("  - Activity ID mismatch")
            print("  - Segment parsing errors")
            print("  - Incorrect TP/FP/FN calculations")
    
    return all_fold_results

def run_fold(fold_idx, train_files, test_files, output_dir, model_name,
             start_error_threshold, evaluator, visualize=False):
    """
    Run training and evaluation for a single fold
    
    Returns:
        multi_results: Evaluation results for this fold
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1} - TRAINING AND EVALUATION")
    print(f"{'='*70}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    
    # Create temp directory
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Train model for this fold
    train_combined = os.path.join(temp_dir, f'fold_{fold_idx}_train.txt')
    print(f"\nCombining {len(train_files)} training files...")
    with open(train_combined, 'w') as outfile:
        for fname in train_files:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())
    
    print("\nTraining model...")
    cmd = [
        'python', 
        './AL/al.py', 
        '--mode', 'TRAIN', 
        '--data', train_combined, 
        '--model', model_name,
        '--ignoreother'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    
    if result.returncode != 0:
        print("ERROR during training:")
        print(result.stderr)
        shutil.rmtree(temp_dir)
        return None
    
    print("[OK] Model trained successfully")
    os.remove(train_combined)
    
    # Process test files
    predicted_files = []
    ground_truth_files = []
    
    for idx, test_file in enumerate(test_files):
        # Clear annotations
        unannotated_file = os.path.join(temp_dir, f'unannotated_{idx}.txt')
        clear_annotations(test_file, unannotated_file)
        
        if not os.path.exists(unannotated_file):
            continue
        
        # Annotate
        annotated_output = './data.al'
        if os.path.exists(annotated_output):
            os.remove(annotated_output)
        
        cmd = [
            'python', 
            './AL/al.py', 
            '--mode', 'ANNOTATE', 
            '--data', unannotated_file, 
            '--model', model_name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0 or not os.path.exists(annotated_output):
            print(f"Warning: Failed to annotate {os.path.basename(test_file)}")
            continue
        
        # Save prediction
        saved_annotated = os.path.join(
            output_dir, 
            f'fold{fold_idx}_pred_{idx}_{os.path.basename(test_file)}'
        )
        shutil.copy(annotated_output, saved_annotated)
        
        predicted_files.append(saved_annotated)
        ground_truth_files.append(test_file)
        
        # Cleanup
        if os.path.exists(annotated_output):
            os.remove(annotated_output)
        if os.path.exists(unannotated_file):
            os.remove(unannotated_file)
    
    print(f"\n[OK] Generated predictions for {len(predicted_files)} files")
    
    if len(predicted_files) == 0:
        print("ERROR: No predictions generated")
        shutil.rmtree(temp_dir)
        return None
    
    # Evaluate
    print(f"\nEvaluating fold {fold_idx + 1}...")
    multi_results = evaluator.evaluate_multiple_files(predicted_files, ground_truth_files)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return multi_results

def print_fold_comparison(all_fold_results, activity_filter):
    """Print comparison of metrics across all folds"""
    print(f"\n{'='*70}")
    print("CROSS-FOLD COMPARISON")
    print(f"{'='*70}")
    
    n_folds = len(all_fold_results)
    
    # Aggregate metrics
    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)
    
    # Micro-average metrics
    micro_precision = [r['micro_avg']['precision'] for r in all_fold_results]
    micro_recall = [r['micro_avg']['recall'] for r in all_fold_results]
    micro_f1 = [r['micro_avg']['f1_score'] for r in all_fold_results]
    
    print(f"{'Micro Precision':<20} {np.mean(micro_precision):>10.4f} {np.std(micro_precision):>10.4f} {np.min(micro_precision):>10.4f} {np.max(micro_precision):>10.4f}")
    print(f"{'Micro Recall':<20} {np.mean(micro_recall):>10.4f} {np.std(micro_recall):>10.4f} {np.min(micro_recall):>10.4f} {np.max(micro_recall):>10.4f}")
    print(f"{'Micro F1':<20} {np.mean(micro_f1):>10.4f} {np.std(micro_f1):>10.4f} {np.min(micro_f1):>10.4f} {np.max(micro_f1):>10.4f}")
    
    # Per-activity metrics
    print(f"\n{'='*70}")
    print("PER-ACTIVITY METRICS ACROSS FOLDS")
    print(f"{'='*70}")
    
    for activity_id in activity_filter:
        print(f"\n--- Activity {activity_id} ---")
        
        # Collect metrics for this activity across folds
        precisions = []
        recalls = []
        f1s = []
        frame_aucs = []
        seg_aucs = []
        
        for fold_result in all_fold_results:
            if activity_id in fold_result['per_class']:
                metrics = fold_result['per_class'][activity_id]
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1s.append(metrics['f1_score'])
            
            if 'frame_level' in fold_result and activity_id in fold_result['frame_level']:
                auc_val = fold_result['frame_level'][activity_id].get('auc')
                if auc_val is not None:
                    frame_aucs.append(auc_val)
            
            if 'segment_level' in fold_result and activity_id in fold_result['segment_level']:
                auc_val = fold_result['segment_level'][activity_id].get('auc')
                if auc_val is not None:
                    seg_aucs.append(auc_val)
        
        if precisions:
            print(f"  Precision:     {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
            print(f"  Recall:        {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
            print(f"  F1 Score:      {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        
        if frame_aucs:
            print(f"  Frame AUC:     {np.mean(frame_aucs):.4f} ± {np.std(frame_aucs):.4f}")
        
        if seg_aucs:
            print(f"  Segment AUC:   {np.mean(seg_aucs):.4f} ± {np.std(seg_aucs):.4f}")
    
    print(f"{'='*70}\n")
    
def create_aggregated_summary(all_fold_results, activity_filter):
    """Create aggregated summary across all folds"""
    
    summary = {
        'num_folds': len(all_fold_results),
        'per_class': {},
        'aggregate': {}
    }
    
    # Aggregate per-class metrics
    for activity_id in activity_filter:
        class_metrics = {
            'TP_duration': [],
            'FP_duration': [],
            'FN_duration': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'iou': [],
            'gt_total_duration': [],
            'pred_total_duration': [],
            'frame_auc': [],
            'segment_auc': []
        }
        
        for fold_result in all_fold_results:
            if activity_id in fold_result.get('per_class', {}):
                metrics = fold_result['per_class'][activity_id]
                class_metrics['TP_duration'].append(metrics['TP_duration'])
                class_metrics['FP_duration'].append(metrics['FP_duration'])
                class_metrics['FN_duration'].append(metrics['FN_duration'])
                class_metrics['precision'].append(metrics['precision'])
                class_metrics['recall'].append(metrics['recall'])
                class_metrics['f1_score'].append(metrics['f1_score'])
                class_metrics['iou'].append(metrics['iou'])
                class_metrics['gt_total_duration'].append(metrics['gt_total_duration'])
                class_metrics['pred_total_duration'].append(metrics['pred_total_duration'])
            
            if 'frame_level' in fold_result and activity_id in fold_result['frame_level']:
                auc_val = fold_result['frame_level'][activity_id].get('auc')
                if auc_val is not None:
                    class_metrics['frame_auc'].append(auc_val)
            
            if 'segment_level' in fold_result and activity_id in fold_result['segment_level']:
                auc_val = fold_result['segment_level'][activity_id].get('auc')
                if auc_val is not None:
                    class_metrics['segment_auc'].append(auc_val)
        
        # Calculate mean ± std for each metric
        summary['per_class'][activity_id] = {}
        for metric, values in class_metrics.items():
            if len(values) > 0:
                summary['per_class'][activity_id][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                summary['per_class'][activity_id][metric] = None
    
    # Aggregate overall metrics
    for avg_type in ['micro_avg', 'macro_avg', 'weighted_avg']:
        agg_metrics = {
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        if avg_type == 'micro_avg':
            agg_metrics['total_TP_duration'] = []
            agg_metrics['total_FP_duration'] = []
            agg_metrics['total_FN_duration'] = []
        
        for fold_result in all_fold_results:
            if avg_type in fold_result:
                agg_metrics['precision'].append(fold_result[avg_type].get('precision', 0))
                agg_metrics['recall'].append(fold_result[avg_type].get('recall', 0))
                agg_metrics['f1_score'].append(fold_result[avg_type].get('f1_score', 0))
                
                if avg_type == 'micro_avg':
                    agg_metrics['total_TP_duration'].append(
                        fold_result[avg_type].get('total_TP_duration', 0))
                    agg_metrics['total_FP_duration'].append(
                        fold_result[avg_type].get('total_FP_duration', 0))
                    agg_metrics['total_FN_duration'].append(
                        fold_result[avg_type].get('total_FN_duration', 0))
        
        summary['aggregate'][avg_type] = {}
        for metric, values in agg_metrics.items():
            if len(values) > 0:
                summary['aggregate'][avg_type][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
    
    return summary

def run_cross_validation(files, unfiltered_dir, output_dir, model_name_base,
                        start_error_threshold, n_folds=5, visualize=False):
    """
    Run k-fold cross-validation
    """
    evaluator = SegmentEvaluator(
        timeline_resolution_seconds=1.0,
        start_time_threshold_seconds=start_error_threshold
    )
    
    activity_filter = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    # Create stratified folds
    folds, file_activity_counts = create_stratified_folds(
        files, evaluator, n_folds=n_folds, activity_ids=activity_filter
    )
    
    total_in_folds = sum(len(fold) for fold in folds)
    
    if total_in_folds != len(files):
        print(f"\nERROR: Lost {len(files) - total_in_folds} files during fold creation!")
        
    all_fold_files = [f for fold in folds for f in fold]
    unique_files = set(all_fold_files)

    if len(all_fold_files) != len(unique_files):
        print("[WARNING] Some files appear in multiple folds!")
        seen, duplicates = set(), []
        for f in all_fold_files:
            if f in seen:
                duplicates.append(f)
            else:
                seen.add(f)
        print(f"Duplicate files ({len(duplicates)}):")
        for dup in duplicates[:10]:
            print(f"  - {dup}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
    else:
        print("[OK] No file overlaps found between folds")
    
    # Run each fold
    all_fold_results = []
    
    for fold_idx in range(n_folds):
        # Create train/test split from FILTERED files
        test_file_paths_filtered = folds[fold_idx]
        train_file_paths_filtered = []
        for i in range(n_folds):
            if i != fold_idx:
                train_file_paths_filtered.extend(folds[i])
        
        # DEBUG: Print filtered split
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds} - DATA SPLIT (FILTERED)")
        print(f"{'='*70}")
        print(f"Test (filtered): {len(test_file_paths_filtered)} files")
        print(f"Train (filtered): {len(train_file_paths_filtered)} files")
        print(f"Total (filtered): {len(test_file_paths_filtered) + len(train_file_paths_filtered)} files")
        
        # Get matching UNFILTERED test files
        test_file_paths_unfiltered = []
        missing_files = []
        for filtered_path in test_file_paths_filtered:
            basename = os.path.basename(filtered_path)
            unfiltered_path = os.path.join(unfiltered_dir, basename)
            if os.path.exists(unfiltered_path):
                test_file_paths_unfiltered.append(unfiltered_path)
            else:
                missing_files.append(basename)
        
        # Print unfiltered split
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds} - DATA SPLIT (UNFILTERED)")
        print(f"{'='*70}")
        print(f"Training: {len(train_file_paths_filtered)} files (filtered)")
        print(f"Testing: {len(test_file_paths_unfiltered)} files (unfiltered)")
        print(f"Expected test files: {len(test_file_paths_filtered)}")
        print(f"Actual test files found: {len(test_file_paths_unfiltered)}")
        print(f"Missing files: {len(missing_files)}")
        
        if missing_files:
            print(f"\nWARNING: Missing unfiltered files:")
            for f in missing_files[:10]:
                print(f"  - {f}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
        
        # Calculate expected percentages
        total_filtered = len(test_file_paths_filtered) + len(train_file_paths_filtered)
        expected_train_pct = len(train_file_paths_filtered) / total_filtered * 100
        expected_test_pct = len(test_file_paths_filtered) / total_filtered * 100
        
        print(f"\nExpected split: {expected_train_pct:.1f}% train / {expected_test_pct:.1f}% test")
        
        # Print activity distribution for this fold
        print(f"\n{'Activity':<12} {'Train Count':>12} {'Test Count':>12}")
        print("-"*40)
        
        for aid in activity_filter:
            train_count = sum(file_activity_counts[f][aid] for f in train_file_paths_filtered)
            test_count = 0
            for unfiltered_path in test_file_paths_unfiltered:
                basename = os.path.basename(unfiltered_path)
                # Find the corresponding filtered file in test set
                for filtered_path in test_file_paths_filtered:
                    if os.path.basename(filtered_path) == basename:
                        test_count += file_activity_counts[filtered_path][aid]
                        break
            
            print(f"{aid:<12} {train_count:>12} {test_count:>12}")
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Run fold
        model_name = f"{model_name_base}_fold{fold_idx}"
        fold_results = run_fold(
            fold_idx, 
            train_file_paths_filtered,
            test_file_paths_unfiltered,
            fold_output_dir,
            model_name,
            start_error_threshold,
            evaluator,
            visualize=visualize
        )
        
        if fold_results is None:
            print(f"ERROR: Fold {fold_idx + 1} failed")
            continue
        
        all_fold_results.append(fold_results)
        
        # Print fold report
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1} RESULTS")
        print(f"{'='*70}")
        evaluator.print_report(fold_results, activity_filter=activity_filter)
        
        # Generate ROC curves for this fold
        combined_roc_file = os.path.join(fold_output_dir, f'roc_fold{fold_idx + 1}_cumulative.png')
        try:
            evaluator.plot_dual_roc_curves(
                fold_results,
                save_path=combined_roc_file,
                activity_filter=activity_filter
            )
            print(f"\n[OK] Fold {fold_idx + 1} ROC curves saved")
        except Exception as e:
            print(f"✗ Could not generate ROC for fold {fold_idx + 1}: {e}")
    
    # Print cross-fold comparison
    if len(all_fold_results) > 0:
        print_fold_comparison(all_fold_results, activity_filter)
        
        print(f"\nGenerating aggregated ROC curves across all {n_folds} folds...")
        evaluator.generate_aggregated_roc_curves(
            all_fold_results,
            output_dir,
            activity_filter
        )
        
        # Filter results and strip ROC arrays
        print("\nFiltering results to activities 1-8 and removing ROC arrays for JSON export...")
        filtered_fold_results = []
        for fold_result in all_fold_results:
            filtered_result = filter_results_by_activity(fold_result, activity_filter)
            filtered_fold_results.append(filtered_result)

        # Save per-fold summaries (CLEAN - no ROC arrays)
        for fold_idx, filtered_result in enumerate(filtered_fold_results):
            fold_json_path = os.path.join(output_dir, f'fold_{fold_idx + 1}', 'fold_summary.json')
            with open(fold_json_path, 'w') as f:
                json.dump({
                    'fold': fold_idx + 1,
                    'start_time_threshold': start_error_threshold,
                    'activity_filter': activity_filter,
                    'results': convert_to_json_serializable(filtered_result)
                }, f, indent=2)

        # Create and save aggregated summary
        aggregated_summary = create_aggregated_summary(filtered_fold_results, activity_filter)
        agg_json_path = os.path.join(output_dir, 'cross_validation_summary.json')
        with open(agg_json_path, 'w') as f:
            json.dump({
                'metadata': {
                    'n_folds': n_folds,
                    'start_time_threshold': start_error_threshold,
                    'activity_filter': activity_filter,
                    'timestamp': datetime.now().isoformat()
                },
                'aggregated': convert_to_json_serializable(aggregated_summary)
            }, f, indent=2)

        print(f"[OK] Cross-validation results saved:")
        print(f"  - Aggregated summary: {agg_json_path}")
        print(f"  - Per-fold summaries: ./fold_N/fold_summary.json")
    
    return all_fold_results

def main():
    # Configuration
    random.seed(42)
    
    filtered_data_dir = './data_filtered'
    unfiltered_data_dir = './data_spaces'
    model_name_base = 'cv_model'
    start_error_threshold = 100
    n_folds = 5
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filtered_dir_name = os.path.basename(os.path.normpath(filtered_data_dir))
    output_dir_name = f'run_{timestamp}_{n_folds}fold_{filtered_dir_name}_thresh{int(start_error_threshold)}s'
    output_dir = os.path.join('./output', output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all filtered files
    all_filtered_files = [
        os.path.join(filtered_data_dir, f) 
        for f in os.listdir(filtered_data_dir) 
        if f.endswith('.txt')
    ]
    
    all_filtered_files = all_filtered_files[:50]
    
    
    # CRITICAL: Only keep files that have matches in BOTH directories
    filtered_files = []
    missing_unfiltered = []
    
    for filtered_file in all_filtered_files:
        basename = os.path.basename(filtered_file)
        unfiltered_file = os.path.join(unfiltered_data_dir, basename)
        if os.path.exists(unfiltered_file):
            filtered_files.append(filtered_file)
        else:
            missing_unfiltered.append(basename)
         
    if len(filtered_files) > 0:
        print("\nTesting first file format...")
        test_file_format(filtered_files[0])
    
    if missing_unfiltered:
        print(f"\nWARNING: {len(missing_unfiltered)} files will be EXCLUDED from cross-validation")
        print(f"because they don't have matching unfiltered versions!")
        print(f"\nFirst 10 missing unfiltered files:")
        for f in missing_unfiltered[:10]:
            print(f"  - {f}")
        
        if len(missing_unfiltered) > 10:
            print(f"  ... and {len(missing_unfiltered) - 10} more")
    
    if len(filtered_files) == 0:
        print(f"\nERROR: No matching file pairs found!")
        print(f"Filtered dir: {filtered_data_dir}")
        print(f"Unfiltered dir: {unfiltered_data_dir}")
        return
    
    print(f"\n[OK] Using {len(filtered_files)} matched file pairs for cross-validation")
    print(f"{'='*70}\n")
    
    # Ask about visualization (commented out for batch mode)
    visualize = input("Enable timeline visualization? (y/n): ").lower().strip() == 'y'
    # visualize = False
    
    # DEBUG: Print what we're passing to run_cross_validation
    print(f"\n{'='*70}")
    print("STARTING CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Passing {len(filtered_files)} files to run_cross_validation")
    print(f"n_folds: {n_folds}")
    print(f"Expected files per fold: ~{len(filtered_files) / n_folds:.1f}")
    print(f"Expected train size per fold: ~{len(filtered_files) * (n_folds - 1) / n_folds:.1f} ({(n_folds - 1) / n_folds * 100:.1f}%)")
    print(f"Expected test size per fold: ~{len(filtered_files) / n_folds:.1f} ({100 / n_folds:.1f}%)")
    print(f"{'='*70}\n")
    
    VALIDATION_MODE = 0  # Set to True to run in validation mode
    
    # Run cross-validation with ONLY the matched files
    if VALIDATION_MODE:
        print("\n" + "="*70)
        print("RUNNING IN VALIDATION MODE")
        print("="*70)
        print("Training is DISABLED. Each file will be compared against itself.")
        print("Expected metrics: Precision = 1.0, Recall = 1.0, F1 = 1.0")
        print("="*70 + "\n")
        
        all_fold_results = run_cross_validation_validation_mode(
            filtered_files,
            output_dir,
            start_error_threshold,
            n_folds=n_folds
        )
    else:
        all_fold_results = run_cross_validation(
            filtered_files,
            unfiltered_data_dir,
            output_dir,
            model_name_base,
            start_error_threshold,
            n_folds=n_folds,
            visualize=False
        )
    
    if all_fold_results:
        print('\a')
        print("\n" + "="*70)
        print("CROSS-VALIDATION COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {output_dir}")
        print(f"Total files used: {len(filtered_files)}")
        if missing_unfiltered:
            print(f"Files excluded: {len(missing_unfiltered)}")
    else:
        print("\nCross-validation completed with errors.")

if __name__ == "__main__":
    main()