import os
import random
import subprocess
import shutil
import sys
from datetime import datetime
import json
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# Add evaluation folder to path
sys.path.append('./evaluation')

from segment_evaluator import SegmentEvaluator
from visualize_segments_timeline import visualize_segments_timeline


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
        if 'activity_classes' in filtered_results['metadata']:
            filtered_results['metadata']['activity_classes'] = [
                ac for ac in filtered_results['metadata']['activity_classes']
                if ac in activity_filter
            ]
    
    # Copy top-level metrics
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
                filtered_results['frame_level'][activity_id] = {
                    'auc': roc_data.get('auc')
                }
    
    # Filter segment_level ROC - KEEP ONLY AUC
    if 'segment_level' in results:
        filtered_results['segment_level'] = {}
        for activity_id, roc_data in results['segment_level'].items():
            if activity_id in activity_filter:
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
    """Remove the last column (activity label) from each line"""
    with open(input_file, 'r', encoding='utf-8-sig') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            words = line.split()
            if len(words) >= 4:
                # Keep only first 4 fields: date, time, sensor, status
                new_line = ' '.join(words[:4]) + '\n'
                f_out.write(new_line)


def count_activity_instances(file_path, evaluator, activity_ids=['1','2','3','4','5','6','7','8']):
    """Count how many instances of each activity are in a file"""
    try:
        df = evaluator.parse_log_file(file_path)
        
        if df is None or len(df) == 0:
            return {aid: 0 for aid in activity_ids}
        
        segments = evaluator.extract_segments(df)
        
        counts = {aid: 0 for aid in activity_ids}
        for seg in segments:
            if seg['segment_id'] in activity_ids:
                counts[seg['segment_id']] += 1
        
        return counts
    except Exception as e:
        print(f"Warning: Could not count activities in {file_path}: {e}")
        return {aid: 0 for aid in activity_ids}


# ============================================================================
# CROSS-VALIDATION FUNCTIONS
# ============================================================================

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
    
    # Create activity presence matrix
    file_list = list(file_activity_counts.keys())
    
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
    
    # Initialize folds
    folds = [[] for _ in range(n_folds)]
    
    # Distribute each pattern group across folds
    for pattern, indices in pattern_groups.items():
        random.shuffle(indices)
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
        
        fold_activity_counts = {aid: 0 for aid in activity_ids}
        for file_path in folds[fold_idx]:
            for aid in activity_ids:
                fold_activity_counts[aid] += file_activity_counts[file_path][aid]
        
        for aid in activity_ids:
            print(f"  Activity {aid}: {fold_activity_counts[aid]} instances")
    
    print(f"{'='*70}\n")
    
    return folds, file_activity_counts


def run_fold(fold_idx, train_files, test_files, output_dir, model_name,
             evaluator, visualize=False):
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
    
    # Visualize if requested
    if visualize:
        print(f"\n{'='*70}")
        print(f"GENERATING TIMELINE VISUALIZATIONS FOR FOLD {fold_idx + 1}")
        print(f"{'='*70}")
        
        for idx, (pred_file, gt_file) in enumerate(zip(predicted_files, ground_truth_files)):
            base_name = os.path.basename(gt_file).replace('.txt', '')
            viz_output = os.path.join(output_dir, f'timeline_{base_name}.png')
            
            try:
                pred_df = evaluator.parse_log_file(pred_file)
                gt_df = evaluator.parse_log_file(gt_file)
                
                pred_segments_raw = evaluator.extract_segments(pred_df)
                gt_segments_raw = evaluator.extract_segments(gt_df)
                
                def ensure_datetime_format(segments):
                    """Ensure segments have proper datetime objects"""
                    formatted = []
                    for seg in segments:
                        formatted_seg = {
                            'segment_id': str(seg['segment_id']),
                            'start_time': seg['start_time'],
                            'end_time': seg['end_time']
                        }
                        if hasattr(formatted_seg['start_time'], 'to_pydatetime'):
                            formatted_seg['start_time'] = formatted_seg['start_time'].to_pydatetime()
                        if hasattr(formatted_seg['end_time'], 'to_pydatetime'):
                            formatted_seg['end_time'] = formatted_seg['end_time'].to_pydatetime()
                        formatted.append(formatted_seg)
                    return formatted
                
                pred_segments = ensure_datetime_format(pred_segments_raw)
                gt_segments = ensure_datetime_format(gt_segments_raw)
                
                fig, ax = visualize_segments_timeline(
                    predicted_segments=pred_segments,
                    ground_truth_segments=gt_segments,
                    segment_ids=['1', '2', '3', '4', '5', '6', '7', '8'],
                    title=f"Timeline: {base_name}"
                )
                
                fig.savefig(viz_output, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  [{idx + 1}/{len(predicted_files)}] Saved: {base_name}")
                    
            except Exception as e:
                print(f"  [{idx + 1}/{len(predicted_files)}] Error: {base_name} - {e}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return multi_results


def print_fold_comparison(all_fold_results, activity_filter):
    """Print comparison of metrics across all folds"""
    print(f"\n{'='*70}")
    print("CROSS-FOLD COMPARISON")
    print(f"{'='*70}")
    
    # Aggregate metrics
    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)
    
    # Micro-average metrics
    micro_precision = [r['micro_avg']['precision'] for r in all_fold_results]
    micro_recall = [r['micro_avg']['recall'] for r in all_fold_results]
    micro_f1 = [r['micro_avg']['f1_score'] for r in all_fold_results]
    
    print(f"{'Micro Precision':<20} {np.mean(micro_precision):>10.4f} {np.std(micro_precision):>10.4f} "
          f"{np.min(micro_precision):>10.4f} {np.max(micro_precision):>10.4f}")
    print(f"{'Micro Recall':<20} {np.mean(micro_recall):>10.4f} {np.std(micro_recall):>10.4f} "
          f"{np.min(micro_recall):>10.4f} {np.max(micro_recall):>10.4f}")
    print(f"{'Micro F1':<20} {np.mean(micro_f1):>10.4f} {np.std(micro_f1):>10.4f} "
          f"{np.min(micro_f1):>10.4f} {np.max(micro_f1):>10.4f}")
    
    # Per-activity metrics
    print(f"\n{'='*70}")
    print("PER-ACTIVITY METRICS ACROSS FOLDS")
    print(f"{'='*70}")
    
    for activity_id in activity_filter:
        print(f"\n--- Activity {activity_id} ---")
        
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
                        evaluator, n_folds=5, visualize=False):
    """Run k-fold cross-validation"""
    
    activity_filter = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    # Create stratified folds
    folds, file_activity_counts = create_stratified_folds(
        files, evaluator, n_folds=n_folds, activity_ids=activity_filter
    )
    
    # Run each fold
    all_fold_results = []
    
    for fold_idx in range(n_folds):
        # Create train/test split
        test_file_paths_filtered = folds[fold_idx]
        train_file_paths_filtered = []
        for i in range(n_folds):
            if i != fold_idx:
                train_file_paths_filtered.extend(folds[i])
        
        # Get matching UNFILTERED test files
        test_file_paths_unfiltered = []
        for filtered_path in test_file_paths_filtered:
            basename = os.path.basename(filtered_path)
            unfiltered_path = os.path.join(unfiltered_dir, basename)
            if os.path.exists(unfiltered_path):
                test_file_paths_unfiltered.append(unfiltered_path)
        
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*70}")
        print(f"Training: {len(train_file_paths_filtered)} files (filtered)")
        print(f"Testing: {len(test_file_paths_unfiltered)} files (unfiltered)")
        
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
        roc_plot_path = os.path.join(fold_output_dir, f'roc_fold{fold_idx + 1}.png')
        try:
            evaluator.plot_dual_roc_curves(
                fold_results,
                save_path=roc_plot_path,
                activity_filter=activity_filter
            )
            print(f"[OK] Fold {fold_idx + 1} ROC curves saved")
        except Exception as e:
            print(f"Error generating ROC for fold {fold_idx + 1}: {e}")
    
    # Print cross-fold comparison and save results
    if len(all_fold_results) > 0:
        print_fold_comparison(all_fold_results, activity_filter)
        
        # Generate aggregated ROC curves
        print(f"\nGenerating aggregated ROC curves across all {n_folds} folds...")
        evaluator.generate_aggregated_roc_curves_vertical_avg(
            all_fold_results,
            output_dir,
            activity_filter
        )
        
        # Filter results and save
        filtered_fold_results = []
        for fold_result in all_fold_results:
            filtered_result = filter_results_by_activity(fold_result, activity_filter)
            filtered_fold_results.append(filtered_result)

        # Save per-fold summaries
        for fold_idx, filtered_result in enumerate(filtered_fold_results):
            fold_json_path = os.path.join(output_dir, f'fold_{fold_idx + 1}', 'fold_summary.json')
            with open(fold_json_path, 'w') as f:
                json.dump({
                    'fold': fold_idx + 1,
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
                    'timestamp': datetime.now().isoformat()
                },
                'aggregated': convert_to_json_serializable(aggregated_summary)
            }, f, indent=2)

        print(f"\n[OK] Cross-validation results saved:")
        print(f"  - Aggregated summary: {agg_json_path}")
        print(f"  - Per-fold summaries: ./fold_N/fold_summary.json")
    
    return all_fold_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    # Configuration
    random.seed(42)
    
    filtered_data_dir = './data_filtered'
    unfiltered_data_dir = './data_spaces'
    model_name_base = 'cv_model'
    n_folds = 5
    
    # Evaluator configuration
    timeline_resolution = 1.0  # seconds
    start_time_threshold = 0   # seconds
    overlap_threshold = 0.5    # segment overlap threshold
    
    
    # Get all filtered files
    all_filtered_files = [
        os.path.join(filtered_data_dir, f) 
        for f in os.listdir(filtered_data_dir) 
        if f.endswith('.txt')
    ]
    
    # Only keep files that have matches in BOTH directories
    filtered_files = []
    for filtered_file in all_filtered_files:
        basename = os.path.basename(filtered_file)
        unfiltered_file = os.path.join(unfiltered_data_dir, basename)
        if os.path.exists(unfiltered_file):
            filtered_files.append(filtered_file)
    
    if len(filtered_files) == 0:
        print(f"\nERROR: No matching file pairs found!")
        print(f"Filtered dir: {filtered_data_dir}")
        print(f"Unfiltered dir: {unfiltered_data_dir}")
        return
    
    print(f"\n[OK] Using {len(filtered_files)} matched file pairs for cross-validation")
    
    # Ask about visualization
    visualize = input("Enable timeline visualization? (y/n): ").lower().strip() == 'y'
    
    # Ask about overlap threshold
    overlap_threshold = float(input("Overlap threshold (0.0-1.0): ").strip())
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'run_{timestamp}_{n_folds}fold_overlap{overlap_threshold:.3f}'
    output_dir = os.path.join('./output', run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save run configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_folds': n_folds,
            'timeline_resolution_seconds': timeline_resolution,
            'start_time_threshold_seconds': start_time_threshold,
            'overlap_threshold': overlap_threshold,
            'filtered_data_dir': filtered_data_dir,
            'unfiltered_data_dir': unfiltered_data_dir,
            'num_files': len(filtered_files)
        }, f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Configuration saved: {config_path}")
    
    # Create evaluator
    evaluator = SegmentEvaluator(
        timeline_resolution_seconds=timeline_resolution,
        start_time_threshold_seconds=start_time_threshold,
        overlap_threshold=overlap_threshold
    )
    
    # Run cross-validation
    print(f"\n{'='*70}")
    print("STARTING CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Files: {len(filtered_files)}")
    print(f"Folds: {n_folds}")
    print(f"Timeline resolution: {timeline_resolution}s")
    print(f"Overlap threshold: {overlap_threshold}")
    print(f"{'='*70}\n")
    
    all_fold_results = run_cross_validation(
        filtered_files,
        unfiltered_data_dir,
        output_dir,
        model_name_base,
        evaluator,
        n_folds=n_folds,
        visualize=visualize
    )
    
    if all_fold_results:
        print('\a')  # Beep
        print("\n" + "="*70)
        print("CROSS-VALIDATION COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {output_dir}")
        print(f"Total files used: {len(filtered_files)}")
    else:
        print("\nCross-validation completed with errors.")


if __name__ == "__main__":
    main()