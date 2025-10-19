import os
import random
import subprocess
import shutil
import sys
from datetime import datetime

# Add evaluation folder to path
sys.path.append('./evaluation')

from segment_evaluator import SegmentEvaluator
from visualize_segments_timeline import visualize_segments_timeline
import matplotlib.pyplot as plt

def write_evaluation_to_file(file_handle, results, evaluator, show_per_segment=True):
    """Write evaluation results to a file in the same format as print_report"""
    file_handle.write("="*60 + "\n")
    file_handle.write("SEGMENT EVALUATION REPORT\n")
    file_handle.write("="*60 + "\n")
    file_handle.write(f"Start Time Threshold: {evaluator.start_threshold}s\n")
    
    file_handle.write("\n--- Detection Metrics (Segment-Based) ---\n")
    file_handle.write(f"F1 Score:         {results['f1_score']:.4f}\n")
    file_handle.write(f"Precision:        {results['precision']:.4f}\n")
    file_handle.write(f"Recall (TPR):     {results['recall']:.4f}\n")
    file_handle.write(f"FPR:              {results['fpr']:.4f}\n")
    auc_val = results.get('auc')
    if auc_val is not None:
        file_handle.write(f"AUC:              {auc_val:.4f}\n")
    else:
        file_handle.write(f"AUC:              N/A (no varied confidence scores)\n")
    
    file_handle.write("\n--- Confusion Matrix (Segments) ---\n")
    file_handle.write(f"True Positives:   {results['true_positives']} segments\n")
    file_handle.write(f"False Positives:  {results['false_positives']} segments\n")
    file_handle.write(f"False Negatives:  {results['false_negatives']} segments\n")
    file_handle.write(f"True Negatives:   {results['true_negatives']} segment IDs\n")
    
    file_handle.write("\n--- Temporal Accuracy ---\n")
    if results['start_errors_mean'] is not None:
        file_handle.write(f"Start Time Error (mean):     {results['start_errors_mean']:.4f}s\n")
        file_handle.write(f"Start Time Error (std):      {results['start_errors_std']:.4f}s\n")
        file_handle.write(f"Start Time Error (max):      {results['start_errors_max']:.4f}s\n")
        file_handle.write(f"End Time Error (mean):       {results['end_errors_mean']:.4f}s\n")
        file_handle.write(f"End Time Error (std):        {results['end_errors_std']:.4f}s\n")
        file_handle.write(f"Duration Error (mean):       {results['duration_errors_mean']:.4f}s\n")
    else:
        file_handle.write("No matched segments for temporal analysis\n")
    
    file_handle.write("\n--- Segment Counts ---\n")
    file_handle.write(f"Predicted Segments:     {results['true_positives'] + results['false_positives']}\n")
    file_handle.write(f"Ground Truth Segments:  {results['true_positives'] + results['false_negatives']}\n")
    file_handle.write(f"Matched Segments:       {results['true_positives']}\n")
    file_handle.write(f"Missed Segments:        {results['false_negatives']}\n")
    file_handle.write(f"False Alarm Segments:   {results['false_positives']}\n")
    
    if 'per_segment_metrics' in results and show_per_segment:
        file_handle.write("\n" + "="*60 + "\n")
        file_handle.write("PER-SEGMENT BREAKDOWN\n")
        file_handle.write("="*60 + "\n")
        
        per_seg = results['per_segment_metrics']
        for seg_id, metrics in sorted(per_seg.items()):
            file_handle.write(f"\n--- Segment ID: {seg_id} ---\n")
            file_handle.write(f"  Predicted:     {metrics['num_predicted']}\n")
            file_handle.write(f"  Ground Truth:  {metrics['num_ground_truth']}\n")
            file_handle.write(f"  Status:        ")
            
            if metrics['detected']:
                if metrics.get('too_early'):
                    file_handle.write("DETECTED (TOO EARLY)\n")
                else:
                    file_handle.write("DETECTED\n")
            elif metrics['false_positive']:
                file_handle.write("FALSE POSITIVE\n")
            elif metrics['false_negative']:
                if metrics.get('too_late'):
                    file_handle.write(f"MISSED (>{evaluator.start_threshold}s LATE)\n")
                else:
                    file_handle.write("MISSED\n")
            else:
                file_handle.write("TRUE NEGATIVE\n")
            
            if metrics['start_error'] is not None:
                file_handle.write(f"  Start Error:   {metrics['start_error']:.4f}s\n")
                file_handle.write(f"  End Error:     {metrics['end_error']:.4f}s\n")
                file_handle.write(f"  Duration Err:  {metrics['duration_error']:.4f}s\n")
    
    file_handle.write("="*60 + "\n")

def clear_annotations(input_file, output_file):
    """Remove the last column (activity label) from each line"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            words = line.strip().split()
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
    model_path = os.path.join('./AL/model', f'{model_name}.pkl.gz')
    return os.path.exists(model_path)

def run_quick_test(train_files, test_files, output_dir, visualize=True, 
                   use_existing_model=False, model_name='quick_test_model',
                   start_error_threshold=15.0):
    """
    Quick test: Train on filtered files, test on unfiltered files
    
    Args:
        train_files: List of file paths for training (from filtered data)
        test_files: List of file paths for testing (from unfiltered data)
        output_dir: Directory to save outputs
        visualize: If True, show timeline visualizations
        use_existing_model: If True, skip training and use existing model
        model_name: Name of the model to use/create
        start_error_threshold: Threshold in seconds for start time error
    """
    print(f"\n{'='*70}")
    print("QUICK TEST MODE - Train on filtered, Test on unfiltered")
    print(f"{'='*70}")
    
    if use_existing_model:
        print(f"Using existing model: {model_name}")
    else:
        print(f"Training new model: {model_name}")
        print(f"Training files (filtered): {len(train_files)}")
    
    print(f"Testing files (unfiltered): {len(test_files)}")
    
    # Create temp directory
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1 & 2: Train model (only if not using existing)
    if not use_existing_model:
        # Combine training files
        train_combined = os.path.join(temp_dir, 'quick_test_train.txt')
        print(f"\nCombining {len(train_files)} training files...")
        with open(train_combined, 'w') as outfile:
            for fname in train_files:
                print(f"  Adding: {fname}")
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())
        
        # Train the model
        print("\nTraining model...")
        cmd = [
            'python', 
            './AL/al.py', 
            '--mode', 'TRAIN', 
            '--data', train_combined, 
            '--model', model_name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        print(result.stdout)
        if result.returncode != 0:
            print("ERROR during training:")
            print(result.stderr)
            return None
        
        print("✓ Model trained successfully")
        
        # Cleanup training file
        os.remove(train_combined)
    else:
        print("\n✓ Using existing trained model")
    
    # Step 3: Initialize evaluator
    evaluator = SegmentEvaluator(
        start_time_threshold_seconds=start_error_threshold
    )
    
    # Step 4: Process each test file
    all_results = []
    
    for idx, test_file in enumerate(test_files):
        print(f"\n{'='*70}")
        print(f"Processing test file {idx+1}/{len(test_files)}: {os.path.basename(test_file)}")
        print(f"{'='*70}")
        
        # Clear annotations
        unannotated_file = os.path.join(temp_dir, f'unannotated_{idx}.txt')
        clear_annotations(test_file, unannotated_file)
        print(f"✓ Cleared annotations")
        
        # Check if unannotated file was created
        if not os.path.exists(unannotated_file):
            print(f"ERROR: Failed to create unannotated file")
            continue
            
        # Count lines
        with open(unannotated_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"  Unannotated file has {line_count} lines")
        
        # Annotate with trained model
        print("\nAnnotating with trained model...")
        
        # The annotated output will be in the current directory as 'data.al'
        annotated_output = './data.al'
        
        # Remove old annotation file if it exists
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
        
        if result.returncode != 0:
            print("ERROR during annotation:")
            print(result.stderr)
            print("\nOutput:")
            print(result.stdout)
            continue
        
        print(result.stdout)
        
        # Check if annotation file was created
        if not os.path.exists(annotated_output):
            print(f"ERROR: Annotation file '{annotated_output}' was not created")
            print("AL might not have generated predictions. Check the model and data.")
            continue
        
        # Count annotated lines
        with open(annotated_output, 'r') as f:
            annotated_line_count = sum(1 for _ in f)
        print(f"✓ Annotated file created with {annotated_line_count} lines")
        
        # Save annotated file for inspection
        saved_annotated = os.path.join(
            output_dir, 
            f'predicted_{idx}_{os.path.basename(test_file)}'
        )
        shutil.copy(annotated_output, saved_annotated)
        print(f"✓ Saved predictions -> {saved_annotated}")
        
        # Step 5: Evaluate using your SegmentEvaluator
        print("\nEvaluating predictions...")
        
        try:
            # Parse both files
            pred_df = evaluator.parse_log_file(saved_annotated)
            gt_df = evaluator.parse_log_file(test_file)
                        
            # Extract confidence scores if available
            confidence_scores = None
            if 'confidence' in pred_df.columns:
                print("  Confidence column found in predictions")
                # Group by segment and get mean confidence per segment
                pred_segments_temp = evaluator.extract_segments(pred_df)
                confidence_scores = []
                
                for seg in pred_segments_temp:
                    seg_id = seg['segment_id']
                    seg_start = seg['start_time']
                    seg_end = seg['end_time']
                    
                    # Get confidence scores for events in this segment's time range
                    # Filter by timestamp and check if annotation matches the segment ID
                    seg_mask = (pred_df['timestamp'] >= seg_start) & (pred_df['timestamp'] <= seg_end)
                    seg_df = pred_df[seg_mask]
                    
                    # Further filter by activity - extract number from annotation
                    matching_rows = []
                    for idx_row, row in seg_df.iterrows():
                        annotation = str(row['annotation']).strip()
                        # Extract activity ID from annotation (handles "1", "1.1", "1-start", etc.)
                        import re
                        match = re.match(r'^(\d+)', annotation)
                        if match and match.group(1) == seg_id:
                            matching_rows.append(row)
                    
                    if matching_rows:
                        # Calculate mean confidence for this segment
                        confidences = [r['confidence'] for r in matching_rows]
                        mean_conf = sum(confidences) / len(confidences)
                        confidence_scores.append(mean_conf)
                    else:
                        # No matching events found, use default
                        confidence_scores.append(0.5)
                
                print(f"  Extracted {len(confidence_scores)} confidence scores for {len(pred_segments_temp)} segments")
                if confidence_scores:
                    import numpy as np
                    print(f"  Confidence range: {np.min(confidence_scores):.3f} to {np.max(confidence_scores):.3f}, mean: {np.mean(confidence_scores):.3f}")
            else:
                print("  No confidence column found in predictions")
            
            # Extract segments
            pred_segments = evaluator.extract_segments(pred_df)
            gt_segments = evaluator.extract_segments(gt_df)
            
            print(f"  Ground truth segments: {len(gt_segments)}")
            print(f"  Predicted segments: {len(pred_segments)}")
            
            if len(gt_segments) == 0:
                print("ERROR: No ground truth segments found. Skipping this file.")
                continue
            if len(pred_segments) == 0:
                print("ERROR: No predicted segments found. Skipping this file.")
                continue
            
            # Calculate metrics with confidence scores
            print("\nCalculating metrics...")
            results = evaluator.evaluate_segments(
                pred_segments, 
                gt_segments,
                confidence_scores=confidence_scores,
                per_segment=True  # We need this to extract per-segment for aggregation
            )
            
            print(f"✓ Evaluation complete")
            
            # Store results
            all_results.append({
                'file': test_file,
                'results': results,
                'pred_segments': pred_segments,
                'gt_segments': gt_segments,
                'confidence_scores': confidence_scores
            })
            
            # Step 6: Visualize timeline
            if visualize and len(gt_segments) > 0:
                print("\nGenerating timeline visualization...")
                try:
                    fig, axes = visualize_segments_timeline(
                        predicted_segments=pred_segments,
                        ground_truth_segments=gt_segments,
                        title=f"Timeline: {os.path.basename(test_file)}"
                    )
                    plt.tight_layout()
                    
                    # Save figure
                    viz_file = os.path.join(
                        output_dir,
                        f'timeline_{idx}_{os.path.splitext(os.path.basename(test_file))[0]}.png'
                    )
                    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
                    print(f"✓ Saved visualization -> {viz_file}")
                    plt.close()
                    
                except Exception as vis_err:
                    print(f"Visualization error: {vis_err}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup annotated file
        if os.path.exists(annotated_output):
            os.remove(annotated_output)
    
    # Summary - consolidated by activity
    print(f"\n{'='*70}")
    print("CONSOLIDATED EVALUATION BY ACTIVITY (ALL TEST FILES)")
    print(f"{'='*70}")
    
    if all_results:
        # Aggregate per-segment metrics across all files
        aggregated_segments = {}
        
        for item in all_results:
            pred_segs = item['pred_segments']
            gt_segs = item['gt_segments']
            
            # Get confidence scores for this file if available
            file_confidence_scores = item.get('confidence_scores', None)
            
            # Initialize aggregated segments
            for seg_id in range(1, 9):
                seg_id_str = str(seg_id)
                
                if seg_id_str not in aggregated_segments:
                    aggregated_segments[seg_id_str] = {
                        'total_predicted': 0,
                        'total_ground_truth': 0,
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'start_errors': [],
                        'end_errors': [],
                        'duration_errors': [],
                        'confidence_scores': [],
                        'prediction_labels': [],
                    }
                
                # Count predicted segments for this ID
                pred_count = len([s for s in pred_segs if s.get('segment_id') == seg_id_str])
                gt_count = len([s for s in gt_segs if s.get('segment_id') == seg_id_str])
                
                aggregated_segments[seg_id_str]['total_predicted'] += pred_count
                aggregated_segments[seg_id_str]['total_ground_truth'] += gt_count
                
            # Calculate TP, FP, FN from the per_segment_metrics
            results = item['results']
            if 'per_segment_metrics' in results:
                per_seg = results['per_segment_metrics']
                for seg_id, metrics in per_seg.items():
                    if seg_id not in aggregated_segments:
                        continue
                        
                    agg = aggregated_segments[seg_id]
                    
                    num_pred = metrics['num_predicted']
                    num_gt = metrics['num_ground_truth']
                    
                    # Collect confidence scores for this specific segment instance
                    if file_confidence_scores:
                        file_pred_segs_for_activity = [s for s in pred_segs if s.get('segment_id') == seg_id]
                        for pred_seg in file_pred_segs_for_activity:
                            # Find the index of this segment in the original pred_segs list
                            try:
                                seg_idx = pred_segs.index(pred_seg)
                                if seg_idx < len(file_confidence_scores):
                                    conf = file_confidence_scores[seg_idx]
                                    
                                    # Determine if this specific prediction is TP or FP based on metrics
                                    if metrics['detected'] and num_pred > 0:
                                        # First prediction is TP, rest are FP
                                        is_first = (file_pred_segs_for_activity.index(pred_seg) == 0)
                                        is_tp = is_first
                                    else:
                                        # Not detected, so all predictions are FP
                                        is_tp = False
                                    
                                    agg['confidence_scores'].append(conf)
                                    agg['prediction_labels'].append(1 if is_tp else 0)
                            except (ValueError, IndexError):
                                pass
                    
                    if metrics['detected']:
                        # Successfully detected (already within start_threshold)
                        agg['true_positives'] += 1
                        
                        # Extra predictions beyond the first match
                        if num_pred > 1:
                            agg['false_positives'] += (num_pred - 1)
                        if num_gt > 1:
                            agg['false_negatives'] += (num_gt - 1)
        
                    elif metrics.get('false_positive'):
                        agg['false_positives'] += num_pred
                        
                    elif metrics.get('false_negative'):
                        agg['false_negatives'] += num_gt
                        
                        if num_pred > 0:
                            agg['false_positives'] += num_pred

                    # Collect timing errors
                    if metrics['start_error'] is not None:
                        agg['start_errors'].append(metrics['start_error'])
                        if metrics['end_error'] is not None:
                            agg['end_errors'].append(metrics['end_error'])
                        if metrics['duration_error'] is not None:
                            agg['duration_errors'].append(metrics['duration_error'])
                        
        # Write consolidated report
        report_path = os.path.join(output_dir, 'evaluation_by_activity.txt')
        with open(report_path, 'w') as report_file:
            report_file.write("="*70 + "\n")
            report_file.write("EVALUATION BY ACTIVITY (CONSOLIDATED ACROSS ALL TEST FILES)\n")
            report_file.write("="*70 + "\n")
            report_file.write(f"Model: {model_name}\n")
            report_file.write(f"Training files: {len(train_files)} (filtered)\n")
            report_file.write(f"Test files: {len(test_files)} (unfiltered)\n")
            report_file.write(f"Start time threshold: {evaluator.start_threshold}s\n")
            report_file.write("="*70 + "\n\n")
            
            # Prepare for combined ROC plot
            from sklearn.metrics import roc_auc_score, roc_curve
            import numpy as np
            
            fig_combined, ax_combined = plt.subplots(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, 8))
            
            for seg_id in sorted(aggregated_segments.keys()):
                agg = aggregated_segments[seg_id]
                
                # Calculate per-segment AUC if we have confidence scores
                seg_auc = None
                fpr_seg = None
                tpr_seg = None
                
                if agg['confidence_scores'] and agg['prediction_labels']:
                    try:
                        y_true = np.array(agg['prediction_labels'])
                        y_scores = np.array(agg['confidence_scores'])
                        
                        print(f"  Debug Activity {seg_id}: {len(y_true)} predictions, {sum(y_true)} TPs, {len(y_true)-sum(y_true)} FPs")
                        print(f"  Debug Activity {seg_id}: {len(np.unique(y_scores))} unique confidence values, {len(np.unique(y_true))} unique labels")
                        
                        # Only calculate if we have varied scores and both classes
                        if len(np.unique(y_scores)) > 1 and len(np.unique(y_true)) > 1:
                            seg_auc = roc_auc_score(y_true, y_scores)
                            
                            # Calculate ROC curve for plotting
                            fpr_seg, tpr_seg, thresholds = roc_curve(y_true, y_scores)
                            
                            # Add to combined plot
                            seg_idx = int(seg_id) - 1
                            ax_combined.plot(fpr_seg, tpr_seg, color=colors[seg_idx], lw=2,
                                           label=f'Activity {seg_id} (AUC = {seg_auc:.3f})')
                            print(f"  Debug Activity {seg_id}: ✓ ROC/AUC calculated successfully: {seg_auc:.4f}")
                        else:
                            if len(np.unique(y_true)) == 1:
                                print(f"  Debug Activity {seg_id}: ✗ Only one class present (all {y_true[0]}s)")
                            else:
                                print(f"  Debug Activity {seg_id}: ✗ All confidence scores identical")
                            
                    except Exception as e:
                        print(f"  Warning: Could not calculate AUC for activity {seg_id}: {e}")
                else:
                    print(f"  Debug Activity {seg_id}: ✗ No confidence scores available")
                
                # Calculate precision, recall, F1 for this activity
                tp = agg['true_positives']
                fp = agg['false_positives']
                fn = agg['false_negatives']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Print to terminal
                print(f"\n--- Activity {seg_id} ---")
                print(f"  Total Predicted:     {agg['total_predicted']}")
                print(f"  Total Ground Truth:  {agg['total_ground_truth']}")
                print(f"  True Positives:      {tp}")
                print(f"  False Positives:     {fp}")
                print(f"  False Negatives:     {fn}")
                print(f"  Precision:           {precision:.4f}")
                print(f"  Recall:              {recall:.4f}")
                print(f"  F1 Score:            {f1:.4f}")
                if seg_auc is not None:
                    print(f"  AUC:                 {seg_auc:.4f}")
                
                # Write to file
                report_file.write(f"\n--- Activity {seg_id} ---\n")
                report_file.write(f"  Total Predicted:     {agg['total_predicted']}\n")
                report_file.write(f"  Total Ground Truth:  {agg['total_ground_truth']}\n")
                report_file.write(f"  True Positives:      {tp}\n")
                report_file.write(f"  False Positives:     {fp}\n")
                report_file.write(f"  False Negatives:     {fn}\n")
                report_file.write(f"  Precision:           {precision:.4f}\n")
                report_file.write(f"  Recall:              {recall:.4f}\n")
                report_file.write(f"  F1 Score:            {f1:.4f}\n")
                if seg_auc is not None:
                    report_file.write(f"  AUC:                 {seg_auc:.4f}\n")
                
                if agg['start_errors']:
                    # Filter out None values
                    start_errs = [e for e in agg['start_errors'] if e is not None]
                    end_errs = [e for e in agg['end_errors'] if e is not None]
                    dur_errs = [e for e in agg['duration_errors'] if e is not None]
                    
                    if start_errs:
                        mean_start = np.mean(start_errs)
                        std_start = np.std(start_errs)
                        total_matched = len(start_errs)
                        
                        print(f"  Start Error (mean):  {mean_start:.4f}s (±{std_start:.4f}s)")
                        report_file.write(f"  Start Error (mean):  {mean_start:.4f}s (±{std_start:.4f}s)\n")
                    
                    if end_errs:
                        mean_end = np.mean(end_errs)
                        print(f"  End Error (mean):    {mean_end:.4f}s")
                        report_file.write(f"  End Error (mean):    {mean_end:.4f}s\n")
                    
                    if dur_errs:
                        mean_duration = np.mean(dur_errs)
                        print(f"  Duration Err (mean): {mean_duration:.4f}s")
                        report_file.write(f"  Duration Err (mean): {mean_duration:.4f}s\n")
                            
            print(f"{'='*70}")
            report_file.write("="*70 + "\n")
            
            # Finalize and save combined ROC plot
            ax_combined.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                            label='Random classifier', alpha=0.5)
            ax_combined.set_xlim([0.0, 1.0])
            ax_combined.set_ylim([0.0, 1.05])
            ax_combined.set_xlabel('False Positive Rate', fontsize=12)
            ax_combined.set_ylabel('True Positive Rate', fontsize=12)
            ax_combined.set_title('ROC Curves by Activity (Aggregated)', fontsize=14, fontweight='bold')
            ax_combined.legend(loc="lower right", fontsize=9)
            ax_combined.grid(alpha=0.3)
            plt.tight_layout()
            
            combined_roc_file = os.path.join(output_dir, 'roc_by_activity.png')
            plt.savefig(combined_roc_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved ROC curve by activity -> {combined_roc_file}")
        
        print(f"✓ Saved consolidated report -> {report_path}")
    
    else:
        print("\nNo results to summarize. Check errors above.")
    
    # Cleanup temp directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    return all_results


def get_matching_unfiltered_file(filtered_file, unfiltered_dir='./data_spaces'):
    """
    Find the corresponding unfiltered file for a filtered file
    
    Args:
        filtered_file: Path to filtered file (e.g., './data_filtered/P001.txt')
        unfiltered_dir: Directory containing unfiltered files
        
    Returns:
        Path to unfiltered file or None if not found
    """
    basename = os.path.basename(filtered_file)
    unfiltered_path = os.path.join(unfiltered_dir, basename)
    
    if os.path.exists(unfiltered_path):
        return unfiltered_path
    else:
        # File doesn't exist - this is expected if not all files are in both dirs
        return None


def main():
    # Configuration
    random.seed(42)
    
    filtered_data_dir = './data_filtered'  # For training
    unfiltered_data_dir = './data_spaces'  # For testing
    model_name = 'quick_test_model'
    start_error_threshold = 15.0  # Define the threshold here
    
    # Create timestamped output directory with descriptive name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract the last part of the filtered_data_dir for the folder name
    filtered_dir_name = os.path.basename(os.path.normpath(filtered_data_dir))
    
    # Create descriptive folder name: timestamp_filtereddir_threshold
    output_dir_name = f'run_{timestamp}_{filtered_dir_name}_thresh{int(start_error_threshold)}s'
    output_dir = os.path.join('./output', output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all filtered data files for training (use ALL of them)
    filtered_files = [
        os.path.join(filtered_data_dir, f) 
        for f in os.listdir(filtered_data_dir) 
        if f.endswith('.txt')
    ]
    
    print(f"Found {len(filtered_files)} filtered files in {filtered_data_dir}")
    
    if len(filtered_files) == 0:
        print(f"ERROR: No filtered files found in {filtered_data_dir}")
        return
    
    # Shuffle filtered files for training
    random.shuffle(filtered_files)
    
    split_index = int(len(filtered_files) * 0.7)
    train_files = filtered_files[:split_index]
    test_candidate_files = filtered_files[split_index:]

    print(f"Split: {len(train_files)} training files (70%), {len(test_candidate_files)} test candidates (30%)")
    
    # Get corresponding unfiltered files for testing
    # Only use unfiltered files that have a match in filtered
    test_files = []
    for filtered_file in test_candidate_files:
        unfiltered_file = get_matching_unfiltered_file(filtered_file, unfiltered_data_dir)
        if unfiltered_file:
            test_files.append(unfiltered_file)
    
    # If we don't have enough test files from the split, try to find more
    if len(test_files) < 3:
        print(f"Warning: Only found {len(test_files)} matching test files from the split.")
        print("Looking for additional matching files...")
        
        # Get all unfiltered files
        all_unfiltered = [
            os.path.join(unfiltered_data_dir, f)
            for f in os.listdir(unfiltered_data_dir)
            if f.endswith('.txt')
        ]
        
        # Find which ones have matches in filtered (but weren't used for training)
        train_basenames = {os.path.basename(f) for f in train_files}
        
        for unfiltered_file in all_unfiltered:
            basename = os.path.basename(unfiltered_file)
            # Check if this file exists in filtered AND wasn't used for training
            filtered_match = os.path.join(filtered_data_dir, basename)
            if os.path.exists(filtered_match) and basename not in train_basenames:
                if unfiltered_file not in test_files:
                    test_files.append(unfiltered_file)
                    if len(test_files) >= 3:
                        break
    
    if len(test_files) == 0:
        print(f"ERROR: Could not find any unfiltered test files in {unfiltered_data_dir}")
        print(f"       that have matching filtered files in {filtered_data_dir}")
        
        # Show what we have
        print(f"\nSample filtered files:")
        for f in filtered_files[:5]:
            print(f"  - {os.path.basename(f)}")
        
        print(f"\nSample unfiltered files:")
        unfiltered_samples = [f for f in os.listdir(unfiltered_data_dir) if f.endswith('.txt')][:5]
        for f in unfiltered_samples:
            print(f"  - {f}")
        
        return
    
    print(f"\n{'='*70}")
    print("DATA CONFIGURATION")
    print(f"{'='*70}")
    print(f"Training on: {len(train_files)} FILTERED files from {filtered_data_dir}")
    print(f"Testing on: {len(test_files)} UNFILTERED files from {unfiltered_data_dir}")
    print(f"Start error threshold: {start_error_threshold}s")
    
    # Check if model already exists
    model_exists = check_existing_model(model_name)
    use_existing = False
    
    if model_exists:
        print(f"\n{'='*70}")
        print(f"EXISTING MODEL FOUND: {model_name}")
        print(f"{'='*70}")
        print(f"Model location: ./AL/model/{model_name}.pkl.gz")
        
        choice = input("\nDo you want to use the existing model? (y/n): ").lower().strip()
        
        if choice == 'y':
            use_existing = True
            print("\n✓ Will use existing model (skipping training)")
        else:
            print("\n✓ Will train a new model (existing model will be overwritten)")
            print(f"\nTrain files ({len(train_files)}) from FILTERED data:")
            for f in train_files[:5]:  # Show first 5
                print(f"  - {os.path.basename(f)}")
            if len(train_files) > 5:
                print(f"  ... and {len(train_files) - 5} more")
    else:
        print(f"\nNo existing model found. Will train new model: {model_name}")
        print(f"\nTrain files ({len(train_files)}) from FILTERED data:")
        for f in train_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(train_files) > 5:
            print(f"  ... and {len(train_files) - 5} more")
    
    print(f"\nTest files ({len(test_files)}) from UNFILTERED data:")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")
    
    # Ask about visualization
    visualize = input("\nEnable timeline visualization? (y/n): ").lower().strip() == 'y'
    
    # Run the test - pass the threshold
    results = run_quick_test(
        train_files, 
        test_files, 
        output_dir,
        visualize=visualize,
        use_existing_model=use_existing,
        model_name=model_name,
        start_error_threshold=start_error_threshold  # Pass threshold here
    )
    
    print(f"\n{'='*70}")
    print("Quick test complete!")
    print(f"{'='*70}")
    
    # Show what files were generated
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    if os.path.exists(output_dir):
        generated = sorted(os.listdir(output_dir))
        for f in generated:
            print(f"  - {f}")
    
    print(f"\nModel location:")
    print(f"  - ./AL/model/{model_name}.pkl.gz")


if __name__ == "__main__":
    main()