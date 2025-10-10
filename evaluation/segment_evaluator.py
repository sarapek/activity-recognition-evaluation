import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
from datetime import datetime
import pandas as pd
from typing import List, Tuple, Dict
import re

class SegmentEvaluator:
    """Evaluates segment-based recognition against ground truth"""
    
    def __init__(self, time_tolerance_seconds=2.0, start_time_threshold_seconds=15.0):
        """
        Args:
            time_tolerance_seconds: Maximum allowed time error for "accurate" detection
            start_time_threshold_seconds: Maximum delay allowed for recognition to count as detection (not a miss)
        """
        self.time_tolerance = time_tolerance_seconds
        self.start_threshold = start_time_threshold_seconds
        
    def parse_log_file(self, filepath: str) -> pd.DataFrame:
        """Parse the log file into a structured DataFrame (space-separated format)"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                # Detect format: 6 parts = AL output (with duplicate sensor)
                #                5 parts = ground truth (no duplicate sensor)
                if len(parts) >= 6:
                    # AL format: date time sensor newsensor status activity
                    try:
                        date_str = parts[0]
                        time_str = parts[1]
                        timestamp_str = f"{date_str} {time_str}"
                        
                        # Parse timestamp
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        sensor_id = parts[2]
                        # parts[3] is newsensor (duplicate, skip)
                        status = parts[4]
                        activity = parts[5]
                        
                        data.append({
                            'timestamp': timestamp,
                            'event_id': sensor_id,
                            'state': status,
                            'annotation': activity,
                            'errors': ''
                        })
                    except (ValueError, IndexError) as e:
                        continue
                        
                elif len(parts) >= 5:
                    # Ground truth format: date time sensor status activity
                    try:
                        date_str = parts[0]
                        time_str = parts[1]
                        timestamp_str = f"{date_str} {time_str}"
                        
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        sensor_id = parts[2]
                        status = parts[3]  # Changed from parts[4]
                        activity = parts[4]  # Changed from parts[5]
                        
                        data.append({
                            'timestamp': timestamp,
                            'event_id': sensor_id,
                            'state': status,
                            'annotation': activity,
                            'errors': ''
                        })
                    except (ValueError, IndexError) as e:
                        continue
        
        return pd.DataFrame(data)
    
    def extract_segments(self, df: pd.DataFrame) -> List[Dict]:
        """Extract segments from annotations (continuous activity numbers or start/end markers)"""
        segments = []
        current_activity = None
        segment_start = None
        
        for idx, row in df.iterrows():
            annotation = str(row['annotation']).strip()
            
            # Skip empty annotations or Other_Activity
            if not annotation or annotation == 'nan' or annotation == 'Other_Activity':
                # Close current segment if any
                if current_activity is not None and segment_start is not None:
                    segments.append({
                        'segment_id': current_activity,
                        'start_time': segment_start,
                        'end_time': prev_timestamp,
                        'duration': (prev_timestamp - segment_start).total_seconds()
                    })
                    current_activity = None
                    segment_start = None
                prev_timestamp = row['timestamp']
                continue
            
            # Extract activity ID from annotation
            # Handle both "1-start", "1-end", "1.1" and plain "1"
            if '-start' in annotation.lower():
                activity_id = annotation.lower().split('-start')[0].strip()
                # Start new segment
                if current_activity is not None and segment_start is not None:
                    # Close previous segment
                    segments.append({
                        'segment_id': current_activity,
                        'start_time': segment_start,
                        'end_time': prev_timestamp,
                        'duration': (prev_timestamp - segment_start).total_seconds()
                    })
                current_activity = activity_id
                segment_start = row['timestamp']
            
            elif '-end' in annotation.lower():
                activity_id = annotation.lower().split('-end')[0].strip()
                # End current segment
                if current_activity == activity_id and segment_start is not None:
                    segments.append({
                        'segment_id': current_activity,
                        'start_time': segment_start,
                        'end_time': row['timestamp'],
                        'duration': (row['timestamp'] - segment_start).total_seconds()
                    })
                    current_activity = None
                    segment_start = None
            
            else:
                # Continuous activity label (e.g., "1", "2", "1.1")
                # Extract just the activity number
                match = re.match(r'^(\d+)', annotation)
                if match:
                    activity_id = match.group(1)
                    
                    # If this is a different activity, close previous and start new
                    if activity_id != current_activity:
                        if current_activity is not None and segment_start is not None:
                            segments.append({
                                'segment_id': current_activity,
                                'start_time': segment_start,
                                'end_time': prev_timestamp,
                                'duration': (prev_timestamp - segment_start).total_seconds()
                            })
                        current_activity = activity_id
                        segment_start = row['timestamp']
                    # Otherwise, continue current segment
            
            prev_timestamp = row['timestamp']
        
        # Close any remaining open segment
        if current_activity is not None and segment_start is not None:
            segments.append({
                'segment_id': current_activity,
                'start_time': segment_start,
                'end_time': prev_timestamp,
                'duration': (prev_timestamp - segment_start).total_seconds()
            })
        
        return segments
    
    def extract_errors(self, df: pd.DataFrame) -> List[Dict]:
        """Extract error events from the log"""
        errors = []
        for idx, row in df.iterrows():
            if row['errors'] and 'error' in row['errors'].lower():
                error_parts = row['errors'].split(',')
                for err in error_parts:
                    if 'error' in err.lower():
                        errors.append({
                            'timestamp': row['timestamp'],
                            'error_type': err.strip(),
                            'event_id': row['event_id']
                        })
        return errors
    
    def calculate_per_segment_metrics(self,
                                     predicted_segments: List[Dict],
                                     ground_truth_segments: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate metrics for each individual segment ID
        
        Returns:
            Dictionary with segment_id as key and metrics as value
        """
        per_segment_results = {}

        # Get all unique segment IDs
        all_segment_ids = set()
        for seg in predicted_segments + ground_truth_segments:
            seg_id = seg.get('segment_id', 'unknown')
            try:
                if int(seg_id) < 9:
                    all_segment_ids.add(seg_id)
            except (ValueError, TypeError):
                continue
        
        # Sort segment IDs numerically if possible, otherwise as strings
        def seg_id_key(x):
            try:
                return int(x)
            except (ValueError, TypeError):
                return float('inf')  # non-integer IDs go last
            
        for seg_id in sorted(all_segment_ids, key=seg_id_key):
            # Filter segments for this ID
            pred_segs = [s for s in predicted_segments if s.get('segment_id') == seg_id]
            gt_segs = [s for s in ground_truth_segments if s.get('segment_id') == seg_id]
            
            result = {
                'segment_id': seg_id,
                'num_predicted': len(pred_segs),
                'num_ground_truth': len(gt_segs),
                'detected': False,
                'false_positive': False,
                'false_negative': False,
                'true_negative': False
            }
            
            # Check if we have both predicted and ground truth
            if pred_segs and gt_segs:
                # Match using start time threshold
                pred_seg = pred_segs[0]  # Take first if multiple
                gt_seg = gt_segs[0]
                
                start_diff = (pred_seg['start_time'] - gt_seg['start_time']).total_seconds()
                
                # Recognition is valid if it starts within threshold (can be late, but not too late)
                if 0 <= start_diff <= self.start_threshold:
                    result['detected'] = True
                    
                    # Calculate timing errors
                    start_error = abs(start_diff)
                    end_error = abs((pred_seg['end_time'] - gt_seg['end_time']).total_seconds())
                    duration_error = abs(pred_seg['duration'] - gt_seg['duration'])
                    
                    result['start_error'] = start_error
                    result['end_error'] = end_error
                    result['duration_error'] = duration_error
                    result['within_tolerance'] = (start_error <= self.time_tolerance and 
                                                 end_error <= self.time_tolerance)
                elif start_diff < 0:
                    # Predicted too early - still detected but with negative error
                    result['detected'] = True
                    result['start_error'] = abs(start_diff)
                    result['end_error'] = abs((pred_seg['end_time'] - gt_seg['end_time']).total_seconds())
                    result['duration_error'] = abs(pred_seg['duration'] - gt_seg['duration'])
                    result['within_tolerance'] = False
                    result['too_early'] = True
                else:
                    # Predicted too late (> threshold) - counts as miss
                    result['false_negative'] = True
                    result['detected'] = False
                    result['start_error'] = start_diff
                    result['too_late'] = True
                    result['end_error'] = None
                    result['duration_error'] = None
                    result['within_tolerance'] = False
            elif pred_segs and not gt_segs:
                result['false_positive'] = True
            elif not pred_segs and gt_segs:
                result['false_negative'] = True
            else:
                result['true_negative'] = True
            
            # Set None for missing timing data
            if 'start_error' not in result:
                result['start_error'] = None
            if 'end_error' not in result:
                result['end_error'] = None
            if 'duration_error' not in result:
                result['duration_error'] = None
            if 'within_tolerance' not in result:
                result['within_tolerance'] = None
            
            per_segment_results[seg_id] = result
        
        return per_segment_results
    
    def calculate_time_errors(self, 
                            predicted_segments: List[Dict], 
                            ground_truth_segments: List[Dict]) -> Dict:
        """
        Calculate temporal alignment errors between predicted and ground truth
        Matches segments if prediction starts within start_threshold of ground truth (both early and late)
        """
        
        time_errors = {
            'start_errors': [],
            'end_errors': [],
            'duration_errors': []
        }
        
        matched_pairs = []
        unmatched_predictions = []
        unmatched_ground_truth = list(ground_truth_segments)
        
        # Match segments by ID and start time threshold
        for pred in predicted_segments:
            best_match = None
            
            for gt in ground_truth_segments:
                # Check if same segment ID
                if pred.get('segment_id') == gt.get('segment_id'):
                    # Calculate start time difference (positive = late, negative = early)
                    start_diff = (pred['start_time'] - gt['start_time']).total_seconds()
                    
                    # Allow prediction to be within start_threshold seconds (both early and late)
                    if abs(start_diff) <= self.start_threshold:
                        best_match = gt
                        break
            
            if best_match:
                matched_pairs.append((pred, best_match))
                if best_match in unmatched_ground_truth:
                    unmatched_ground_truth.remove(best_match)
                
                # Calculate errors (absolute values)
                start_error = abs((pred['start_time'] - best_match['start_time']).total_seconds())
                end_error = abs((pred['end_time'] - best_match['end_time']).total_seconds())
                duration_error = abs(pred['duration'] - best_match['duration'])
                
                time_errors['start_errors'].append(start_error)
                time_errors['end_errors'].append(end_error)
                time_errors['duration_errors'].append(duration_error)
            else:
                unmatched_predictions.append(pred)
        
        # Calculate statistics
        results = {}
        for key in time_errors:
            if time_errors[key]:
                results[f'{key}_mean'] = np.mean(time_errors[key])
                results[f'{key}_std'] = np.std(time_errors[key])
                results[f'{key}_max'] = np.max(time_errors[key])
                results[f'{key}_median'] = np.median(time_errors[key])
            else:
                results[f'{key}_mean'] = None
                results[f'{key}_std'] = None
                results[f'{key}_max'] = None
                results[f'{key}_median'] = None
        
        return results
       
    def calculate_segment_metrics(self,
                                  predicted_segments: List[Dict],
                                  ground_truth_segments: List[Dict],
                                  confidence_scores: List[float] = None) -> Dict:
        """
        Calculate segment-based detection metrics including F1, precision, recall, TPR, FPR.
        Also calculates ROC and AUC if confidence scores are provided.
        """
        
        # Match segments and count TP, FP, FN
        matched_gt = set()
        matched_pred = set()
        
        true_positives = 0
        false_positives = 0
        
        # Match predicted segments to ground truth
        for i, pred in enumerate(predicted_segments):
            best_match = None
            best_match_idx = None
            
            for j, gt in enumerate(ground_truth_segments):
                if j in matched_gt:
                    continue
                    
                # Check if same segment ID
                if pred.get('segment_id') == gt.get('segment_id'):
                    # Calculate start time difference
                    start_diff = (pred['start_time'] - gt['start_time']).total_seconds()
                    
                    # Allow prediction within start_threshold
                    if abs(start_diff) <= self.start_threshold:
                        best_match = gt
                        best_match_idx = j
                        break
            
            if best_match is not None:
                true_positives += 1
                matched_gt.add(best_match_idx)
                matched_pred.add(i)
            else:
                false_positives += 1
        
        # Count false negatives (ground truth segments not matched)
        false_negatives = len(ground_truth_segments) - len(matched_gt)
        
        # True negatives: For segment-based evaluation, this would be segment IDs
        # that appear in neither prediction nor ground truth
        # For simplicity, we'll calculate it from the expected segment IDs (1-8)
        all_segment_ids = set([str(i) for i in range(1, 9)])
        pred_segment_ids = set([s['segment_id'] for s in predicted_segments])
        gt_segment_ids = set([s['segment_id'] for s in ground_truth_segments])
        
        # TN = segment IDs with no activity in both GT and predictions
        true_negatives = len(all_segment_ids - pred_segment_ids - gt_segment_ids)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        # TPR (True Positive Rate) = Recall = Sensitivity
        tpr = recall
        
        # FPR (False Positive Rate) = FP / (FP + TN)
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        
        results = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tpr': tpr,
            'fpr': fpr,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
        }
        
        # Calculate ROC and AUC only if confidence scores are provided
        if confidence_scores is not None and len(confidence_scores) > 0:
            # Only calculate if we have varied confidence scores
            if len(np.unique(confidence_scores)) > 1:
                try:
                    # Create binary labels for each prediction
                    y_true = np.array([1 if i in matched_pred else 0 for i in range(len(predicted_segments))])
                    y_scores = np.array(confidence_scores[:len(predicted_segments)])
                    
                    # Calculate ROC curve and AUC
                    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr_curve, tpr_curve)
                    
                    results['auc'] = roc_auc
                except Exception as e:
                    print(f"Warning: Could not calculate ROC/AUC: {e}")
                    results['auc'] = None
            else:
                results['auc'] = None
        else:
            results['auc'] = None
        
        return results
    
    def evaluate(self,
                predicted_file: str,
                ground_truth_file: str,
                confidence_scores: List[float] = None,
                per_segment: bool = False) -> Dict:
        """
        Complete evaluation comparing predicted vs ground truth
        
        Args:
            predicted_file: Path to predicted segments log
            ground_truth_file: Path to ground truth log
            confidence_scores: Optional confidence scores for each predicted segment
            per_segment: If True, include per-segment breakdown
            
        Returns:
            Dictionary with all evaluation metrics
        """
        
        # Parse files
        pred_df = self.parse_log_file(predicted_file)
        gt_df = self.parse_log_file(ground_truth_file)
        
        # Extract segments
        pred_segments = self.extract_segments(pred_df)
        gt_segments = self.extract_segments(gt_df)
        
        # Calculate all metrics
        segment_metrics = self.calculate_segment_metrics(
            pred_segments, gt_segments, confidence_scores
        )
        
        time_errors = self.calculate_time_errors(pred_segments, gt_segments)
        
        # Combine results
        results = {
            **segment_metrics,
            **time_errors
        }
        
        if per_segment:
            results['per_segment_metrics'] = self.calculate_per_segment_metrics(
                pred_segments, gt_segments
            )
        
        return results
    
    def evaluate_segments(self,
                         predicted_segments: List[Dict],
                         ground_truth_segments: List[Dict],
                         confidence_scores: List[float] = None,
                         per_segment: bool = False) -> Dict:
        """
        Evaluate segments directly without reading files
        
        Args:
            predicted_segments: List of dicts with 'start_time', 'end_time', 'segment_id'
            ground_truth_segments: Same format
            confidence_scores: Optional confidence for each predicted segment
            per_segment: If True, include per-segment breakdown
        """
        segment_metrics = self.calculate_segment_metrics(
            predicted_segments, ground_truth_segments, confidence_scores
        )
        
        time_errors = self.calculate_time_errors(predicted_segments, ground_truth_segments)
        
        results = {
            **segment_metrics,
            **time_errors
        }
        
        if per_segment:
            results['per_segment_metrics'] = self.calculate_per_segment_metrics(
                predicted_segments, ground_truth_segments
            )
        
        return results
    
    def print_report(self, results: Dict, show_per_segment: bool = True):
        """Print a formatted evaluation report"""
        print("=" * 60)
        print("SEGMENT EVALUATION REPORT")
        print("=" * 60)
        print(f"Start Time Threshold: {self.start_threshold}s")
        print(f"Accuracy Tolerance:   {self.time_tolerance}s")
        
        print("\n--- Detection Metrics (Segment-Based) ---")
        print(f"F1 Score:         {results['f1_score']:.4f}")
        print(f"Precision:        {results['precision']:.4f}")
        print(f"Recall (TPR):     {results['recall']:.4f}")
        print(f"FPR:              {results['fpr']:.4f}")
        auc_val = results.get('auc')
        if auc_val is not None:
            print(f"AUC:              {auc_val:.4f}")
        else:
            print(f"AUC:              N/A (no varied confidence scores)")
        
        print("\n--- Confusion Matrix (Segments) ---")
        print(f"True Positives:   {results['true_positives']} segments")
        print(f"False Positives:  {results['false_positives']} segments")
        print(f"False Negatives:  {results['false_negatives']} segments")
        print(f"True Negatives:   {results['true_negatives']} segment IDs")
        
        print("\n--- Temporal Accuracy ---")
        if results['start_errors_mean'] is not None:
            print(f"Start Time Error (mean):     {results['start_errors_mean']:.4f}s")
            print(f"Start Time Error (std):      {results['start_errors_std']:.4f}s")
            print(f"Start Time Error (max):      {results['start_errors_max']:.4f}s")
            print(f"End Time Error (mean):       {results['end_errors_mean']:.4f}s")
            print(f"End Time Error (std):        {results['end_errors_std']:.4f}s")
            print(f"Duration Error (mean):       {results['duration_errors_mean']:.4f}s")
        else:
            print("No matched segments for temporal analysis")
        
        print("\n--- Segment Counts ---")
        print(f"Predicted Segments:     {results['true_positives'] + results['false_positives']}")
        print(f"Ground Truth Segments:  {results['true_positives'] + results['false_negatives']}")
        print(f"Matched Segments:       {results['true_positives']}")
        print(f"Missed Segments:        {results['false_negatives']}")
        print(f"False Alarm Segments:   {results['false_positives']}")
        
        if 'per_segment_metrics' in results and show_per_segment:
            print("\n" + "=" * 60)
            print("PER-SEGMENT BREAKDOWN")
            print("=" * 60)
            
            per_seg = results['per_segment_metrics']
            for seg_id, metrics in sorted(per_seg.items()):
                print(f"\n--- Segment ID: {seg_id} ---")
                print(f"  Predicted:     {metrics['num_predicted']}")  # Changed from true_positives + false_positives
                print(f"  Ground Truth:  {metrics['num_ground_truth']}")  # Changed from true_positives + false_negatives
                print(f"  Status:        ", end="")
                
                if metrics['detected']:
                    if metrics.get('too_early'):
                        print("⚠ DETECTED (TOO EARLY)")
                    else:
                        print("✓ DETECTED")
                elif metrics['false_positive']:
                    print("✗ FALSE POSITIVE")
                elif metrics['false_negative']:
                    if metrics.get('too_late'):
                        print(f"✗ MISSED (>{self.start_threshold}s LATE)")
                    else:
                        print("✗ MISSED")
                else:
                    print("○ TRUE NEGATIVE")
                
                if metrics['start_error'] is not None:
                    print(f"  Start Error:   {metrics['start_error']:.4f}s")
                    print(f"  End Error:     {metrics['end_error']:.4f}s")
                    print(f"  Duration Err:  {metrics['duration_error']:.4f}s")
                    tolerance_str = "✓" if metrics['within_tolerance'] else "✗"
                    print(f"  Within Tol:    {tolerance_str}")
                
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    
    # Initialize evaluator with thresholds
    evaluator = SegmentEvaluator(
        time_tolerance_seconds=2.0,
        start_time_threshold_seconds=15.0
    )
    
    # Example 1: Evaluation without confidence scores
    print("EVALUATION WITHOUT CONFIDENCE SCORES")
    results = evaluator.evaluate(
        predicted_file='data_modified.txt',
        ground_truth_file='data_original.txt',
        per_segment=True
    )
    evaluator.print_report(results)
    
    print("\n\n")
    
    # Example 2: Evaluation with confidence scores
    print("EVALUATION WITH CONFIDENCE SCORES")
    
    # Confidence scores for each predicted segment
    confidence_scores = [0.95, 0.87, 0.92, 0.78, 0.88, 0.91, 0.85, 0.93]
    
    results_with_conf = evaluator.evaluate(
        predicted_file='svm_predictions.txt',
        ground_truth_file='data_original.txt',
        confidence_scores=confidence_scores,
        per_segment=True
    )
    evaluator.print_report(results_with_conf)