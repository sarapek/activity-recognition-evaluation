import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from datetime import datetime
import pandas as pd
from typing import List, Tuple, Dict

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
        """Parse the log file into a structured DataFrame"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f')
                    event_id = parts[1]
                    state = parts[2]
                    annotation = parts[3] if len(parts) > 3 else ''
                    errors = parts[4] if len(parts) > 4 else ''
                    
                    data.append({
                        'timestamp': timestamp,
                        'event_id': event_id,
                        'state': state,
                        'annotation': annotation,
                        'errors': errors
                    })
        
        return pd.DataFrame(data)
    
    def extract_segments(self, df: pd.DataFrame) -> List[Dict]:
        """Extract segments from annotations (e.g., '1-start', '1-end')"""
        segments = []
        segment_starts = {}
        
        for idx, row in df.iterrows():
            annotation = str(row['annotation']).strip()
            
            # Skip empty annotations
            if not annotation or annotation == 'nan':
                continue
            
            # Split by comma in case of multiple annotations like "1-start,1.1"
            annotation_parts = [a.strip() for a in annotation.split(',')]
            
            for part in annotation_parts:
                # Check for segment start markers
                if '-start' in part.lower():
                    seg_id = part.lower().split('-start')[0].strip()
                    segment_starts[seg_id] = row['timestamp']
                
                # Check for segment end markers
                elif '-end' in part.lower():
                    seg_id = part.lower().split('-end')[0].strip()
                    
                    if seg_id in segment_starts:
                        segments.append({
                            'segment_id': seg_id,
                            'start_time': segment_starts[seg_id],
                            'end_time': row['timestamp'],
                            'duration': (row['timestamp'] - segment_starts[seg_id]).total_seconds()
                        })
                        del segment_starts[seg_id]
        
        # Check for unclosed segments
        if segment_starts:
            print(f"WARNING: Unclosed segments: {list(segment_starts.keys())}")
        
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
            elif pred_segs and not gt_segs:
                result['false_positive'] = True
            elif not pred_segs and gt_segs:
                result['false_negative'] = True
            else:
                result['true_negative'] = True
            
            # Set None for missing timing data
            if 'start_error' not in result:
                result['start_error'] = None
                result['end_error'] = None
                result['duration_error'] = None
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
            'duration_errors': [],
            'detection_delays': [],  # How late was the detection (only positive)
            'detection_early': []    # How early was the detection (only positive)
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
                
                # Calculate errors
                start_diff = (pred['start_time'] - best_match['start_time']).total_seconds()
                start_error = abs(start_diff)
                end_error = abs((pred['end_time'] - best_match['end_time']).total_seconds())
                duration_error = abs(pred['duration'] - best_match['duration'])
                
                time_errors['start_errors'].append(start_error)
                time_errors['end_errors'].append(end_error)
                time_errors['duration_errors'].append(duration_error)
                time_errors['detection_delays'].append(max(0, start_diff))  # Only positive delays
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
        
        results['num_matched'] = len(matched_pairs)
        results['num_predicted'] = len(predicted_segments)
        results['num_ground_truth'] = len(ground_truth_segments)
        results['num_missed'] = len(unmatched_ground_truth)
        results['num_false_alarms'] = len(unmatched_predictions)
        
        return results
       
    def calculate_segment_metrics(self,
                                  predicted_segments: List[Dict],
                                  ground_truth_segments: List[Dict],
                                  confidence_scores: List[float] = None) -> Dict:
        """Calculate ROC, AUC, F1 and other metrics for segment detection"""
        
        # Create binary labels for each time window
        # We'll discretize time into windows and mark presence/absence
        all_times = []
        for seg in predicted_segments + ground_truth_segments:
            all_times.extend([seg['start_time'], seg['end_time']])
        
        if not all_times:
            return {
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc': None,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'true_negatives': 0
            }
        
        min_time = min(all_times)
        max_time = max(all_times)
        
        # Create time windows (1 second intervals)
        time_range = (max_time - min_time).total_seconds()
        num_windows = int(time_range) + 1
        
        y_true = np.zeros(num_windows)
        y_pred = np.zeros(num_windows)
        y_scores = np.zeros(num_windows)
        
        # Mark ground truth segments
        for seg in ground_truth_segments:
            start_idx = int((seg['start_time'] - min_time).total_seconds())
            end_idx = int((seg['end_time'] - min_time).total_seconds())
            y_true[start_idx:end_idx+1] = 1
        
        # Mark predicted segments
        for i, seg in enumerate(predicted_segments):
            start_idx = int((seg['start_time'] - min_time).total_seconds())
            end_idx = int((seg['end_time'] - min_time).total_seconds())
            y_pred[start_idx:end_idx+1] = 1
            
            # Use confidence scores if provided
            if confidence_scores and i < len(confidence_scores):
                y_scores[start_idx:end_idx+1] = confidence_scores[i]
            else:
                y_scores[start_idx:end_idx+1] = 1.0
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Only calculate AUC if there are varied confidence scores
        if len(np.unique(y_scores[y_pred == 1])) > 1:
            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
            except:
                fpr, tpr = None, None
                roc_auc = None
        else:
            roc_auc = None
            fpr, tpr = None, None
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': roc_auc,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'fpr': fpr,
            'tpr': tpr
        }
    
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
        
        print("\n--- Detection Metrics ---")
        print(f"F1 Score:         {results['f1_score']:.4f}")
        print(f"Precision:        {results['precision']:.4f}")
        print(f"Recall:           {results['recall']:.4f}")
        auc_val = results['auc']
        if auc_val is not None:
            print(f"AUC:              {auc_val:.4f}")
        else:
            print(f"AUC:              N/A (no confidence scores)")
        
        print("\n--- Confusion Matrix ---")
        print(f"True Positives:   {results['true_positives']}")
        print(f"False Positives:  {results['false_positives']}")
        print(f"False Negatives:  {results['false_negatives']}")
        print(f"True Negatives:   {results['true_negatives']}")
        
        print("\n--- Temporal Accuracy ---")
        if results['start_errors_mean'] is not None:
            print(f"Start Time Error (mean):     {results['start_errors_mean']:.4f}s")
            print(f"Start Time Error (std):      {results['start_errors_std']:.4f}s")
            print(f"Start Time Error (max):      {results['start_errors_max']:.4f}s")
            print(f"End Time Error (mean):       {results['end_errors_mean']:.4f}s")
            print(f"End Time Error (std):        {results['end_errors_std']:.4f}s")
            print(f"Duration Error (mean):       {results['duration_errors_mean']:.4f}s")
            print(f"Detection Delay (mean):      {results['detection_delays_mean']:.4f}s")
            print(f"Detection Delay (max):       {results['detection_delays_max']:.4f}s")
        else:
            print("No matched segments for temporal analysis")
        
        print("\n--- Segment Counts ---")
        print(f"Predicted Segments:     {results['num_predicted']}")
        print(f"Ground Truth Segments:  {results['num_ground_truth']}")
        print(f"Matched Segments:       {results['num_matched']}")
        print(f"Missed (Too Late):      {results['num_missed']}")
        print(f"False Alarms:           {results['num_false_alarms']}")
        
        if 'per_segment_metrics' in results and show_per_segment:
            print("\n" + "=" * 60)
            print("PER-SEGMENT BREAKDOWN")
            print("=" * 60)
            
            per_seg = results['per_segment_metrics']
            for seg_id, metrics in sorted(per_seg.items()):
                print(f"\n--- Segment ID: {seg_id} ---")
                print(f"  Predicted:     {metrics['num_predicted']}")
                print(f"  Ground Truth:  {metrics['num_ground_truth']}")
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
    # start_time_threshold_seconds: maximum delay for detection (15s)
    # time_tolerance_seconds: threshold for "accurate" timing (2s)
    evaluator = SegmentEvaluator(
        time_tolerance_seconds=2.0,
        start_time_threshold_seconds=15.0
    )
    
    # Example 1: Sanity check with randomly modified file
    print("SANITY CHECK EVALUATION")
    results_sanity = evaluator.evaluate(
        predicted_file='data_modified.txt',
        ground_truth_file='data_original.txt',
        per_segment=True
    )
    evaluator.print_report(results_sanity)
    
    print("\n\n")
    
    # Example 2: SVM-based recognition with confidence scores
    print("SVM RECOGNITION EVALUATION")
    
    # These would come from your SVM classifier
    svm_confidence_scores = [0.95, 0.87, 0.92, 0.78, 0.88]
    
    results_svm = evaluator.evaluate(
        predicted_file='svm_predictions.txt',
        ground_truth_file='data_original.txt',
        confidence_scores=svm_confidence_scores,
        per_segment=True
    )
    evaluator.print_report(results_svm)