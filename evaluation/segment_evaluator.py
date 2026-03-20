import numpy as np
from sklearn.metrics import roc_curve, auc
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple, Dict
import re
import os
from itertools import cycle

class SegmentEvaluator:
    """Evaluates segment-based recognition with both frame-level and segment-level ROC"""

    # Activity name mapping
    @staticmethod
    def get_activity_names():
        """Get activity names in Slovene"""
        return {
            '1': 'Pometanje tal',
            '2': 'Jemanje zdravil',
            '3': 'Igre s kartami',
            '4': 'Predvajanje DVD-ja',
            '5': 'Zalivanje rož',
            '6': 'Telefonski klic',
            '7': 'Kuhanje',
            '8': 'Izbira oblačil'
        }

    # Fixed color mapping for activities 1-8 (Option B - balanced & distinct)
    @staticmethod
    def get_activity_colors():
        """Get default matplotlib colors for ROC curves"""
        return {
            '1': '#1f77b4',  # Blue
            '2': '#ff7f0e',  # Orange
            '3': '#2ca02c',  # Green
            '4': '#d62728',  # Red
            '5': '#9467bd',  # Purple
            '6': '#8c564b',  # Brown
            '7': '#e377c2',  # Pink
            '8': '#7f7f7f'   # Gray
        }

    def __init__(self, timeline_resolution_seconds=1.0, start_time_threshold_seconds=30, overlap_threshold=0.5, confidence_threshold=0.0, max_gap_seconds=60.0):
        """
        Args:
            timeline_resolution_seconds: Time resolution for timeline (default 1.0 = per second)
            start_time_threshold_seconds: Maximum allowed start time difference in seconds (default 30).
                Predicted segments starting more than this many seconds before/after GT start
                are REJECTED and not matched (even if tIoU is high enough).
                Set to 0 to disable filtering (only use tIoU for matching).
            overlap_threshold: Minimum temporal IoU (tIoU) for segment matching (default 0.5)
                Uses standard Intersection over Union: tIoU = intersection / union
                where union = GT_duration + Pred_duration - intersection
                For a match, BOTH tIoU >= threshold AND start_diff <= start_time_threshold must be met.
            confidence_threshold: Minimum confidence for accepting predictions (default 0.0 = accept all)
                Predictions below this threshold are treated as 'Other_Activity'
            max_gap_seconds: Maximum time gap between events before splitting segment (default 60.0)
                If gap between consecutive events > this threshold, treat as separate segments.
                Applied to both GT and predictions. Set to large value (e.g., 99999) to disable.
        """
        self.resolution = timeline_resolution_seconds
        self.start_time_threshold = start_time_threshold_seconds
        self.overlap_threshold = overlap_threshold
        self.confidence_threshold = confidence_threshold
        self.max_gap_seconds = max_gap_seconds
        
    def parse_log_file(self, filepath: str) -> pd.DataFrame:
        """Parse the log file into a structured DataFrame (space-separated format)"""
        data = []
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                try:
                    if len(parts) >= 7:
                        # AL format WITH confidence: date time sensor newsensor status activity confidence
                        date_str = parts[0]
                        time_str = parts[1]
                        timestamp_str = f"{date_str} {time_str}"
                        
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        sensor_id = parts[2]
                        status = parts[4]
                        activity = parts[5]
                        confidence = float(parts[6])
                        
                        data.append({
                            'timestamp': timestamp,
                            'event_id': sensor_id,
                            'state': status,
                            'annotation': activity,
                            'confidence': confidence,
                            'errors': ''
                        })
                        
                    elif len(parts) == 6:
                        date_str = parts[0]
                        time_str = parts[1]
                        timestamp_str = f"{date_str} {time_str}"
                        
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        sensor_id = parts[2]
                        
                        # Check if parts[5] contains 'error' - if so, it's GT format
                        if 'error' in parts[5].lower():
                            # Ground truth format: date time sensor status activity errors
                            status = parts[3]
                            activity = parts[4]
                            
                            data.append({
                                'timestamp': timestamp,
                                'event_id': sensor_id,
                                'state': status,
                                'annotation': activity,
                                'errors': parts[5]
                            })
                        else:
                            # Prediction format: date time sensor newsensor status activity
                            status = parts[4]
                            activity = parts[5]
                            
                            data.append({
                                'timestamp': timestamp,
                                'event_id': sensor_id,
                                'state': status,
                                'annotation': activity,
                                'errors': ''
                            })
                        
                    elif len(parts) >= 5:
                        # Ground truth format: date time sensor status activity
                        date_str = parts[0]
                        time_str = parts[1]
                        timestamp_str = f"{date_str} {time_str}"
                        
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        sensor_id = parts[2]
                        status = parts[3]
                        activity = parts[4]
                        
                        data.append({
                            'timestamp': timestamp,
                            'event_id': sensor_id,
                            'state': status,
                            'annotation': activity,
                            'errors': ''
                        })
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(data)

    def extract_segments(self, df: pd.DataFrame, confidence_threshold: float = 0.0, max_gap_seconds: float = 60.0) -> List[Dict]:
        """
        Extract segments from annotations

        Args:
            df: DataFrame with annotations
            confidence_threshold: Minimum confidence to accept predictions
            max_gap_seconds: Maximum time gap between events before splitting segment (default 60s)
                           If gap between consecutive events > this threshold, treat as separate segments
        """
        segments = []
        current_activity = None
        segment_start = None
        prev_timestamp = None
        segment_confidences = []
        
        if df is None or len(df) == 0:
            return segments
        
        # ADD THIS CHECK TOO
        if 'annotation' not in df.columns:
            print(f"Warning: 'annotation' column not found in DataFrame. Columns: {df.columns.tolist()}")
            return segments
                
        for idx, row in df.iterrows():
            annotation = str(row['annotation']).strip()
            timestamp = row['timestamp']
            
            confidence = row.get('confidence', 1.0) if 'confidence' in df.columns else 1.0
            
            if confidence < confidence_threshold:
                annotation = 'Other_Activity'
            
            # Skip empty annotations or Other_Activity
            if not annotation or annotation == 'nan' or annotation == 'Other_Activity':
                prev_timestamp = timestamp
                continue
            
            # Handle comma-separated annotations (e.g., "1.4,1-end")
            annotations = [a.strip() for a in annotation.split(',')]
            
            for ann in annotations:
                if '-end' in ann.lower():
                    activity_id = ann.lower().split('-end')[0].strip()
                    
                    if current_activity == activity_id and segment_start is not None:
                        segment_confidences.append(confidence)
                        first_confidence = segment_confidences[0] if segment_confidences else 1.0
                        segments.append({
                            'segment_id': current_activity,
                            'start_time': segment_start,
                            'end_time': timestamp,
                            'duration': (timestamp - segment_start).total_seconds(),
                            'confidence': first_confidence
                        })
                        current_activity = None
                        segment_start = None
                        segment_confidences = []
                
                elif '-start' in ann.lower():
                    activity_id = ann.lower().split('-start')[0].strip()
                    
                    if current_activity is not None and segment_start is not None:
                        first_confidence = segment_confidences[0] if segment_confidences else 1.0
                        segments.append({
                            'segment_id': current_activity,
                            'start_time': segment_start,
                            'end_time': timestamp,
                            'duration': (timestamp - segment_start).total_seconds(),
                            'confidence': first_confidence
                        })
                    
                    current_activity = activity_id
                    segment_start = timestamp
                    segment_confidences = [confidence]
                
                else:
                    # Continuous activity label
                    match = re.match(r'^(\d+)', ann)
                    if match:
                        activity_id = match.group(1)

                        if activity_id != current_activity:
                            # Activity changed - close previous segment
                            if current_activity is not None and segment_start is not None and prev_timestamp is not None:
                                first_confidence = segment_confidences[0] if segment_confidences else 1.0
                                segments.append({
                                    'segment_id': current_activity,
                                    'start_time': segment_start,
                                    'end_time': prev_timestamp,
                                    'duration': (prev_timestamp - segment_start).total_seconds(),
                                    'confidence': first_confidence
                                })
                            current_activity = activity_id
                            segment_start = timestamp
                            segment_confidences = [confidence]
                        else:
                            # Same activity - check for large gap
                            if prev_timestamp is not None and segment_start is not None:
                                gap = (timestamp - prev_timestamp).total_seconds()
                                if gap > max_gap_seconds:
                                    # Gap too large - close current segment and start new one
                                    first_confidence = segment_confidences[0] if segment_confidences else 1.0
                                    segments.append({
                                        'segment_id': current_activity,
                                        'start_time': segment_start,
                                        'end_time': prev_timestamp,
                                        'duration': (prev_timestamp - segment_start).total_seconds(),
                                        'confidence': first_confidence
                                    })
                                    # Start new segment for same activity
                                    segment_start = timestamp
                                    segment_confidences = [confidence]
                                else:
                                    # Continue current segment
                                    segment_confidences.append(confidence)
                            else:
                                segment_confidences.append(confidence)
            
            prev_timestamp = timestamp
        
        # Close any remaining open segment
        if current_activity is not None and segment_start is not None and prev_timestamp is not None:
            first_confidence = segment_confidences[0] if segment_confidences else 1.0
            segments.append({
                'segment_id': current_activity,
                'start_time': segment_start,
                'end_time': prev_timestamp,
                'duration': (prev_timestamp - segment_start).total_seconds(),
                'confidence': first_confidence
            })
        
        return segments

    def _compute_overlap(self, gt_seg, pred_seg):
        """
        Compute overlap duration between a GT and predicted segment.
        No adjustment - raw overlap calculation.

        Returns:
            overlap_duration in seconds (0.0 if no overlap)
        """
        overlap_start = max(gt_seg['start_time'], pred_seg['start_time'])
        overlap_end = min(gt_seg['end_time'], pred_seg['end_time'])

        if overlap_start < overlap_end:
            return (overlap_end - overlap_start).total_seconds()
        return 0.0

    def _is_within_start_threshold(self, gt_seg, pred_seg):
        """
        Check if predicted segment start is within start_time_threshold of GT start.

        Returns:
            True if within threshold (or threshold is 0), False otherwise
        """
        if self.start_time_threshold <= 0:
            return True  # No filtering

        start_diff = abs((pred_seg['start_time'] - gt_seg['start_time']).total_seconds())
        return start_diff <= self.start_time_threshold

    def _adjust_segments_for_tolerance(self, pred_segments, gt_segments):
        """
        DISABLED: Previously adjusted predicted segment start times based on start_time_threshold.
        Now returns segments unchanged - no adjustment performed.

        Used for timeline-based metrics (time-continuous precision/recall/F1).
        """
        # No adjustment - return segments as-is
        return pred_segments

        # Old adjustment code disabled:
        if self.start_time_threshold <= 0:
            return pred_segments

        adjusted = []
        for pred_seg in pred_segments:
            best_gt = None
            best_overlap = 0

            for gt_seg in gt_segments:
                if gt_seg['segment_id'] != pred_seg['segment_id']:
                    continue
                overlap_duration = self._compute_overlap(gt_seg, pred_seg)
                if overlap_duration > best_overlap:
                    best_gt = gt_seg
                    best_overlap = overlap_duration

            if best_gt is not None:
                start_diff = abs((pred_seg['start_time'] - best_gt['start_time']).total_seconds())
                if start_diff > 0 and start_diff <= self.start_time_threshold:
                    new_start = best_gt['start_time']
                    if new_start < pred_seg['end_time']:
                        new_seg = pred_seg.copy()
                        new_seg['start_time'] = new_start
                        new_seg['duration'] = (new_seg['end_time'] - new_seg['start_time']).total_seconds()
                        adjusted.append(new_seg)
                    else:
                        adjusted.append(pred_seg)
                else:
                    adjusted.append(pred_seg)
            else:
                adjusted.append(pred_seg)

        return adjusted

    def create_timeline(self, segments: List[Dict], start_time: datetime, end_time: datetime) -> Tuple[np.ndarray, List[str]]:
        """
        Create a continuous timeline representation of segments
        
        Args:
            segments: List of segment dictionaries
            start_time: Start of timeline
            end_time: End of timeline
            
        Returns:
            timeline: Array where timeline[t] = activity_id at time t (or empty string)
            activity_classes: List of all unique activity IDs found
        """
        total_seconds = (end_time - start_time).total_seconds()
        timeline_length = int(total_seconds / self.resolution)
        
        # Initialize timeline (empty string = no activity)
        timeline = np.array([''] * timeline_length, dtype=object)
        
        # Get all unique activity classes
        activity_classes = sorted(list(set(seg['segment_id'] for seg in segments)))
        
        # Fill timeline with activities
        for seg in segments:
            start_idx = int((seg['start_time'] - start_time).total_seconds() / self.resolution)
            end_idx = int((seg['end_time'] - start_time).total_seconds() / self.resolution)
            
            # Clip to valid range
            start_idx = max(0, min(start_idx, timeline_length))
            end_idx = max(0, min(end_idx, timeline_length))
            
            timeline[start_idx:end_idx] = seg['segment_id']
        
        return timeline, activity_classes
    
    def calculate_time_continuous_metrics(self,
                                          pred_timeline: np.ndarray,
                                          gt_timeline: np.ndarray,
                                          activity_classes: List[str]) -> Dict:
        """
        Calculate multi-class time-continuous metrics

        For each activity class c:
            TP_c = duration where both predict class c
            FP_c = duration where prediction is c but GT is not c
            FN_c = duration where GT is c but prediction is not c
            TN_c = duration where neither GT nor prediction is class c
        """
        per_class_metrics = {}

        # Calculate metrics for each activity class
        for activity_id in activity_classes:
            # Binary masks for this class
            pred_mask = (pred_timeline == activity_id)
            gt_mask = (gt_timeline == activity_id)

            # Calculate durations (in resolution units, convert to seconds)
            TP_duration = np.sum(pred_mask & gt_mask) * self.resolution
            FP_duration = np.sum(pred_mask & ~gt_mask) * self.resolution
            FN_duration = np.sum(~pred_mask & gt_mask) * self.resolution
            TN_duration = np.sum(~pred_mask & ~gt_mask) * self.resolution

            # Calculate metrics
            precision = TP_duration / (TP_duration + FP_duration) if (TP_duration + FP_duration) > 0 else 0.0
            recall = TP_duration / (TP_duration + FN_duration) if (TP_duration + FN_duration) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # IoU (Intersection over Union)
            iou = TP_duration / (TP_duration + FP_duration + FN_duration) if (TP_duration + FP_duration + FN_duration) > 0 else 0.0

            # Per-activity accuracy: (TP + TN) / (TP + TN + FP + FN)
            total_duration = TP_duration + TN_duration + FP_duration + FN_duration
            accuracy = (TP_duration + TN_duration) / total_duration if total_duration > 0 else 0.0

            per_class_metrics[activity_id] = {
                'TP_duration': TP_duration,
                'FP_duration': FP_duration,
                'FN_duration': FN_duration,
                'TN_duration': TN_duration,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'iou': iou,
                'accuracy': accuracy,
                'gt_total_duration': np.sum(gt_mask) * self.resolution,
                'pred_total_duration': np.sum(pred_mask) * self.resolution
            }
        
        # Calculate aggregate metrics

        # Macro-average
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
        macro_iou = np.mean([m['iou'] for m in per_class_metrics.values()])

        # Micro-average
        total_TP = sum(m['TP_duration'] for m in per_class_metrics.values())
        total_FP = sum(m['FP_duration'] for m in per_class_metrics.values())
        total_FN = sum(m['FN_duration'] for m in per_class_metrics.values())

        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        micro_iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0.0

        # Calculate overall accuracy (proportion of time correctly predicted)
        total_timeline_length = len(pred_timeline)
        correct_predictions = np.sum(pred_timeline == gt_timeline)
        accuracy = correct_predictions / total_timeline_length if total_timeline_length > 0 else 0.0

        # Weighted-average
        total_gt_duration = sum(m['gt_total_duration'] for m in per_class_metrics.values())
        if total_gt_duration > 0:
            weighted_precision = sum(m['precision'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
            weighted_recall = sum(m['recall'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
            weighted_f1 = sum(m['f1_score'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
            weighted_iou = sum(m['iou'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
        else:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0
            weighted_iou = 0.0
        
        return {
            'per_class': per_class_metrics,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1,
                'iou': macro_iou
            },
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1,
                'iou': micro_iou,
                'accuracy': accuracy,
                'total_TP_duration': total_TP,
                'total_FP_duration': total_FP,
                'total_FN_duration': total_FN
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1,
                'iou': weighted_iou
            }
        }
    
    def calculate_segment_roc_auc_per_class(self, pred_segments, gt_segments, overlap_threshold=None):
        """Calculate segment-level ROC and AUC for each activity class"""
        
        if overlap_threshold is None:
            overlap_threshold = self.overlap_threshold
        
        activity_classes = set()
        for seg in pred_segments + gt_segments:
            activity_classes.add(seg['segment_id'])
        activity_classes = sorted(list(activity_classes))
            
        roc_auc_results = {}
            
        for activity_id in activity_classes:
            gt_activity_segments = [seg for seg in gt_segments if seg['segment_id'] == activity_id]
            pred_activity_segments = [seg for seg in pred_segments if seg['segment_id'] == activity_id]
            
            if len(gt_activity_segments) == 0:
                roc_auc_results[activity_id] = {'auc': None}
                continue
            
            # Temporal overlap check
            print(f"\n{'='*80}")
            print(f"CHECKING TEMPORAL OVERLAP FOR ACTIVITY {activity_id}")
            print(f"{'='*80}")

            if len(pred_activity_segments) > 0:
                # Get time ranges
                gt_times = [(seg['start_time'], seg['end_time']) for seg in gt_activity_segments]
                pred_times = [(seg['start_time'], seg['end_time']) for seg in pred_activity_segments]
                
                gt_min = min(t[0] for t in gt_times)
                gt_max = max(t[1] for t in gt_times)
                pred_min = min(t[0] for t in pred_times)
                pred_max = max(t[1] for t in pred_times)
                
                print(f"GT time span:   {gt_min} to {gt_max}")
                print(f"Pred time span: {pred_min} to {pred_max}")
                print(f"Days match: {gt_min.date() == pred_min.date()}")
                
                # Check if ANY temporal overlap exists
                global_overlap = not (gt_max < pred_min or pred_max < gt_min)
                print(f"Global temporal overlap exists: {global_overlap}")
                
                if not global_overlap:
                    print(f"WARNING: Predictions and GT are in completely different time periods!")
            
            print(f"\n{'='*80}")
            print(f"ACTIVITY {activity_id} - OVERLAP THRESHOLD = {overlap_threshold}")
            if self.start_time_threshold > 0:
                print(f"  Start time tolerance: {self.start_time_threshold}s")
            print(f"{'='*80}")
            
            y_true = []
            y_scores = []
            matched_pred_indices = set()
            
            # Step 1: For each GT segment, find best matching prediction
            for gt_idx, gt_seg in enumerate(gt_activity_segments):
                best_confidence = 0.0
                best_overlap_ratio = 0.0
                best_pred_idx = None
                
                gt_duration = gt_seg['duration']
                
                print(f"\n--- GT Segment {gt_idx} ---")
                print(f"  GT: {gt_seg['start_time'].strftime('%H:%M:%S')} -> {gt_seg['end_time'].strftime('%H:%M:%S')} ({gt_duration:.1f}s)")
                
                # Track all overlaps for this GT segment
                overlaps_found = []
                
                for pred_idx, pred_seg in enumerate(pred_activity_segments):
                    overlap_duration = self._compute_overlap(gt_seg, pred_seg)

                    if overlap_duration > 0:
                        pred_duration = pred_seg['duration']

                        if gt_duration is not None and gt_duration > 0:
                            # Calculate tIoU (temporal Intersection over Union)
                            # tIoU = intersection / union where union = GT + Pred - intersection
                            union_duration = gt_duration + pred_duration - overlap_duration
                            overlap_ratio = overlap_duration / union_duration if union_duration > 0 else 0

                            confidence = pred_seg.get('confidence', 0.5)
                            if confidence is None:
                                confidence = 0.5

                            # Check if start time is within threshold
                            within_start_threshold = self._is_within_start_threshold(gt_seg, pred_seg)
                            start_diff = abs((pred_seg['start_time'] - gt_seg['start_time']).total_seconds())

                            # Store this overlap for debugging
                            overlaps_found.append({
                                'pred_idx': pred_idx,
                                'pred_start': pred_seg['start_time'],
                                'pred_end': pred_seg['end_time'],
                                'pred_duration': pred_duration,
                                'overlap_duration': overlap_duration,
                                'overlap_ratio': overlap_ratio,
                                'confidence': confidence,
                                'start_diff': start_diff,
                                'within_start_threshold': within_start_threshold,
                                'meets_threshold': overlap_ratio >= overlap_threshold and within_start_threshold
                            })

                            # Match requires BOTH tIoU >= threshold AND start within threshold
                            if overlap_ratio >= overlap_threshold and within_start_threshold:
                                if overlap_ratio > best_overlap_ratio or \
                                (overlap_ratio == best_overlap_ratio and confidence > best_confidence):
                                    best_confidence = confidence
                                    best_overlap_ratio = overlap_ratio
                                    best_pred_idx = pred_idx
                
                # Print all overlaps for this GT segment
                if len(overlaps_found) > 0:
                    print(f"  Found {len(overlaps_found)} overlapping predictions:")
                    for i, ov in enumerate(overlaps_found[:5]):  # Show first 5
                        meets = "[MATCH]" if ov['meets_threshold'] else "[REJECTED]"
                        start_status = f"start_diff={ov['start_diff']:.1f}s"
                        if not ov['within_start_threshold']:
                            start_status += f" [>threshold={self.start_time_threshold}s]"
                        print(f"    Pred {ov['pred_idx']}: {ov['pred_start'].strftime('%H:%M:%S')}->{ov['pred_end'].strftime('%H:%M:%S')} "
                            f"tIoU={ov['overlap_ratio']:.4f} {start_status} conf={ov['confidence']:.3f} {meets}")
                    if len(overlaps_found) > 5:
                        print(f"    ... and {len(overlaps_found) - 5} more")
                else:
                    print(f"  No overlapping predictions found")

                # Record result
                if best_overlap_ratio >= overlap_threshold:
                    print(f"  -> BEST MATCH: Pred {best_pred_idx}, tIoU={best_overlap_ratio:.4f}, conf={best_confidence:.3f} [TP]")
                    y_true.append(1)
                    y_scores.append(best_confidence)
                    matched_pred_indices.add(best_pred_idx)
                else:
                    # No match found - determine why
                    if len(overlaps_found) == 0:
                        print(f"  -> NO MATCH: No overlapping predictions [FN]")
                    else:
                        best_ov = max(overlaps_found, key=lambda x: x['overlap_ratio'])
                        reasons = []
                        if best_ov['overlap_ratio'] < overlap_threshold:
                            reasons.append(f"tIoU={best_ov['overlap_ratio']:.4f}<{overlap_threshold}")
                        if not best_ov['within_start_threshold']:
                            reasons.append(f"start_diff={best_ov['start_diff']:.1f}s>{self.start_time_threshold}s")
                        print(f"  -> NO MATCH: {', '.join(reasons)} [FN]")
                    # FIX: False negative means GT segment (positive sample) was not detected
                    # y_true=1 because it exists in GT (is a positive)
                    # y_score=0.0 because we didn't detect it (gave it lowest score)
                    y_true.append(1)
                    y_scores.append(0.0)
            
            # Step 2: Add FALSE POSITIVES
            unmatched_count = 0
            for pred_idx, pred_seg in enumerate(pred_activity_segments):
                if pred_idx not in matched_pred_indices:
                    y_true.append(0)
                    confidence = pred_seg.get('confidence', 0.5)
                    if confidence is None:
                        confidence = 0.5
                    y_scores.append(confidence)
                    unmatched_count += 1
            
            print(f"\n--- Summary for Activity {activity_id} ---")
            print(f"  GT segments: {len(gt_activity_segments)}")
            print(f"  Pred segments: {len(pred_activity_segments)}")
            print(f"  Matched predictions: {len(matched_pred_indices)}")
            print(f"  Unmatched predictions (FPs): {unmatched_count}")
            print(f"  Overlap threshold: {overlap_threshold}")
            
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)

            print(f"  Total data points: {len(y_true)}")
            print(f"  TPs: {sum(y_true)}")
            print(f"  FPs+FNs: {len(y_true) - sum(y_true)}")
            print(f"  Unique scores: {len(np.unique(y_scores))}")

            # Handle edge cases for perfect/worst classifiers
            num_positives = sum(y_true)
            num_samples = len(y_true)

            # Edge case 1: All predictions are perfect (all TPs, no FPs)
            # This happens when y_true is all 1s AND all scores > 0
            # If scores are all 0, it means all GT were missed (all FNs)
            if num_positives == num_samples and num_samples > 0:
                # Check if actually detected or all missed
                if np.all(y_scores == 0):
                    # All GT segments were missed (all FNs)
                    print(f"  -> All GT missed (all FN): AUC = 0.0000")
                    roc_auc_results[activity_id] = {
                        'auc': 0.0,
                        'fpr': np.array([0.0, 1.0]),
                        'tpr': np.array([0.0, 0.0]),
                        'thresholds': np.array([1.0, 0.0])
                    }
                else:
                    # Perfect classifier - all detected
                    print(f"  -> Perfect classifier: AUC = 1.0000")
                    roc_auc_results[activity_id] = {
                        'auc': 1.0,
                        'fpr': np.array([0.0, 0.0]),
                        'tpr': np.array([0.0, 1.0]),
                        'thresholds': np.array([2.0, 1.0])  # Dummy thresholds
                    }
            # Edge case 2: All predictions are wrong (all FNs and/or all FPs wrong)
            # num_positives > 0 means we have GT positives (possibly missed)
            # If all were missed AND we have FPs, AUC should be 0
            elif len(np.unique(y_true)) < 2:
                # Not enough variation in ground truth (all same class)
                print(f"  -> Not enough variation in ground truth for AUC")
                roc_auc_results[activity_id] = {'auc': None}
            # Normal case: have both positive and negative samples
            elif len(np.unique(y_scores)) > 1:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    print(f"  -> AUC: {roc_auc:.4f}")

                    roc_auc_results[activity_id] = {
                        'auc': roc_auc,
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds
                    }
                except Exception as e:
                    print(f"  Warning: Could not calculate ROC/AUC: {e}")
                    roc_auc_results[activity_id] = {'auc': None}
            else:
                # All scores are the same but have both positive/negative samples
                # Can't calculate meaningful ROC without score variation
                print(f"  -> All scores identical, cannot calculate ROC")
                roc_auc_results[activity_id] = {'auc': None}

            # Calculate segment-level precision/recall/F1 based on segment counts
            num_gt_segments = len(gt_activity_segments)
            num_pred_segments = len(pred_activity_segments)
            num_matched = len(matched_pred_indices)

            # Segment-level metrics
            seg_tp = num_matched  # True positives: matched segments
            seg_fp = num_pred_segments - num_matched  # False positives: unmatched predictions
            seg_fn = num_gt_segments - num_matched  # False negatives: unmatched GT segments

            seg_precision = seg_tp / (seg_tp + seg_fp) if (seg_tp + seg_fp) > 0 else 0.0
            seg_recall = seg_tp / (seg_tp + seg_fn) if (seg_tp + seg_fn) > 0 else 0.0
            seg_f1 = 2 * (seg_precision * seg_recall) / (seg_precision + seg_recall) if (seg_precision + seg_recall) > 0 else 0.0

            # Add segment-level metrics to results
            if activity_id in roc_auc_results:
                roc_auc_results[activity_id].update({
                    'segment_precision': seg_precision,
                    'segment_recall': seg_recall,
                    'segment_f1': seg_f1,
                    'num_gt_segments': num_gt_segments,
                    'num_pred_segments': num_pred_segments,
                    'num_matched_segments': num_matched
                })

        return roc_auc_results
    
    def create_true_frame_level_timeline_from_events(self, 
                                          pred_df: pd.DataFrame,
                                          start_time: datetime,
                                          end_time: datetime,
                                          aggregation='average') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create frame-level timeline by carrying forward predictions between events
        
        Uses carry-forward logic: each prediction persists until the next event.
        This matches how GT timelines are created from segments.
        """
        total_seconds = (end_time - start_time).total_seconds()
        timeline_length = int(total_seconds / self.resolution)
        
        # Initialize timelines
        activity_timeline = np.array(['None'] * timeline_length, dtype=object)
        confidence_timeline = np.zeros(timeline_length, dtype=float)
        
        # Sort events by timestamp
        pred_df = pred_df.sort_values('timestamp').reset_index(drop=True)
                
        # Track current state (carry-forward)
        current_activity = None
        current_confidence = 0.0
        
        for idx, row in pred_df.iterrows():
            timestamp = row['timestamp']
            annotation = str(row['annotation']).strip()
            
            # Skip invalid annotations
            if not annotation or annotation == 'nan' or annotation == 'Other_Activity':
                continue
            
            # Get confidence
            confidence = row.get('confidence', 1.0) if 'confidence' in pred_df.columns else 1.0
            
            # Extract activity ID from annotation
            activity_id = None
            
            # Handle comma-separated annotations
            annotations = [a.strip() for a in annotation.split(',')]
            
            for ann in annotations:
                if '-start' in ann.lower():
                    activity_id = ann.lower().split('-start')[0].strip()
                    current_activity = activity_id
                    current_confidence = confidence
                    break
                elif '-end' in ann.lower():
                    activity_id = ann.lower().split('-end')[0].strip()
                    # End marker - stop carrying forward this activity
                    if current_activity == activity_id:
                        current_activity = None
                        current_confidence = 0.0
                    break
                else:
                    # Continuous label
                    match = re.match(r'^(\d+)', ann)
                    if match:
                        activity_id = match.group(1)
                        current_activity = activity_id
                        current_confidence = confidence
                        break
            
            # Calculate frame index
            elapsed = (timestamp - start_time).total_seconds()
            frame_idx = int(elapsed / self.resolution)
            
            # Clip to valid range
            if frame_idx < 0 or frame_idx >= timeline_length:
                continue
            
            # Fill from this frame until next event (carry-forward)
            # BUT: only carry forward if next event is the same activity
            if idx < len(pred_df) - 1:
                next_row = pred_df.iloc[idx + 1]
                next_timestamp = next_row['timestamp']
                next_annotation = str(next_row['annotation']).strip()

                # Check if next event is different activity
                next_activity_id = None
                if next_annotation and next_annotation != 'nan':
                    if '-start' in next_annotation.lower():
                        next_activity_id = next_annotation.lower().split('-start')[0].strip()
                    elif '-end' not in next_annotation.lower():
                        match = re.match(r'^(\d+)', next_annotation)
                        if match:
                            next_activity_id = match.group(1)

                # Only carry forward if same activity or next is end marker
                if next_activity_id == current_activity or '-end' in next_annotation.lower():
                    next_elapsed = (next_timestamp - start_time).total_seconds()
                    end_frame = int(next_elapsed / self.resolution)
                else:
                    # Different activity - don't carry forward, just fill current frame
                    end_frame = frame_idx + 1
            else:
                # Last event - carry forward to end of timeline
                end_frame = timeline_length

            # Clip end frame
            end_frame = min(end_frame, timeline_length)

            # Fill frames with current state
            if current_activity is not None and frame_idx < end_frame:
                activity_timeline[frame_idx:end_frame] = current_activity
                confidence_timeline[frame_idx:end_frame] = current_confidence
                
        return activity_timeline, confidence_timeline
    
    def calculate_roc_auc_per_class(self,
                                    pred_df: pd.DataFrame,
                                    gt_timeline: np.ndarray,
                                    start_time: datetime,
                                    end_time: datetime,
                                    activity_classes: List[str],
                                    aggregation='average') -> Dict:
        """
        Calculate TRUE frame-level ROC and AUC for each activity class
        
        Uses carry-forward approach for sparse predictions.
        """
        # Create TRUE frame-level timeline from events
        pred_timeline, pred_confidence_timeline = self.create_true_frame_level_timeline_from_events(
            pred_df, start_time, end_time, aggregation=aggregation
        )
        
        if len(pred_timeline) != len(gt_timeline):
            min_len = min(len(pred_timeline), len(gt_timeline))
            pred_timeline = pred_timeline[:min_len]
            pred_confidence_timeline = pred_confidence_timeline[:min_len]
            gt_timeline = gt_timeline[:min_len]
        
        roc_auc_results = {}
        
        for activity_id in activity_classes:
            # Binary labels for this class
            y_true_full = (gt_timeline == activity_id).astype(int)

            # Get confidence scores where this class was predicted
            y_scores_full = np.zeros(len(pred_confidence_timeline))
            for i in range(len(pred_confidence_timeline)):
                if pred_timeline[i] == activity_id:
                    y_scores_full[i] = pred_confidence_timeline[i]

            # IMPORTANT FIX: Only include frames where activity is either in GT or predicted
            # This prevents True Negatives (other activities) from inflating AUC
            # when they're assigned score=0.0
            relevant_mask = (gt_timeline == activity_id) | (pred_timeline == activity_id)
            y_true = y_true_full[relevant_mask]
            y_scores = y_scores_full[relevant_mask]

            # Only calculate if we have both positive and negative samples AND varied scores
            if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_auc_results[activity_id] = {
                        'auc': roc_auc,
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate ROC/AUC for class {activity_id}: {e}")
                    roc_auc_results[activity_id] = {'auc': None}
            else:
                roc_auc_results[activity_id] = {'auc': None}
        
        return roc_auc_results
    
    def validate_frame_level_roc(self, activity_timeline, confidence_timeline, verbose=True):
        """
        Validate that the timeline represents TRUE frame-level (not segment-level)
        
        Checks if confidence varies within continuous activity segments.
        """
        segments_with_variation = 0
        segments_constant = 0
        
        # Find continuous segments
        current_activity = None
        segment_confidences = []
        
        for i in range(len(activity_timeline)):
            if activity_timeline[i] != current_activity:
                # Segment boundary - analyze previous segment
                if len(segment_confidences) > 1:
                    confidences = np.array(segment_confidences)
                    confidences = confidences[confidences > 0]
                    
                    if len(confidences) > 1:
                        std_dev = np.std(confidences)
                        if std_dev < 1e-6:
                            segments_constant += 1
                        else:
                            segments_with_variation += 1
                
                current_activity = activity_timeline[i]
                segment_confidences = [confidence_timeline[i]]
            else:
                segment_confidences.append(confidence_timeline[i])
        
        if verbose:
            print(f"\nFrame-Level ROC Validation:")
            print(f"  Segments with varying confidence: {segments_with_variation}")
            print(f"  Segments with constant confidence: {segments_constant}")
            
            if segments_with_variation > 0:
                print(f"[OK] TRUE frame-level: Confidence varies within segments!")
            else:
                print(f"[WARNING] Appears to be segment-level (constant confidence)")
                
        return segments_with_variation > 0
    
    def segment_overlap(self, seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        """Calculate overlap duration between two segments"""
        s1, e1 = seg1
        s2, e2 = seg2
        return max(0, min(e1, e2) - max(s1, s2))
    
    def evaluate_segments(self,
                         predicted_segments: List[Dict],
                         ground_truth_segments: List[Dict]) -> Dict:
        """
        Evaluate segments directly without reading files
        """
        if len(predicted_segments) == 0 or len(ground_truth_segments) == 0:
            print("Warning: No segments provided")
            return {}
        
        # Determine timeline bounds
        all_times = []
        for seg in predicted_segments + ground_truth_segments:
            all_times.extend([seg['start_time'], seg['end_time']])
        
        start_time = min(all_times)
        end_time = max(all_times)
        
        # Create timelines
        pred_timeline, pred_classes = self.create_timeline(predicted_segments, start_time, end_time)
        gt_timeline, gt_classes = self.create_timeline(ground_truth_segments, start_time, end_time)
        
        all_classes = sorted(list(set(pred_classes + gt_classes)))
        
        # Calculate metrics
        results = self.calculate_time_continuous_metrics(pred_timeline, gt_timeline, all_classes)
        
        results['metadata'] = {
            'timeline_start': start_time,
            'timeline_end': end_time,
            'total_duration_seconds': (end_time - start_time).total_seconds(),
            'resolution_seconds': self.resolution,
            'num_pred_segments': len(predicted_segments),
            'num_gt_segments': len(ground_truth_segments),
            'activity_classes': all_classes
        }
        
        return results
    
    def evaluate_with_dual_roc(self,
                                predicted_file: str,
                                ground_truth_file: str,
                                aggregation='average',
                                activity_filter=None) -> Dict:
        """
        Evaluate with both TRUE frame-level and segment-level ROC curves

        Args:
            predicted_file: Path to predictions
            ground_truth_file: Path to ground truth
            aggregation: How to aggregate frame-level confidences ('average', 'max', 'median')
            activity_filter: List of activity IDs to include (e.g., ['1', '2', '3'])
        """
        # Parse files
        pred_df = self.parse_log_file(predicted_file)
        gt_df = self.parse_log_file(ground_truth_file)

        # Extract segments
        pred_segments = self.extract_segments(pred_df, confidence_threshold=self.confidence_threshold, max_gap_seconds=self.max_gap_seconds)
        gt_segments = self.extract_segments(gt_df, confidence_threshold=0.0, max_gap_seconds=self.max_gap_seconds)  # GT: accept all

        # Filter segments by activities if specified
        # IMPORTANT: We filter segments but NOT pred_df
        # The dataframe needs ALL events for correct carry-forward timeline creation
        # Otherwise, filtering creates artificial gaps that get filled incorrectly
        if activity_filter is not None:
            pred_segments = [s for s in pred_segments if s['segment_id'] in activity_filter]
            gt_segments = [s for s in gt_segments if s['segment_id'] in activity_filter]

        # Adjust predicted segments for start time tolerance (timeline metrics)
        pred_segments_adjusted = self._adjust_segments_for_tolerance(pred_segments, gt_segments) #DISABLED

        # Get timeline bounds
        all_times = []
        for seg in pred_segments + gt_segments:
            all_times.extend([seg['start_time'], seg['end_time']])
        
        if not all_times:
            return {}
        
        start_time = min(all_times)
        end_time = max(all_times)
        
        # Get activity classes
        activity_classes = sorted(list(set(
            [s['segment_id'] for s in pred_segments + gt_segments]
        )))
        
        # Create ground truth timeline
        gt_timeline, _ = self.create_timeline(gt_segments, start_time, end_time)
        
        # Calculate standard segment-based metrics (using tolerance-adjusted segments)
        metrics = self.evaluate_segments(pred_segments_adjusted, gt_segments)
        
        # Calculate TRUE frame-level ROC
        frame_roc = self.calculate_roc_auc_per_class(
            pred_df,
            gt_timeline,
            start_time,
            end_time,
            activity_classes,
            aggregation=aggregation
        )
        
        # Validate it's truly frame-level
        pred_timeline, pred_confidence_timeline = self.create_true_frame_level_timeline_from_events(
            pred_df, start_time, end_time, aggregation=aggregation
        )
        
        is_true_frame_level = self.validate_frame_level_roc(
            pred_timeline, pred_confidence_timeline, verbose=True
        )
        
        # Calculate segment-level ROC
        segment_roc = self.calculate_segment_roc_auc_per_class(pred_segments, gt_segments)

        # Calculate macro-average AUC
        frame_aucs = [frame_roc[act]['auc'] for act in activity_classes
                      if act in frame_roc and frame_roc[act].get('auc') is not None]
        segment_aucs = [segment_roc[act]['auc'] for act in activity_classes
                        if act in segment_roc and segment_roc[act].get('auc') is not None]

        macro_frame_auc = np.mean(frame_aucs) if len(frame_aucs) > 0 else None
        macro_segment_auc = np.mean(segment_aucs) if len(segment_aucs) > 0 else None

        # Calculate macro-average segment precision/recall/F1
        segment_precisions = [segment_roc[act]['segment_precision'] for act in activity_classes
                              if act in segment_roc and 'segment_precision' in segment_roc[act]]
        segment_recalls = [segment_roc[act]['segment_recall'] for act in activity_classes
                           if act in segment_roc and 'segment_recall' in segment_roc[act]]
        segment_f1s = [segment_roc[act]['segment_f1'] for act in activity_classes
                       if act in segment_roc and 'segment_f1' in segment_roc[act]]

        macro_segment_precision = np.mean(segment_precisions) if len(segment_precisions) > 0 else None
        macro_segment_recall = np.mean(segment_recalls) if len(segment_recalls) > 0 else None
        macro_segment_f1 = np.mean(segment_f1s) if len(segment_f1s) > 0 else None

        # Add AUC and segment metrics to macro_avg
        macro_avg = metrics.get('macro_avg', {}).copy()
        macro_avg['frame_auc'] = macro_frame_auc
        macro_avg['segment_auc'] = macro_segment_auc
        macro_avg['segment_precision'] = macro_segment_precision
        macro_avg['segment_recall'] = macro_segment_recall
        macro_avg['segment_f1'] = macro_segment_f1

        # Add AUC and segment metrics to weighted_avg
        weighted_avg = metrics.get('weighted_avg', {}).copy()
        if len(frame_aucs) > 0 and 'per_class' in metrics:
            total_gt_duration = sum(metrics['per_class'][act]['gt_total_duration']
                                   for act in activity_classes if act in metrics['per_class'])
            if total_gt_duration > 0:
                weighted_frame_auc = sum(
                    frame_roc[act]['auc'] * metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in frame_roc and frame_roc[act].get('auc') is not None
                    and act in metrics['per_class']
                ) / total_gt_duration
                weighted_segment_auc = sum(
                    segment_roc[act]['auc'] * metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc and segment_roc[act].get('auc') is not None
                    and act in metrics['per_class']
                ) / total_gt_duration
                weighted_segment_precision = sum(
                    segment_roc[act]['segment_precision'] * metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc and 'segment_precision' in segment_roc[act]
                    and act in metrics['per_class']
                ) / total_gt_duration
                weighted_segment_recall = sum(
                    segment_roc[act]['segment_recall'] * metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc and 'segment_recall' in segment_roc[act]
                    and act in metrics['per_class']
                ) / total_gt_duration
                weighted_segment_f1 = sum(
                    segment_roc[act]['segment_f1'] * metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc and 'segment_f1' in segment_roc[act]
                    and act in metrics['per_class']
                ) / total_gt_duration
                weighted_avg['frame_auc'] = weighted_frame_auc
                weighted_avg['segment_auc'] = weighted_segment_auc
                weighted_avg['segment_precision'] = weighted_segment_precision
                weighted_avg['segment_recall'] = weighted_segment_recall
                weighted_avg['segment_f1'] = weighted_segment_f1

        # Combine results
        results = {
            'per_class': metrics.get('per_class', {}),
            'macro_avg': macro_avg,
            'micro_avg': metrics.get('micro_avg', {}),
            'weighted_avg': weighted_avg,
            'frame_level': frame_roc,
            'segment_level': segment_roc,
            'metadata': {
                'timeline_start': start_time,
                'timeline_end': end_time,
                'total_duration_seconds': (end_time - start_time).total_seconds(),
                'resolution_seconds': self.resolution,
                'start_time_threshold': self.start_time_threshold,
                'num_pred_segments': len(pred_segments),
                'num_gt_segments': len(gt_segments),
                'activity_classes': activity_classes,
                'aggregation_method': aggregation,
                'is_true_frame_level': is_true_frame_level
            }
        }
        
        return results
    
    def plot_dual_roc_curves(self, results: Dict, save_path: str = 'roc_comparison.png', activity_filter=None):
        """
        Plot both frame-level and segment-level ROC curves side by side
        """
        import matplotlib.pyplot as plt
        import locale

        # Set locale for comma decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, 'sl_SI.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_NUMERIC, 'Slovenian_Slovenia.1250')
            except:
                pass  # If locale setting fails, continue with default

        # Configure matplotlib to use comma as decimal separator
        plt.rcParams['axes.formatter.use_locale'] = True

        if 'frame_level' not in results or 'segment_level' not in results:
            print("Both ROC types not available in results")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Filter activity classes
        activity_classes = results['metadata']['activity_classes']
        if activity_filter:
            activity_classes = [a for a in activity_classes if a in activity_filter]

        # Get colors and names
        activity_colors = self.get_activity_colors()
        activity_names = self.get_activity_names()

        # Left: Frame-level ROC
        for activity_id in activity_classes:
            color = activity_colors.get(activity_id, 'gray')
            activity_name = activity_names.get(activity_id, f"Activity {activity_id}")
            if activity_id in results['frame_level'] and results['frame_level'][activity_id]['auc'] is not None:
                fpr = results['frame_level'][activity_id]['fpr']
                tpr = results['frame_level'][activity_id]['tpr']
                auc_val = results['frame_level'][activity_id]['auc']
                # Format AUC with comma as decimal separator
                auc_str = f"{auc_val:.2f}".replace('.', ',')
                ax1.plot(fpr, tpr, color=color, lw=2,
                        label=f"{activity_name} (AUC={auc_str})")

        ax1.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title('ROC krivulja na nivoju časovne enote')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right: Segment-level ROC
        for activity_id in activity_classes:
            color = activity_colors.get(activity_id, 'gray')
            activity_name = activity_names.get(activity_id, f"Activity {activity_id}")
            if activity_id in results['segment_level'] and results['segment_level'][activity_id]['auc'] is not None:
                fpr = results['segment_level'][activity_id]['fpr']
                tpr = results['segment_level'][activity_id]['tpr']
                auc_val = results['segment_level'][activity_id]['auc']
                # Format AUC with comma as decimal separator
                auc_str = f"{auc_val:.2f}".replace('.', ',')
                ax2.plot(fpr, tpr, color=color, lw=2,
                        label=f"{activity_name} (AUC={auc_str})")

        ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.set_title('ROC krivulja na nivoju segmenta')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nROC curves saved to: {save_path}")
    
    def print_report(self, results: Dict, activity_filter=None):
        """Print a formatted evaluation report"""
        if not results:
            print("No results to display")
            return
        
        print("=" * 70)
        print("TIME-CONTINUOUS MULTI-CLASS EVALUATION REPORT")
        print("=" * 70)
        
        if 'metadata' in results:
            meta = results['metadata']
            
            if 'total_duration_seconds' in meta:
                print(f"\nTimeline Duration: {meta['total_duration_seconds']:.0f} seconds")
                print(f"Resolution: {meta['resolution_seconds']}s")
            elif 'num_files' in meta:
                print(f"\nNumber of files: {meta['num_files']}")
                print(f"Total predicted segments: {meta['num_pred_segments']}")
                print(f"Total ground truth segments: {meta['num_gt_segments']}")

            if meta.get('start_time_threshold', 0) > 0:
                print(f"Start Time Tolerance: {meta['start_time_threshold']}s")

            activity_classes = meta.get('activity_classes', [])
            if activity_filter:
                activity_classes = [a for a in activity_classes if a in activity_filter]
            print(f"Activity Classes: {', '.join(activity_classes)}")
        
        # Aggregate metrics
        print("\n" + "=" * 70)
        print("AGGREGATE METRICS")
        print("=" * 70)
        
        if 'macro_avg' in results:
            print("\n--- Macro-Average (simple average across classes) ---")
            print(f"Precision:        {results['macro_avg']['precision']:.4f}")
            print(f"Recall:           {results['macro_avg']['recall']:.4f}")
            print(f"F1 Score:         {results['macro_avg']['f1_score']:.4f}")
            if 'iou' in results['macro_avg']:
                print(f"IoU:              {results['macro_avg']['iou']:.4f}")
            if 'frame_auc' in results['macro_avg'] and results['macro_avg']['frame_auc'] is not None:
                print(f"Frame-Level AUC:  {results['macro_avg']['frame_auc']:.4f}")
            if 'segment_auc' in results['macro_avg'] and results['macro_avg']['segment_auc'] is not None:
                print(f"Segment-Level AUC: {results['macro_avg']['segment_auc']:.4f}")

        if 'micro_avg' in results:
            print("\n--- Micro-Average (pooled TP/FP/FN across classes) ---")
            print(f"Precision: {results['micro_avg']['precision']:.4f}")
            print(f"Recall:    {results['micro_avg']['recall']:.4f}")
            print(f"F1 Score:  {results['micro_avg']['f1_score']:.4f}")
            if 'iou' in results['micro_avg']:
                print(f"IoU:       {results['micro_avg']['iou']:.4f}")
            if 'accuracy' in results['micro_avg']:
                print(f"Accuracy:  {results['micro_avg']['accuracy']:.4f}")
            print(f"Total TP Duration: {results['micro_avg']['total_TP_duration']:.1f}s")
            print(f"Total FP Duration: {results['micro_avg']['total_FP_duration']:.1f}s")
            print(f"Total FN Duration: {results['micro_avg']['total_FN_duration']:.1f}s")

        if 'weighted_avg' in results:
            print("\n--- Weighted-Average (weighted by GT duration) ---")
            print(f"Precision:        {results['weighted_avg']['precision']:.4f}")
            print(f"Recall:           {results['weighted_avg']['recall']:.4f}")
            print(f"F1 Score:         {results['weighted_avg']['f1_score']:.4f}")
            if 'iou' in results['weighted_avg']:
                print(f"IoU:              {results['weighted_avg']['iou']:.4f}")
            if 'frame_auc' in results['weighted_avg'] and results['weighted_avg'].get('frame_auc') is not None:
                print(f"Frame-Level AUC:  {results['weighted_avg']['frame_auc']:.4f}")
            if 'segment_auc' in results['weighted_avg'] and results['weighted_avg'].get('segment_auc') is not None:
                print(f"Segment-Level AUC: {results['weighted_avg']['segment_auc']:.4f}")
        
        # Per-class metrics
        if 'per_class' in results:
            print("\n" + "=" * 70)
            print("PER-CLASS METRICS")
            print("=" * 70)
            
            activities_to_show = sorted(results['per_class'].keys())
            if activity_filter:
                activities_to_show = [a for a in activities_to_show if a in activity_filter]
            
            for activity_id in activities_to_show:
                metrics = results['per_class'][activity_id]
                print(f"\n--- Activity {activity_id} ---")
                print(f"  Ground Truth Duration: {metrics['gt_total_duration']:.1f}s")
                print(f"  Predicted Duration:    {metrics['pred_total_duration']:.1f}s")
                print(f"  TP Duration:           {metrics['TP_duration']:.1f}s")
                print(f"  FP Duration:           {metrics['FP_duration']:.1f}s")
                print(f"  FN Duration:           {metrics['FN_duration']:.1f}s")
                if 'TN_duration' in metrics:
                    print(f"  TN Duration:           {metrics['TN_duration']:.1f}s")
                print(f"  Precision:             {metrics['precision']:.4f}")
                print(f"  Recall:                {metrics['recall']:.4f}")
                print(f"  F1 Score:              {metrics['f1_score']:.4f}")
                if 'accuracy' in metrics:
                    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
                print(f"  IoU:                   {metrics['iou']:.4f}")

                # Add frame-level AUC if available
                if 'frame_level' in results and activity_id in results['frame_level']:
                    auc_val = results['frame_level'][activity_id].get('auc')
                    if auc_val is not None:
                        print(f"  Frame-Level AUC:       {auc_val:.4f}")

                # Add segment-level AUC if available
                if 'segment_level' in results and activity_id in results['segment_level']:
                    auc_val = results['segment_level'][activity_id].get('auc')
                    if auc_val is not None:
                        print(f"  Segment-Level AUC:     {auc_val:.4f}")
        
        print("\n" + "=" * 70)
        
    def evaluate_multiple_files(self, 
                            predicted_files: List[str],
                            ground_truth_files: List[str],
                            aggregation='average') -> Dict:
        """
        Evaluate predictions across multiple files with TRUE frame-level ROC
        
        Args:
            predicted_files: List of paths to prediction files
            ground_truth_files: List of paths to ground truth files
            aggregation: How to aggregate events per frame ('average', 'max', 'median')
            
        Returns:
            Dictionary containing aggregated metrics and per-file results
        """
        if len(predicted_files) != len(ground_truth_files):
            raise ValueError("Number of prediction and ground truth files must match")
        
        print(f"\nEvaluating {len(predicted_files)} files...")
        
        # Storage for aggregated data
        all_pred_segments = []
        all_gt_segments = []
        per_file_results = []
        
        # Store per-file data for cumulative ROC calculation
        all_file_pred_dfs = []
        all_file_gt_timelines = []
        all_file_start_times = []
        all_file_end_times = []
        
        # Get activity classes from all files
        activity_classes_set = set()
        
        # Process each file
        for file_idx, (pred_file, gt_file) in enumerate(zip(predicted_files, ground_truth_files)):
            # Parse files
            pred_df = self.parse_log_file(pred_file)
            gt_df = self.parse_log_file(gt_file)
            
            # Extract segments
            pred_segments = self.extract_segments(pred_df, confidence_threshold=self.confidence_threshold, max_gap_seconds=self.max_gap_seconds)
            gt_segments = self.extract_segments(gt_df, confidence_threshold=0.0, max_gap_seconds=self.max_gap_seconds)  # GT: accept all
            
            # Collect activity classes
            for seg in pred_segments + gt_segments:
                activity_classes_set.add(seg['segment_id'])
            
            # Skip if no segments
            if len(pred_segments) == 0 and len(gt_segments) == 0:
                continue
            
            # Get timeline bounds for THIS file
            all_times = []
            for seg in pred_segments + gt_segments:
                all_times.extend([seg['start_time'], seg['end_time']])
            
            if not all_times:
                continue
            
            file_start_time = min(all_times)
            file_end_time = max(all_times)
            
            # Get activity classes for this specific file
            file_classes = sorted(list(set([s['segment_id'] for s in pred_segments + gt_segments])))
            
            # Create ground truth timeline for THIS file
            file_gt_timeline, _ = self.create_timeline(gt_segments, file_start_time, file_end_time)
            
            # Calculate per-file TRUE frame-level ROC
            frame_roc_file = self.calculate_roc_auc_per_class(
                pred_df,
                file_gt_timeline,
                file_start_time,
                file_end_time,
                file_classes,
                aggregation=aggregation
            )
            
            # Store data for cumulative calculation
            all_file_pred_dfs.append(pred_df)
            all_file_gt_timelines.append(file_gt_timeline)
            all_file_start_times.append(file_start_time)
            all_file_end_times.append(file_end_time)
            
            # Calculate segment-level metrics for this file (using tolerance-adjusted segments)
            pred_segments_adjusted = self._adjust_segments_for_tolerance(pred_segments, gt_segments)
            file_metrics = self.evaluate_segments(pred_segments_adjusted, gt_segments)
            
            # Calculate segment-level ROC for this file
            segment_roc_file = self.calculate_segment_roc_auc_per_class(pred_segments, gt_segments)
            
            # Store per-file results
            per_file_results.append({
                'pred_file': pred_file,
                'gt_file': gt_file,
                'num_pred_segments': len(pred_segments),
                'num_gt_segments': len(gt_segments),
                'metrics': file_metrics,
                'frame_level': frame_roc_file,
                'segment_level': segment_roc_file
            })
            
            # Add to cumulative collections
            all_pred_segments.extend(pred_segments)
            all_gt_segments.extend(gt_segments)
        
        # Calculate cumulative metrics
        print(f"Calculating cumulative metrics across {len(all_pred_segments)} predicted and {len(all_gt_segments)} GT segments...")
        
        activity_classes = sorted(list(activity_classes_set))
        
        # Calculate cumulative segment-level metrics (using tolerance-adjusted segments)
        all_pred_segments_adjusted = self._adjust_segments_for_tolerance(all_pred_segments, all_gt_segments)
        cumulative_metrics = self.evaluate_segments(all_pred_segments_adjusted, all_gt_segments)
        
        # Calculate cumulative segment-level ROC
        segment_roc_cumulative = self.calculate_segment_roc_auc_per_class(
            all_pred_segments,
            all_gt_segments
        )
        
        # Calculate cumulative TRUE frame-level ROC
        if len(all_file_gt_timelines) > 0:
            cumulative_gt_timeline = np.concatenate(all_file_gt_timelines)
            
            # For frame-level, we need to handle each file's events separately
            # then concatenate the resulting confidence timelines
            frame_level_timelines = []
            
            for pred_df, start_time, end_time in zip(all_file_pred_dfs, 
                                                    all_file_start_times,
                                                    all_file_end_times):
                _, conf_timeline = self.create_true_frame_level_timeline_from_events(
                    pred_df, start_time, end_time, aggregation=aggregation
                )
                frame_level_timelines.append(conf_timeline)
            
            # Now calculate ROC on concatenated data
            frame_roc_cumulative = {}
            cumulative_pred_timeline = []
            
            for pred_df, start_time, end_time in zip(all_file_pred_dfs,
                                                    all_file_start_times,
                                                    all_file_end_times):
                pred_timeline, _ = self.create_true_frame_level_timeline_from_events(
                    pred_df, start_time, end_time, aggregation=aggregation
                )
                cumulative_pred_timeline.append(pred_timeline)
            
            cumulative_pred_timeline = np.concatenate(cumulative_pred_timeline)
            cumulative_conf_timeline = np.concatenate(frame_level_timelines)
            
            # Calculate ROC for each activity
            for activity_id in activity_classes:
                y_true = (cumulative_gt_timeline == activity_id).astype(int)
                y_scores = np.zeros(len(cumulative_conf_timeline))
                
                for i in range(len(cumulative_pred_timeline)):
                    if cumulative_pred_timeline[i] == activity_id:
                        y_scores[i] = cumulative_conf_timeline[i]
                
                if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
                    try:
                        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                        roc_auc = auc(fpr, tpr)
                        frame_roc_cumulative[activity_id] = {
                            'auc': roc_auc,
                            'fpr': fpr,
                            'tpr': tpr,
                            'thresholds': thresholds
                        }
                    except:
                        frame_roc_cumulative[activity_id] = {'auc': None}
                else:
                    frame_roc_cumulative[activity_id] = {'auc': None}
            
            # Validate cumulative frame-level
            print("\nCumulative Frame-Level ROC Validation:")
            is_true_frame_level = self.validate_frame_level_roc(
                cumulative_pred_timeline, cumulative_conf_timeline, verbose=True
            )
        else:
            frame_roc_cumulative = {}
            is_true_frame_level = False

        # Calculate macro-average AUC
        frame_aucs = [frame_roc_cumulative[act]['auc'] for act in activity_classes
                      if act in frame_roc_cumulative and frame_roc_cumulative[act].get('auc') is not None]
        segment_aucs = [segment_roc_cumulative[act]['auc'] for act in activity_classes
                        if act in segment_roc_cumulative and segment_roc_cumulative[act].get('auc') is not None]

        macro_frame_auc = np.mean(frame_aucs) if len(frame_aucs) > 0 else None
        macro_segment_auc = np.mean(segment_aucs) if len(segment_aucs) > 0 else None

        # Calculate macro-average segment precision/recall/F1
        segment_precisions = [segment_roc_cumulative[act]['segment_precision'] for act in activity_classes
                              if act in segment_roc_cumulative and 'segment_precision' in segment_roc_cumulative[act]]
        segment_recalls = [segment_roc_cumulative[act]['segment_recall'] for act in activity_classes
                           if act in segment_roc_cumulative and 'segment_recall' in segment_roc_cumulative[act]]
        segment_f1s = [segment_roc_cumulative[act]['segment_f1'] for act in activity_classes
                       if act in segment_roc_cumulative and 'segment_f1' in segment_roc_cumulative[act]]

        macro_segment_precision = np.mean(segment_precisions) if len(segment_precisions) > 0 else None
        macro_segment_recall = np.mean(segment_recalls) if len(segment_recalls) > 0 else None
        macro_segment_f1 = np.mean(segment_f1s) if len(segment_f1s) > 0 else None

        # Add AUC and segment metrics to macro_avg
        macro_avg = cumulative_metrics.get('macro_avg', {}).copy()
        macro_avg['frame_auc'] = macro_frame_auc
        macro_avg['segment_auc'] = macro_segment_auc
        macro_avg['segment_precision'] = macro_segment_precision
        macro_avg['segment_recall'] = macro_segment_recall
        macro_avg['segment_f1'] = macro_segment_f1

        # Add AUC to weighted_avg
        weighted_avg = cumulative_metrics.get('weighted_avg', {}).copy()
        if len(frame_aucs) > 0 and 'per_class' in cumulative_metrics:
            total_gt_duration = sum(cumulative_metrics['per_class'][act]['gt_total_duration']
                                   for act in activity_classes if act in cumulative_metrics['per_class'])
            if total_gt_duration > 0:
                weighted_frame_auc = sum(
                    frame_roc_cumulative[act]['auc'] * cumulative_metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in frame_roc_cumulative and frame_roc_cumulative[act].get('auc') is not None
                    and act in cumulative_metrics['per_class']
                ) / total_gt_duration
                weighted_segment_auc = sum(
                    segment_roc_cumulative[act]['auc'] * cumulative_metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc_cumulative and segment_roc_cumulative[act].get('auc') is not None
                    and act in cumulative_metrics['per_class']
                ) / total_gt_duration
                weighted_segment_precision = sum(
                    segment_roc_cumulative[act]['segment_precision'] * cumulative_metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc_cumulative and 'segment_precision' in segment_roc_cumulative[act]
                    and act in cumulative_metrics['per_class']
                ) / total_gt_duration
                weighted_segment_recall = sum(
                    segment_roc_cumulative[act]['segment_recall'] * cumulative_metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc_cumulative and 'segment_recall' in segment_roc_cumulative[act]
                    and act in cumulative_metrics['per_class']
                ) / total_gt_duration
                weighted_segment_f1 = sum(
                    segment_roc_cumulative[act]['segment_f1'] * cumulative_metrics['per_class'][act]['gt_total_duration']
                    for act in activity_classes
                    if act in segment_roc_cumulative and 'segment_f1' in segment_roc_cumulative[act]
                    and act in cumulative_metrics['per_class']
                ) / total_gt_duration
                weighted_avg['frame_auc'] = weighted_frame_auc
                weighted_avg['segment_auc'] = weighted_segment_auc
                weighted_avg['segment_precision'] = weighted_segment_precision
                weighted_avg['segment_recall'] = weighted_segment_recall
                weighted_avg['segment_f1'] = weighted_segment_f1

        # Prepare results
        results = {
            'metadata': {
                'num_files': len(predicted_files),
                'num_pred_segments': len(all_pred_segments),
                'num_gt_segments': len(all_gt_segments),
                'activity_classes': activity_classes,
                'start_time_threshold': self.start_time_threshold,
                'aggregation_method': aggregation,
                'is_true_frame_level': is_true_frame_level
            },
            'per_class': cumulative_metrics.get('per_class', {}),
            'macro_avg': macro_avg,
            'micro_avg': cumulative_metrics.get('micro_avg', {}),
            'weighted_avg': weighted_avg,
            'frame_level': frame_roc_cumulative,
            'segment_level': segment_roc_cumulative,
            'per_file_results': per_file_results
        }

        return results
    
    def generate_aggregated_roc_curves_vertical_avg(self, all_fold_results, output_dir, activity_filter):
        """
        Generate ROC curves using vertical averaging (macro-averaging)
        
        Interpolates all fold ROC curves to common FPR points and averages TPR.
        This is the most common approach in cross-validation.
        """
        import matplotlib.pyplot as plt
        
        print(f"\nAggregating ROC data from {len(all_fold_results)} folds using vertical averaging...")
        
        if not activity_filter:
            activity_filter = all_fold_results[0]['metadata']['activity_classes']
        
        # Common FPR points for interpolation
        mean_fpr = np.linspace(0, 1, 100)
        
        averaged_frame_roc = {}
        averaged_segment_roc = {}
        
        for activity_id in activity_filter:
            # Collect TPR curves for each fold (interpolated to common FPR)
            frame_tprs = []
            frame_aucs = []
            segment_tprs = []
            segment_aucs = []
            
            for fold_result in all_fold_results:
                # Frame-level
                if 'frame_level' in fold_result and activity_id in fold_result['frame_level']:
                    frame_data = fold_result['frame_level'][activity_id]
                    if frame_data.get('auc') is not None:
                        # Interpolate this fold's TPR to common FPR points
                        interp_tpr = np.interp(mean_fpr, frame_data['fpr'], frame_data['tpr'])
                        frame_tprs.append(interp_tpr)
                        frame_aucs.append(frame_data['auc'])
                
                # Segment-level
                if 'segment_level' in fold_result and activity_id in fold_result['segment_level']:
                    seg_data = fold_result['segment_level'][activity_id]
                    if seg_data.get('auc') is not None:
                        interp_tpr = np.interp(mean_fpr, seg_data['fpr'], seg_data['tpr'])
                        segment_tprs.append(interp_tpr)
                        segment_aucs.append(seg_data['auc'])
            
            # Average TPR across folds
            if len(frame_tprs) > 0:
                mean_tpr = np.mean(frame_tprs, axis=0)
                mean_tpr[0] = 0.0  # Ensure it starts at (0,0)
                mean_tpr[-1] = 1.0  # Ensure it ends at (1,1)
                
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(frame_aucs)
                
                averaged_frame_roc[activity_id] = {
                    'fpr': mean_fpr,
                    'tpr': mean_tpr,
                    'auc': mean_auc,
                    'auc_std': std_auc
                }
            
            if len(segment_tprs) > 0:
                mean_tpr = np.mean(segment_tprs, axis=0)
                mean_tpr[0] = 0.0
                mean_tpr[-1] = 1.0
                
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(segment_aucs)
                
                averaged_segment_roc[activity_id] = {
                    'fpr': mean_fpr,
                    'tpr': mean_tpr,
                    'auc': mean_auc,
                    'auc_std': std_auc
                }
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Configure matplotlib to use comma as decimal separator
        plt.rcParams['axes.formatter.use_locale'] = True

        # Get colors and names
        activity_colors = self.get_activity_colors()
        activity_names = self.get_activity_names()

        # Frame-level
        for activity_id in activity_filter:
            color = activity_colors.get(activity_id, 'gray')
            activity_name = activity_names.get(activity_id, f"Activity {activity_id}")
            if activity_id in averaged_frame_roc:
                fpr = averaged_frame_roc[activity_id]['fpr']
                tpr = averaged_frame_roc[activity_id]['tpr']
                mean_auc = averaged_frame_roc[activity_id]['auc']
                std_auc = averaged_frame_roc[activity_id]['auc_std']

                # Format with comma as decimal separator
                auc_str = f"{mean_auc:.2f}±{std_auc:.2f}".replace('.', ',')
                ax1.plot(fpr, tpr, color=color, lw=2,
                        label=f"{activity_name} (AUC={auc_str})")

        ax1.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title(f'ROC krivulja na nivoju časovne enote (povprečje {len(all_fold_results)} pregibov)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Segment-level
        for activity_id in activity_filter:
            color = activity_colors.get(activity_id, 'gray')
            activity_name = activity_names.get(activity_id, f"Activity {activity_id}")
            if activity_id in averaged_segment_roc:
                fpr = averaged_segment_roc[activity_id]['fpr']
                tpr = averaged_segment_roc[activity_id]['tpr']
                mean_auc = averaged_segment_roc[activity_id]['auc']
                std_auc = averaged_segment_roc[activity_id]['auc_std']

                # Format with comma as decimal separator
                auc_str = f"{mean_auc:.2f}±{std_auc:.2f}".replace('.', ',')
                ax2.plot(fpr, tpr, color=color, lw=2,
                        label=f"{activity_name} (AUC={auc_str})")
        
        ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.set_title(f'ROC krivulja na nivoju segmenta (povprečje {len(all_fold_results)} pregibov)')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'roc_all_folds_averaged.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Averaged ROC curves saved: {save_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("AVERAGED AUC ACROSS ALL FOLDS")
        print(f"{'='*70}")
        
        print("\nFrame-Level AUC (Mean ± Std):")
        for activity_id in activity_filter:
            if activity_id in averaged_frame_roc:
                mean_auc = averaged_frame_roc[activity_id]['auc']
                std_auc = averaged_frame_roc[activity_id]['auc_std']
                print(f"  Activity {activity_id}: {mean_auc:.4f} ± {std_auc:.4f}")
        
        print("\nSegment-Level AUC (Mean ± Std):")
        for activity_id in activity_filter:
            if activity_id in averaged_segment_roc:
                mean_auc = averaged_segment_roc[activity_id]['auc']
                std_auc = averaged_segment_roc[activity_id]['auc_std']
                print(f"  Activity {activity_id}: {mean_auc:.4f} ± {std_auc:.4f}")
        
        print(f"{'='*70}\n")