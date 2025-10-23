import numpy as np
from sklearn.metrics import roc_curve, auc
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple, Dict
import re
import os

class SegmentEvaluator:
    """Evaluates segment-based recognition with both frame-level and segment-level ROC"""
    
    def __init__(self, timeline_resolution_seconds=1.0, start_time_threshold_seconds=0):
        """
        Args:
            timeline_resolution_seconds: Time resolution for timeline (default 1.0 = per second)
            start_time_threshold_seconds: Minimum time from start to consider valid (default 0)
        """
        self.resolution = timeline_resolution_seconds
        self.start_time_threshold = start_time_threshold_seconds
        
        print(f"Start time threshold:     {self.start_time_threshold}")
        
    def parse_log_file(self, filepath: str) -> pd.DataFrame:
        """Parse the log file into a structured DataFrame (space-separated format)
        
        Handles formats:
        - 7 parts: date time sensor newsensor status activity confidence
        - 6 parts: date time sensor newsensor status activity (AL output)
        - 5 parts: date time sensor status activity (ground truth)
        """
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
                        
                    elif len(parts) >= 6:
                        # AL format WITHOUT confidence: date time sensor newsensor status activity
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
                except (ValueError, IndexError) as e:
                    continue

        return pd.DataFrame(data)

    def extract_segments(self, df: pd.DataFrame, confidence_threshold: float = 0.0) -> List[Dict]:
        """Extract segments from annotations (continuous activity numbers or start/end markers)
        
        Args:
            df: DataFrame with annotations
            confidence_threshold: Minimum confidence to consider a prediction valid (default 0.0 = no filtering)
        """
        segments = []
        current_activity = None
        segment_start = None
        prev_timestamp = None
        segment_confidences = []
        
        for idx, row in df.iterrows():
            annotation = str(row['annotation']).strip()
            
            # Get confidence if available
            confidence = row.get('confidence', 1.0) if 'confidence' in df.columns else 1.0
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                annotation = 'Other_Activity'
            
            # Skip empty annotations or Other_Activity
            if not annotation or annotation == 'nan' or annotation == 'Other_Activity':
                # Close current segment if any
                if current_activity is not None and segment_start is not None and prev_timestamp is not None:
                    min_confidence = min(segment_confidences) if segment_confidences else 1.0
                    mean_confidence = np.mean(segment_confidences) if segment_confidences else 1.0
                    segments.append({
                        'segment_id': current_activity,
                        'start_time': segment_start,
                        'end_time': prev_timestamp,
                        'duration': (prev_timestamp - segment_start).total_seconds(),
                        'confidence': mean_confidence
                    })
                    current_activity = None
                    segment_start = None
                    segment_confidences = []
                prev_timestamp = row['timestamp']
                continue
            
            # Extract activity ID from annotation
            if '-start' in annotation.lower():
                activity_id = annotation.lower().split('-start')[0].strip()
                if current_activity is not None and segment_start is not None and prev_timestamp is not None:
                    min_confidence = min(segment_confidences) if segment_confidences else 1.0
                    mean_confidence = np.mean(segment_confidences) if segment_confidences else 1.0
                    segments.append({
                        'segment_id': current_activity,
                        'start_time': segment_start,
                        'end_time': prev_timestamp,
                        'duration': (prev_timestamp - segment_start).total_seconds(),
                        'confidence': mean_confidence
                    })
                current_activity = activity_id
                segment_start = row['timestamp']
                segment_confidences = [confidence]
            
            elif '-end' in annotation.lower():
                activity_id = annotation.lower().split('-end')[0].strip()
                if current_activity == activity_id and segment_start is not None:
                    segment_confidences.append(confidence)
                    min_confidence = min(segment_confidences) if segment_confidences else 1.0
                    mean_confidence = np.mean(segment_confidences) if segment_confidences else 1.0
                    segments.append({
                        'segment_id': current_activity,
                        'start_time': segment_start,
                        'end_time': row['timestamp'],
                        'duration': (row['timestamp'] - segment_start).total_seconds(),
                        'confidence': mean_confidence
                    })
                    current_activity = None
                    segment_start = None
                    segment_confidences = []
            
            else:
                # Continuous activity label (e.g., "1", "2", "1.1")
                match = re.match(r'^(\d+)', annotation)
                if match:
                    activity_id = match.group(1)
                    
                    if activity_id != current_activity:
                        if current_activity is not None and segment_start is not None and prev_timestamp is not None:
                            min_confidence = min(segment_confidences) if segment_confidences else 1.0
                            mean_confidence = np.mean(segment_confidences) if segment_confidences else 1.0
                            segments.append({
                                'segment_id': current_activity,
                                'start_time': segment_start,
                                'end_time': prev_timestamp,
                                'duration': (prev_timestamp - segment_start).total_seconds(),
                                'confidence': mean_confidence
                            })
                        current_activity = activity_id
                        segment_start = row['timestamp']
                        segment_confidences = [confidence]
                    else:
                        # Continue current segment
                        segment_confidences.append(confidence)
            
            prev_timestamp = row['timestamp']
        
        # Close any remaining open segment
        if current_activity is not None and segment_start is not None and prev_timestamp is not None:
            min_confidence = min(segment_confidences) if segment_confidences else 1.0
            mean_confidence = np.mean(segment_confidences) if segment_confidences else 1.0
            segments.append({
                'segment_id': current_activity,
                'start_time': segment_start,
                'end_time': prev_timestamp,
                'duration': (prev_timestamp - segment_start).total_seconds(),
                'confidence': mean_confidence
            })
        
        return segments
    
    def create_timeline(self, segments: List[Dict], start_time: datetime, end_time: datetime) -> Tuple[np.ndarray, List[str]]:
        """
        Create a continuous timeline representation of segments
        
        Args:
            segments: List of segment dictionaries
            start_time: Start of timeline
            end_time: End of timeline
            
        Returns:
            timeline: Array where timeline[t] = activity_id at time t (or None)
            activity_classes: List of all unique activity IDs found
        """
        # Calculate timeline length in resolution units
        total_seconds = (end_time - start_time).total_seconds()
        timeline_length = int(total_seconds / self.resolution)
        
        # Initialize timeline (empty string = no activity)
        timeline = np.array([''] * timeline_length, dtype=object)
        
        # Get all unique activity classes
        activity_classes = sorted(list(set(seg['segment_id'] for seg in segments)))
        
        # Fill timeline with activities
        for seg in segments:
            # Convert timestamps to indices
            start_idx = int((seg['start_time'] - start_time).total_seconds() / self.resolution)
            end_idx = int((seg['end_time'] - start_time).total_seconds() / self.resolution)
            
            # Clip to valid range
            start_idx = max(0, min(start_idx, timeline_length))
            end_idx = max(0, min(end_idx, timeline_length))
            
            # Fill this segment
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
        
        Args:
            pred_timeline: Predicted activity at each time point
            gt_timeline: Ground truth activity at each time point
            activity_classes: List of activity class IDs
            
        Returns:
            Dictionary with per-class and aggregate metrics
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
            
            # Calculate metrics
            precision = TP_duration / (TP_duration + FP_duration) if (TP_duration + FP_duration) > 0 else 0.0
            recall = TP_duration / (TP_duration + FN_duration) if (TP_duration + FN_duration) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # IoU (Intersection over Union)
            iou = TP_duration / (TP_duration + FP_duration + FN_duration) if (TP_duration + FP_duration + FN_duration) > 0 else 0.0
            
            per_class_metrics[activity_id] = {
                'TP_duration': TP_duration,
                'FP_duration': FP_duration,
                'FN_duration': FN_duration,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'iou': iou,
                'gt_total_duration': np.sum(gt_mask) * self.resolution,
                'pred_total_duration': np.sum(pred_mask) * self.resolution
            }
        
        # Calculate aggregate metrics
        
        # Macro-average (simple average across classes)
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
        
        # Micro-average (pool all TP/FP/FN across classes)
        total_TP = sum(m['TP_duration'] for m in per_class_metrics.values())
        total_FP = sum(m['FP_duration'] for m in per_class_metrics.values())
        total_FN = sum(m['FN_duration'] for m in per_class_metrics.values())
        
        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Weighted-average (weighted by ground truth duration)
        total_gt_duration = sum(m['gt_total_duration'] for m in per_class_metrics.values())
        if total_gt_duration > 0:
            weighted_precision = sum(m['precision'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
            weighted_recall = sum(m['recall'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
            weighted_f1 = sum(m['f1_score'] * m['gt_total_duration'] for m in per_class_metrics.values()) / total_gt_duration
        else:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0
        
        return {
            'per_class': per_class_metrics,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1,
                'total_TP_duration': total_TP,
                'total_FP_duration': total_FP,
                'total_FN_duration': total_FN
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            }
        }
    
    def calculate_segment_roc_auc_per_class(self, pred_segments: List[Dict], gt_segments: List[Dict]) -> Dict:
        """
        Calculate segment-level ROC and AUC for each activity class
        
        Uses the confidence score associated with each predicted segment.
        
        Args:
            pred_segments: List of predicted segments with confidence scores
            gt_segments: List of ground truth segments
            
        Returns:
            Dictionary with per-class ROC curves and AUC scores
        """
        from sklearn.metrics import roc_curve, auc
        
        # Get all activity classes
        activity_classes = set()
        for seg in pred_segments + gt_segments:
            activity_classes.add(seg['segment_id'])
        activity_classes = sorted(list(activity_classes))
        
        roc_auc_results = {}
        
        for activity_id in activity_classes:
            # Get ground truth segments for this activity
            gt_activity_segments = [seg for seg in gt_segments if seg['segment_id'] == activity_id]
            
            # Get predicted segments for this activity
            pred_activity_segments = [seg for seg in pred_segments if seg['segment_id'] == activity_id]
            
            if len(gt_activity_segments) == 0:
                # No ground truth for this activity
                roc_auc_results[activity_id] = {'auc': None}
                continue
            
            # Create binary labels and confidence scores
            # Strategy: Match predicted segments to ground truth segments
            y_true = []
            y_scores = []
            
            # For each ground truth segment, find if there's a matching prediction
            for gt_seg in gt_segments:
                is_target_class = 1 if gt_seg['segment_id'] == activity_id else 0
                
                # Find overlapping predicted segment with highest confidence
                best_confidence = 0.0
                best_match = None
                
                for pred_seg in pred_segments:
                    # Check if segments overlap
                    overlap_start = max(gt_seg['start_time'], pred_seg['start_time'])
                    overlap_end = min(gt_seg['end_time'], pred_seg['end_time'])
                    
                    if overlap_start < overlap_end:
                        # Segments overlap
                        if pred_seg['segment_id'] == activity_id:
                            # Predicted as target class
                            confidence = pred_seg.get('confidence', 0.5)
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_match = pred_seg
                
                y_true.append(is_target_class)
                y_scores.append(best_confidence)
            
            # Only calculate if we have both positive and negative samples AND varied scores
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            
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
                    print(f"Warning: Could not calculate segment ROC/AUC for class {activity_id}: {e}")
                    roc_auc_results[activity_id] = {'auc': None}
            else:
                roc_auc_results[activity_id] = {'auc': None}
        
        return roc_auc_results
    
    def calculate_roc_auc_per_class(self,
                                pred_timeline: np.ndarray,
                                gt_timeline: np.ndarray,
                                pred_confidence_timeline: np.ndarray,
                                activity_classes: List[str]) -> Dict:
        """
        Calculate frame-level ROC and AUC for each activity class
        
        Args:
            pred_timeline: Predicted activity at each time point
            gt_timeline: Ground truth activity at each time point
            pred_confidence_timeline: Confidence score at each time point
            activity_classes: List of activity class IDs
            
        Returns:
            Dictionary with per-class ROC curves and AUC scores
        """
        roc_auc_results = {}
        
        for activity_id in activity_classes:
            # Binary labels for this class
            y_true = (gt_timeline == activity_id).astype(int)
            
            # Get confidence scores where this class was predicted
            # Convert confidence timeline to float, treating non-numeric as 0.0
            y_scores = np.zeros(len(pred_confidence_timeline))
            for i in range(len(pred_confidence_timeline)):
                if pred_timeline[i] == activity_id:
                    # This frame predicted this activity
                    conf_val = pred_confidence_timeline[i]
                    try:
                        y_scores[i] = float(conf_val) if conf_val else 0.0
                    except (ValueError, TypeError):
                        y_scores[i] = 0.0
                else:
                    y_scores[i] = 0.0
            
            # Only calculate if we have both positive and negative samples AND varied scores
            if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
                try:
                    from sklearn.metrics import roc_curve, auc
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
    
    def create_confidence_timeline(self, segments: List[Dict], confidence_scores: List[float],
                                   start_time: datetime, end_time: datetime) -> np.ndarray:
        """
        Create timeline of confidence scores
        
        Args:
            segments: List of predicted segments
            confidence_scores: Confidence score for each segment
            start_time: Start of timeline
            end_time: End of timeline
            
        Returns:
            confidence_timeline: Array of confidence scores at each time point
        """
        total_seconds = (end_time - start_time).total_seconds()
        timeline_length = int(total_seconds / self.resolution)
        
        confidence_timeline = np.zeros(timeline_length)
        
        if confidence_scores is None or len(confidence_scores) == 0:
            return confidence_timeline
        
        for seg, conf in zip(segments, confidence_scores):
            start_idx = int((seg['start_time'] - start_time).total_seconds() / self.resolution)
            end_idx = int((seg['end_time'] - start_time).total_seconds() / self.resolution)
            
            start_idx = max(0, min(start_idx, timeline_length))
            end_idx = max(0, min(end_idx, timeline_length))
            
            # Use maximum confidence if overlapping predictions
            confidence_timeline[start_idx:end_idx] = np.maximum(
                confidence_timeline[start_idx:end_idx],
                conf
            )
        
        return confidence_timeline
    
    def segment_overlap(self, seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        """Calculate overlap duration between two segments"""
        s1, e1 = seg1
        s2, e2 = seg2
        return max(0, min(e1, e2) - max(s1, s2))
    
    def calculate_segment_based_roc(self,
                                pred_segments: List[Dict],
                                gt_segments: List[Dict],
                                confidence_scores: List[float],
                                activity_classes: List[str],
                                overlap_threshold: float = 0.5) -> Dict:
        """Calculate segment-level ROC curves using sklearn's roc_curve for robustness"""
        from sklearn.metrics import roc_curve, auc
        
        roc_results = {}
        
        for activity_id in activity_classes:
            # Filter segments for this activity
            gt_segs_class = [s for s in gt_segments if s['segment_id'] == activity_id]
            pred_segs_class = [(s, conf) for s, conf in zip(pred_segments, confidence_scores) 
                            if s['segment_id'] == activity_id]
            
            if len(gt_segs_class) == 0:
                roc_results[activity_id] = {'auc': None}
                continue
            
            # Build arrays of predictions and labels
            y_true = []  # 1 for TP, 0 for FP
            y_scores = []  # confidence scores
            
            for pred_seg, confidence in pred_segs_class:
                pred_interval = (
                    pred_seg['start_time'].timestamp(),
                    pred_seg['end_time'].timestamp()
                )
                pred_dur = pred_seg['duration']
                
                # Find maximum overlap with any GT segment
                max_overlap = 0
                best_match_gt = None
                for gt_seg in gt_segs_class:
                    gt_interval = (
                        gt_seg['start_time'].timestamp(),
                        gt_seg['end_time'].timestamp()
                    )
                    overlap = self.segment_overlap(pred_interval, gt_interval)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match_gt = gt_seg
                
                # Determine if TP or FP
                is_valid_match = False
                if pred_dur > 0 and (max_overlap / pred_dur) >= overlap_threshold:
                    if self.start_time_threshold > 0 and best_match_gt is not None:
                        start_time_diff = abs((pred_seg['start_time'] - best_match_gt['start_time']).total_seconds())
                        if start_time_diff <= self.start_time_threshold:
                            is_valid_match = True
                    else:
                        is_valid_match = True
                
                y_true.append(1 if is_valid_match else 0)
                y_scores.append(confidence)
            
            # Calculate ROC curve using sklearn (handles sorting automatically)
            if len(y_true) > 0 and len(set(y_true)) > 1:  # Need both classes
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_results[activity_id] = {
                        'auc': roc_auc,
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate ROC for activity {activity_id}: {e}")
                    roc_results[activity_id] = {'auc': None}
            else:
                # Not enough data for ROC curve
                roc_results[activity_id] = {'auc': None}
        
        return roc_results

    def create_segment_based_confidence_timeline(self, pred_segments, start_time, end_time):
        """
        Create confidence timeline from segments (not individual events)
        
        Args:
            pred_segments: List of predicted segments (dicts with 'confidence', 'start_time', 'end_time')
            start_time: Start of timeline
            end_time: End of timeline
            
        Returns:
            confidence_timeline: Array of confidence scores at each time point
        """
        total_seconds = (end_time - start_time).total_seconds()
        timeline_length = int(total_seconds / self.resolution)
        
        confidence_timeline = np.zeros(timeline_length, dtype=float)
        
        # Fill timeline with segment-level confidences
        for seg in pred_segments:
            if not isinstance(seg, dict):
                continue
            
            confidence = seg.get('confidence', 1.0)
            seg_start = seg.get('start_time')
            seg_end = seg.get('end_time')
            
            if seg_start is None or seg_end is None:
                continue
            
            start_idx = int((seg_start - start_time).total_seconds() / self.resolution)
            end_idx = int((seg_end - start_time).total_seconds() / self.resolution)
            
            start_idx = max(0, min(start_idx, timeline_length))
            end_idx = max(0, min(end_idx, timeline_length))
            
            if start_idx < end_idx:
                confidence_timeline[start_idx:end_idx] = np.maximum(
                    confidence_timeline[start_idx:end_idx],
                    confidence
                )
        
        return confidence_timeline
    
    def calculate_segment_based_roc_multi_timeline(self,
                                               all_pred_segments: List[List[Dict]],
                                               all_gt_segments: List[List[Dict]],
                                               all_confidence_scores: List[List[float]],
                                               activity_classes: List[str],
                                               overlap_threshold: float = 0.5) -> Dict:
        """
        Calculate segment-level ROC curves across multiple timelines using sklearn
        
        This properly handles multiple recording sessions by evaluating each timeline
        separately and then pooling the results.
        
        Args:
            all_pred_segments: List of predicted segment lists (one per timeline)
            all_gt_segments: List of GT segment lists (one per timeline)
            all_confidence_scores: List of confidence score lists (one per timeline)
            activity_classes: List of activity class IDs
            overlap_threshold: Minimum overlap ratio to count as TP (default 0.5)
            
        Returns:
            Dictionary with per-class segment-level ROC curves and AUC
        """
        from sklearn.metrics import roc_curve, auc
        
        roc_results = {}
        
        for activity_id in activity_classes:
            # Collect all predictions and labels for this activity across all timelines
            y_true = []  # 1 for TP, 0 for FP
            y_scores = []  # confidence scores
            
            for timeline_idx in range(len(all_pred_segments)):
                pred_segments = all_pred_segments[timeline_idx]
                gt_segments = all_gt_segments[timeline_idx]
                confidence_scores = all_confidence_scores[timeline_idx]
                
                # Filter for this activity
                gt_segs_class = [s for s in gt_segments if s['segment_id'] == activity_id]
                pred_segs_class = [(s, conf) for s, conf in zip(pred_segments, confidence_scores) 
                                if s['segment_id'] == activity_id]
                
                # Process each prediction in this timeline
                for pred_seg, confidence in pred_segs_class:
                    pred_interval = (
                        pred_seg['start_time'].timestamp(),
                        pred_seg['end_time'].timestamp()
                    )
                    pred_dur = pred_seg['duration']
                    
                    # Find best match in this timeline's GT
                    max_overlap = 0
                    best_match_gt = None
                    for gt_seg in gt_segs_class:
                        gt_interval = (
                            gt_seg['start_time'].timestamp(),
                            gt_seg['end_time'].timestamp()
                        )
                        overlap = self.segment_overlap(pred_interval, gt_interval)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_match_gt = gt_seg
                    
                    # Check if this prediction is TP or FP
                    is_valid_match = False
                    if pred_dur > 0 and (max_overlap / pred_dur) >= overlap_threshold:
                        # Check start time threshold if specified
                        if self.start_time_threshold > 0 and best_match_gt is not None:
                            start_time_diff = abs((pred_seg['start_time'] - best_match_gt['start_time']).total_seconds())
                            if start_time_diff <= self.start_time_threshold:
                                is_valid_match = True
                        else:
                            is_valid_match = True
                    
                    # Store label and score
                    y_true.append(1 if is_valid_match else 0)
                    y_scores.append(confidence)
            
            # Calculate ROC curve using sklearn
            if len(y_true) > 0 and len(set(y_true)) > 1:  # Need both classes
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_results[activity_id] = {
                        'auc': roc_auc,
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate ROC for activity {activity_id}: {e}")
                    roc_results[activity_id] = {'auc': None}
            else:
                # Not enough data for ROC curve
                if len(y_true) > 0:
                    print(f"Activity {activity_id}: Only one class present ({len([x for x in y_true if x == 1])} TPs, {len([x for x in y_true if x == 0])} FPs)")
                roc_results[activity_id] = {'auc': None}
        
        return roc_results
    
    def evaluate_multiple_files(self, predicted_files: List[str], ground_truth_files: List[str]) -> Dict:
        """
        Evaluate predictions across multiple files
        
        Args:
            predicted_files: List of paths to prediction files
            ground_truth_files: List of paths to ground truth files
            
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
        
        # Store per-file timelines for cumulative ROC calculation
        all_file_pred_timelines = []
        all_file_gt_timelines = []
        all_file_confidence_timelines = []
        
        # Get activity classes from all files
        activity_classes_set = set()
        
        # Process each file
        for file_idx, (pred_file, gt_file) in enumerate(zip(predicted_files, ground_truth_files)):
            # Parse files
            pred_df = self.parse_log_file(pred_file)
            gt_df = self.parse_log_file(gt_file)
            
            # Extract segments
            pred_segments = self.extract_segments(pred_df)
            gt_segments = self.extract_segments(gt_df)
            
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
            
            # Create timelines for THIS file
            file_pred_timeline, _ = self.create_timeline(pred_segments, file_start_time, file_end_time)
            file_gt_timeline, _ = self.create_timeline(gt_segments, file_start_time, file_end_time)
            
            # Create segment-based confidence timeline for THIS file
            file_conf_timeline = self.create_segment_based_confidence_timeline(
                pred_segments,
                file_start_time,
                file_end_time
            )
            
            # Calculate per-file frame-level ROC
            frame_roc_file = self.calculate_roc_auc_per_class(
                file_pred_timeline,
                file_gt_timeline,
                file_conf_timeline,
                file_classes
            )
            
            # Store timelines for cumulative calculation
            all_file_pred_timelines.append(file_pred_timeline)
            all_file_gt_timelines.append(file_gt_timeline)
            all_file_confidence_timelines.append(file_conf_timeline)
            
            # Calculate segment-level metrics for this file
            file_metrics = self.evaluate_segments(pred_segments, gt_segments)
            
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
        
        # Calculate cumulative segment-level metrics
        cumulative_metrics = self.evaluate_segments(all_pred_segments, all_gt_segments)
        
        # Check if cumulative_metrics is valid
        if not cumulative_metrics or 'per_class' not in cumulative_metrics:
            print("WARNING: Failed to calculate cumulative metrics")
            cumulative_metrics = {
                'per_class': {},
                'macro_avg': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'micro_avg': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'weighted_avg': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            }
        
        # Calculate cumulative segment-level ROC
        segment_roc_cumulative = self.calculate_segment_roc_auc_per_class(
            all_pred_segments,
            all_gt_segments
        )
        
        # Calculate cumulative frame-level ROC
        if len(all_file_pred_timelines) > 0:
            cumulative_pred_timeline = np.concatenate(all_file_pred_timelines)
            cumulative_gt_timeline = np.concatenate(all_file_gt_timelines)
            cumulative_conf_timeline = np.concatenate(all_file_confidence_timelines)
            
            frame_roc_cumulative = self.calculate_roc_auc_per_class(
                cumulative_pred_timeline,
                cumulative_gt_timeline,
                cumulative_conf_timeline,
                activity_classes
            )
        else:
            frame_roc_cumulative = {}
        
        # Prepare results
        results = {
            'metadata': {
                'num_files': len(predicted_files),
                'num_pred_segments': len(all_pred_segments),
                'num_gt_segments': len(all_gt_segments),
                'activity_classes': activity_classes,
                'start_time_threshold': self.start_time_threshold
            },
            'per_class': cumulative_metrics.get('per_class', {}),
            'macro_avg': cumulative_metrics.get('macro_avg', {}),
            'micro_avg': cumulative_metrics.get('micro_avg', {}),
            'weighted_avg': cumulative_metrics.get('weighted_avg', {}),
            'frame_level': frame_roc_cumulative,
            'segment_level': segment_roc_cumulative,
            'per_file_results': per_file_results
        }
        
        return results
    
    def calculate_both_roc_types(self,
                                pred_segments: List[Dict],
                                gt_segments: List[Dict],
                                confidence_scores: List[float],
                                start_time: datetime,
                                end_time: datetime) -> Dict:
        """
        Calculate both frame-level and segment-level ROC curves
        
        Returns:
            Dictionary with both 'frame_level' and 'segment_level' ROC results
        """
        # Create timelines
        pred_timeline, pred_classes = self.create_timeline(pred_segments, start_time, end_time)
        gt_timeline, gt_classes = self.create_timeline(gt_segments, start_time, end_time)
        all_classes = sorted(list(set(pred_classes + gt_classes)))
        
        results = {}
        
        # 1. Frame-level ROC (existing method)
        if confidence_scores is not None and len(confidence_scores) > 0:
            conf_timeline = self.create_confidence_timeline(
                pred_segments, confidence_scores, start_time, end_time
            )
            results['frame_level'] = self.calculate_roc_auc_per_class(
                pred_timeline, gt_timeline, conf_timeline, all_classes
            )
        
        # 2. Segment-level ROC (new method)
        if confidence_scores is not None and len(confidence_scores) > 0:
            results['segment_level'] = self.calculate_segment_based_roc(
                pred_segments, gt_segments, confidence_scores, all_classes
            )
        
        return results
    
    def evaluate_with_dual_roc(self,
                               predicted_file: str,
                               ground_truth_file: str) -> Dict:
        """
        Complete evaluation with both frame-level and segment-level ROC
        
        Args:
            predicted_file: Path to predictions (must include confidence scores)
            ground_truth_file: Path to ground truth
            
        Returns:
            Dictionary with all metrics including both ROC types
        """
        # Parse files
        pred_df = self.parse_log_file(predicted_file)
        gt_df = self.parse_log_file(ground_truth_file)
        
        # Extract segments
        pred_segments = self.extract_segments(pred_df)
        gt_segments = self.extract_segments(gt_df)
        
        if len(pred_segments) == 0 or len(gt_segments) == 0:
            print("Warning: No segments found")
            return {}
        
        # Extract confidence scores from pred_df if available
        confidence_scores = []
        if 'confidence' in pred_df.columns:
            # Group by segments and average confidence
            for seg in pred_segments:
                mask = (pred_df['timestamp'] >= seg['start_time']) & \
                       (pred_df['timestamp'] <= seg['end_time']) & \
                       (pred_df['annotation'] == seg['segment_id'])
                seg_confidences = pred_df[mask]['confidence'].values
                if len(seg_confidences) > 0:
                    confidence_scores.append(np.mean(seg_confidences))
                else:
                    confidence_scores.append(1.0)
        
        # Determine timeline bounds
        all_times = []
        for seg in pred_segments + gt_segments:
            all_times.extend([seg['start_time'], seg['end_time']])
        start_time = min(all_times)
        end_time = max(all_times)
        
        # Create timelines
        pred_timeline, pred_classes = self.create_timeline(pred_segments, start_time, end_time)
        gt_timeline, gt_classes = self.create_timeline(gt_segments, start_time, end_time)
        all_classes = sorted(list(set(pred_classes + gt_classes)))
        
        # Calculate time-continuous metrics
        results = self.calculate_time_continuous_metrics(pred_timeline, gt_timeline, all_classes)
        
        # Calculate both types of ROC
        if len(confidence_scores) > 0:
            roc_results = self.calculate_both_roc_types(
                pred_segments, gt_segments, confidence_scores,
                start_time, end_time
            )
            results.update(roc_results)
        
        # Add metadata
        results['metadata'] = {
            'timeline_start': start_time,
            'timeline_end': end_time,
            'total_duration_seconds': (end_time - start_time).total_seconds(),
            'resolution_seconds': self.resolution,
            'num_pred_segments': len(pred_segments),
            'num_gt_segments': len(gt_segments),
            'activity_classes': all_classes
        }
        
        return results
    
    def plot_dual_roc_curves(self, results: Dict, save_path: str = 'roc_comparison.png', activity_filter=None):
        """
        Plot both frame-level and segment-level ROC curves side by side
        
        Args:
            results: Results dictionary from evaluate_with_dual_roc()
            save_path: Path to save the figure
            activity_filter: List of activity IDs to plot (e.g., ['1', '2', '3', ..., '8'])
        """
        import matplotlib.pyplot as plt
        from itertools import cycle
        
        if 'frame_level' not in results or 'segment_level' not in results:
            print("Both ROC types not available in results")
            return
        
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                    "tab:purple", "tab:brown", "tab:pink", "tab:gray", 
                    "tab:olive", "tab:cyan"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter activity classes
        activity_classes = results['metadata']['activity_classes']
        if activity_filter:
            activity_classes = [a for a in activity_classes if a in activity_filter]
        
        # Left: Frame-level ROC
        colors_iter = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                            "tab:olive", "tab:cyan"])
        for activity_id, color in zip(activity_classes, colors_iter):
            if activity_id in results['frame_level'] and results['frame_level'][activity_id]['auc'] is not None:
                fpr = results['frame_level'][activity_id]['fpr']
                tpr = results['frame_level'][activity_id]['tpr']
                auc_val = results['frame_level'][activity_id]['auc']
                ax1.plot(fpr, tpr, color=color, lw=2, 
                        label=f"Activity {activity_id} (AUC={auc_val:.2f})")
        
        ax1.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Frame-Level ROC (Per-Time-Unit)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Right: Segment-level ROC
        colors_iter = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                            "tab:olive", "tab:cyan"])
        for activity_id, color in zip(activity_classes, colors_iter):
            if activity_id in results['segment_level'] and results['segment_level'][activity_id]['auc'] is not None:
                fpr = results['segment_level'][activity_id]['fpr']
                tpr = results['segment_level'][activity_id]['tpr']
                auc_val = results['segment_level'][activity_id]['auc']
                ax2.plot(fpr, tpr, color=color, lw=2,
                        label=f"Activity {activity_id} (AUC={auc_val:.2f})")
        
        ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Segment-Level ROC (Overlap-Based)')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
        print(f"\nROC curves saved to: {save_path}")
    
    def print_per_file_summary(self, results: Dict):
        """Print summary of per-file results"""
        if 'per_file_results' not in results:
            print("No per-file results available")
            return
        
        print("\n" + "=" * 70)
        print("PER-FILE SUMMARY")
        print("=" * 70)
        
        for idx, file_result in enumerate(results['per_file_results']):
            print(f"\n--- File {idx + 1}: {file_result['pred_file']} ---")
            print(f"Ground Truth: {file_result['gt_file']}")
            print(f"Pred Segments: {file_result['num_pred_segments']}, GT Segments: {file_result['num_gt_segments']}")
            
            # Print AUC scores
            print("\nFrame-Level AUC:")
            for activity_id in sorted(file_result['frame_level'].keys()):
                auc_val = file_result['frame_level'][activity_id].get('auc')
                if auc_val is not None:
                    print(f"  Activity {activity_id}: {auc_val:.3f}")
            
            print("\nSegment-Level AUC:")
            for activity_id in sorted(file_result['segment_level'].keys()):
                auc_val = file_result['segment_level'][activity_id].get('auc')
                if auc_val is not None:
                    print(f"  Activity {activity_id}: {auc_val:.3f}")
            
            # Print key metrics
            if 'micro_avg' in file_result['metrics']:
                micro = file_result['metrics']['micro_avg']
                print(f"\nMicro-Avg: P={micro['precision']:.3f}, R={micro['recall']:.3f}, F1={micro['f1_score']:.3f}")
    
    def plot_per_file_roc_comparison(self, results: Dict, activity_id: str, save_path: str = None):
        """
        Plot ROC curves for a specific activity across all files
        
        Args:
            results: Results from evaluate_multiple_files()
            activity_id: Which activity to plot
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        from itertools import cycle
        
        if 'per_file_results' not in results:
            print("No per-file results available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple", 
                       "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
        
        # Left: Frame-level per file
        for idx, (file_result, color) in enumerate(zip(results['per_file_results'], colors)):
            if activity_id in file_result['frame_level'] and file_result['frame_level'][activity_id]['auc'] is not None:
                fpr = file_result['frame_level'][activity_id]['fpr']
                tpr = file_result['frame_level'][activity_id]['tpr']
                auc_val = file_result['frame_level'][activity_id]['auc']
                ax1.plot(fpr, tpr, color=color, lw=1, alpha=0.5,
                        label=f"File {idx+1} (AUC={auc_val:.2f})")
        
        # Add cumulative
        if activity_id in results['frame_level'] and results['frame_level'][activity_id]['auc'] is not None:
            fpr = results['frame_level'][activity_id]['fpr']
            tpr = results['frame_level'][activity_id]['tpr']
            auc_val = results['frame_level'][activity_id]['auc']
            ax1.plot(fpr, tpr, 'k-', lw=3, label=f"Cumulative (AUC={auc_val:.2f})")
        
        ax1.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'Frame-Level ROC - Activity {activity_id}')
        ax1.legend(loc='best', fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Right: Segment-level per file
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple", 
                       "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
        for idx, (file_result, color) in enumerate(zip(results['per_file_results'], colors)):
            if activity_id in file_result['segment_level'] and file_result['segment_level'][activity_id]['auc'] is not None:
                fpr = file_result['segment_level'][activity_id]['fpr']
                tpr = file_result['segment_level'][activity_id]['tpr']
                auc_val = file_result['segment_level'][activity_id]['auc']
                ax2.plot(fpr, tpr, color=color, lw=1, alpha=0.5,
                        label=f"File {idx+1} (AUC={auc_val:.2f})")
        
        # Add cumulative
        if activity_id in results['segment_level'] and results['segment_level'][activity_id]['auc'] is not None:
            fpr = results['segment_level'][activity_id]['fpr']
            tpr = results['segment_level'][activity_id]['tpr']
            auc_val = results['segment_level'][activity_id]['auc']
            ax2.plot(fpr, tpr, 'k-', lw=3, label=f"Cumulative (AUC={auc_val:.2f})")
        
        ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'Segment-Level ROC - Activity {activity_id}')
        ax2.legend(loc='best', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.step(fpr, tpr, where='post')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        if save_path:
            print(f"\nPer-file ROC curves saved to: {save_path}")
            
    def generate_aggregated_roc_curves(self, all_fold_results, output_dir, activity_filter):
        """
        Generate ROC curves aggregated across all folds
        
        Two approaches:
        1. Pool all predictions from all folds (micro-averaging)
        2. Average ROC curves from each fold (macro-averaging)
        """
        import matplotlib.pyplot as plt
        from itertools import cycle
        from sklearn.metrics import auc
        
        print(f"\nAggregating ROC data from {len(all_fold_results)} folds...")
        
        # Filter activities
        if not activity_filter:
            activity_filter = all_fold_results[0]['metadata']['activity_classes']
        
        # ========================================
        # Method 1: Pooled ROC (Pool all data points from all folds)
        # ========================================
        # This is the most common approach for cross-validation
        
        pooled_frame_roc = {}
        pooled_segment_roc = {}
        
        for activity_id in activity_filter:
            # Collect all FPR/TPR points from all folds
            all_frame_fpr = []
            all_frame_tpr = []
            all_segment_fpr = []
            all_segment_tpr = []
            
            for fold_result in all_fold_results:
                # Frame-level
                if 'frame_level' in fold_result and activity_id in fold_result['frame_level']:
                    frame_data = fold_result['frame_level'][activity_id]
                    if frame_data.get('auc') is not None:
                        all_frame_fpr.extend(frame_data['fpr'])
                        all_frame_tpr.extend(frame_data['tpr'])
                
                # Segment-level
                if 'segment_level' in fold_result and activity_id in fold_result['segment_level']:
                    seg_data = fold_result['segment_level'][activity_id]
                    if seg_data.get('auc') is not None:
                        all_segment_fpr.extend(seg_data['fpr'])
                        all_segment_tpr.extend(seg_data['tpr'])
            
            # Compute pooled ROC by sorting and calculating AUC
            if len(all_frame_fpr) > 1:
                # Sort by FPR
                sorted_pairs = sorted(zip(all_frame_fpr, all_frame_tpr))
                fpr = np.array([p[0] for p in sorted_pairs])
                tpr = np.array([p[1] for p in sorted_pairs])
                
                try:
                    pooled_auc = auc(fpr, tpr)
                    pooled_frame_roc[activity_id] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': pooled_auc
                    }
                except:
                    pass
            
            if len(all_segment_fpr) > 1:
                sorted_pairs = sorted(zip(all_segment_fpr, all_segment_tpr))
                fpr = np.array([p[0] for p in sorted_pairs])
                tpr = np.array([p[1] for p in sorted_pairs])
                
                try:
                    pooled_auc = auc(fpr, tpr)
                    pooled_segment_roc[activity_id] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': pooled_auc
                    }
                except:
                    pass
        
        # ========================================
        # Plot Aggregated ROC
        # ========================================
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                        "tab:purple", "tab:brown", "tab:pink", "tab:gray"])
        
        # Left: Frame-level ROC (pooled)
        for activity_id, color in zip(activity_filter, colors):
            if activity_id in pooled_frame_roc:
                fpr = pooled_frame_roc[activity_id]['fpr']
                tpr = pooled_frame_roc[activity_id]['tpr']
                roc_auc = pooled_frame_roc[activity_id]['auc']
                ax1.plot(fpr, tpr, color=color, lw=2, 
                        label=f"Activity {activity_id} (AUC={roc_auc:.2f})")
        
        ax1.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'Frame-Level ROC (Pooled Across {len(all_fold_results)} Folds)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Right: Segment-level ROC (pooled)
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                        "tab:purple", "tab:brown", "tab:pink", "tab:gray"])
        for activity_id, color in zip(activity_filter, colors):
            if activity_id in pooled_segment_roc:
                fpr = pooled_segment_roc[activity_id]['fpr']
                tpr = pooled_segment_roc[activity_id]['tpr']
                roc_auc = pooled_segment_roc[activity_id]['auc']
                ax2.plot(fpr, tpr, color=color, lw=2,
                        label=f"Activity {activity_id} (AUC={roc_auc:.2f})")
        
        ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'Segment-Level ROC (Pooled Across {len(all_fold_results)} Folds)')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(output_dir, 'roc_all_folds_aggregated.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Aggregated ROC curves saved: {save_path}")
        
        # Also print aggregated AUC summary
        print(f"\n{'='*70}")
        print("AGGREGATED AUC ACROSS ALL FOLDS")
        print(f"{'='*70}")
        
        print("\nFrame-Level AUC:")
        for activity_id in activity_filter:
            if activity_id in pooled_frame_roc:
                auc_val = pooled_frame_roc[activity_id]['auc']
                print(f"  Activity {activity_id}: {auc_val:.4f}")
        
        print("\nSegment-Level AUC:")
        for activity_id in activity_filter:
            if activity_id in pooled_segment_roc:
                auc_val = pooled_segment_roc[activity_id]['auc']
                print(f"  Activity {activity_id}: {auc_val:.4f}")
        
        print(f"{'='*70}\n")
    
    def evaluate(self,
                predicted_file: str,
                ground_truth_file: str,
                confidence_scores: List[float] = None) -> Dict:
        """
        Complete time-continuous evaluation comparing predicted vs ground truth
        
        Args:
            predicted_file: Path to predicted segments log
            ground_truth_file: Path to ground truth log
            confidence_scores: Optional confidence scores for each predicted segment
            
        Returns:
            Dictionary with all evaluation metrics
        """
        
        # Parse files
        pred_df = self.parse_log_file(predicted_file)
        gt_df = self.parse_log_file(ground_truth_file)
        
        # Extract segments
        pred_segments = self.extract_segments(pred_df)
        gt_segments = self.extract_segments(gt_df)
        
        if len(pred_segments) == 0 or len(gt_segments) == 0:
            print("Warning: No segments found in one or both files")
            return {}
        
        # Determine timeline bounds
        all_times = []
        for seg in pred_segments + gt_segments:
            all_times.extend([seg['start_time'], seg['end_time']])
        
        start_time = min(all_times)
        end_time = max(all_times)
        
        # Create timelines
        pred_timeline, pred_classes = self.create_timeline(pred_segments, start_time, end_time)
        gt_timeline, gt_classes = self.create_timeline(gt_segments, start_time, end_time)
        
        # Get all unique activity classes
        all_classes = sorted(list(set(pred_classes + gt_classes)))
        
        # Calculate time-continuous metrics
        results = self.calculate_time_continuous_metrics(pred_timeline, gt_timeline, all_classes)
        
        # Calculate ROC/AUC if confidence scores provided
        if confidence_scores is not None and len(confidence_scores) > 0:
            conf_timeline = self.create_confidence_timeline(pred_segments, confidence_scores, start_time, end_time)
            roc_auc_results = self.calculate_roc_auc_per_class(pred_timeline, gt_timeline, conf_timeline, all_classes)
            results['roc_auc'] = roc_auc_results
        
        # Add metadata
        results['metadata'] = {
            'timeline_start': start_time,
            'timeline_end': end_time,
            'total_duration_seconds': (end_time - start_time).total_seconds(),
            'resolution_seconds': self.resolution,
            'num_pred_segments': len(pred_segments),
            'num_gt_segments': len(gt_segments),
            'activity_classes': all_classes
        }
        
        return results
    
    def evaluate_segments(self,
                         predicted_segments: List[Dict],
                         ground_truth_segments: List[Dict],
                         confidence_scores: List[float] = None) -> Dict:
        """
        Evaluate segments directly without reading files
        
        Args:
            predicted_segments: List of dicts with 'start_time', 'end_time', 'segment_id'
            ground_truth_segments: Same format
            confidence_scores: Optional confidence for each predicted segment
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
        
        # Calculate ROC/AUC if confidence scores provided
        if confidence_scores is not None and len(confidence_scores) > 0:
            conf_timeline = self.create_confidence_timeline(predicted_segments, confidence_scores, start_time, end_time)
            roc_auc_results = self.calculate_roc_auc_per_class(pred_timeline, gt_timeline, conf_timeline, all_classes)
            results['roc_auc'] = roc_auc_results
        
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
    
    def print_report(self, results: Dict, activity_filter=None):
        """Print a formatted evaluation report
        
        Args:
            results: Results dictionary from evaluation
            activity_filter: List of activity IDs to include (e.g., ['1', '2', '3', ..., '8'])
                            If None, all activities are included
        """
        if not results:
            print("No results to display")
            return
        
        print("=" * 70)
        print("TIME-CONTINUOUS MULTI-CLASS EVALUATION REPORT")
        print("=" * 70)
        
        if 'metadata' in results:
            meta = results['metadata']
            
            # Handle both single-file and multi-file metadata
            if 'total_duration_seconds' in meta:
                # Single-file metadata
                print(f"\nTimeline Duration: {meta['total_duration_seconds']:.0f} seconds")
                print(f"Resolution: {meta['resolution_seconds']}s")
            elif 'num_files' in meta:
                # Multi-file metadata
                print(f"\nNumber of files: {meta['num_files']}")
                print(f"Total predicted segments: {meta['num_pred_segments']}")
                print(f"Total ground truth segments: {meta['num_gt_segments']}")
            
            # Filter activity classes if specified
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
            print(f"Precision: {results['macro_avg']['precision']:.4f}")
            print(f"Recall:    {results['macro_avg']['recall']:.4f}")
            print(f"F1 Score:  {results['macro_avg']['f1_score']:.4f}")
        
        if 'micro_avg' in results:
            print("\n--- Micro-Average (pooled TP/FP/FN across classes) ---")
            print(f"Precision: {results['micro_avg']['precision']:.4f}")
            print(f"Recall:    {results['micro_avg']['recall']:.4f}")
            print(f"F1 Score:  {results['micro_avg']['f1_score']:.4f}")
            print(f"Total TP Duration: {results['micro_avg']['total_TP_duration']:.1f}s")
            print(f"Total FP Duration: {results['micro_avg']['total_FP_duration']:.1f}s")
            print(f"Total FN Duration: {results['micro_avg']['total_FN_duration']:.1f}s")
        
        if 'weighted_avg' in results:
            print("\n--- Weighted-Average (weighted by GT duration) ---")
            print(f"Precision: {results['weighted_avg']['precision']:.4f}")
            print(f"Recall:    {results['weighted_avg']['recall']:.4f}")
            print(f"F1 Score:  {results['weighted_avg']['f1_score']:.4f}")
        
        # Per-class metrics - FILTERED
        if 'per_class' in results:
            print("\n" + "=" * 70)
            print("PER-CLASS METRICS")
            print("=" * 70)
            
            # Filter activities
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
                print(f"  Precision:             {metrics['precision']:.4f}")
                print(f"  Recall:                {metrics['recall']:.4f}")
                print(f"  F1 Score:              {metrics['f1_score']:.4f}")
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

# Example usage
if __name__ == "__main__":
    
    # Initialize evaluator
    evaluator = SegmentEvaluator(timeline_resolution_seconds=1.0, start_time_threshold_seconds=0)
    
    # Example 1: Basic evaluation (backward compatible)
    print("=" * 70)
    print("EXAMPLE 1: BASIC EVALUATION")
    print("=" * 70)
    results = evaluator.evaluate(
        predicted_file='predicted.txt',
        ground_truth_file='ground_truth.txt'
    )
    evaluator.print_report(results)
    
    print("\n\n")
    
    # Example 2: Evaluation with dual ROC curves (new feature)
    print("=" * 70)
    print("EXAMPLE 2: EVALUATION WITH DUAL ROC CURVES - SINGLE FILE")
    print("=" * 70)
    
    results_with_roc = evaluator.evaluate_with_dual_roc(
        predicted_file='predicted_92_P080.txt',
        ground_truth_file='P080.txt'
    )
    
    evaluator.print_report(results_with_roc)
    
    # Plot both types of ROC curves
    evaluator.plot_dual_roc_curves(results_with_roc, save_path='dual_roc_curves.png')
    
    print("\n\n")
    
    # Example 3: Evaluation across multiple files (PROPER WAY FOR 100 FILES)
    print("=" * 70)
    print("EXAMPLE 3: MULTI-FILE EVALUATION (100+ FILES)")
    print("=" * 70)
    
    # Your file lists
    pred_files = ['predicted_92_P080.txt', 'predicted_92_P081.txt']  # ... add all 100 files
    gt_files = ['P080.txt', 'P081.txt']  # ... add all 100 files
    
    results_multi = evaluator.evaluate_multiple_files(pred_files, gt_files)
    
    # Print cumulative report
    print("\n--- CUMULATIVE RESULTS ---")
    evaluator.print_report(results_multi)
    
    # Print per-file summary
    evaluator.print_per_file_summary(results_multi)
    
    # Plot cumulative ROC curves
    evaluator.plot_dual_roc_curves(results_multi, save_path='cumulative_roc.png')
    
    # Plot per-file comparison for a specific activity
    evaluator.plot_per_file_roc_comparison(results_multi, activity_id='1', 
                                           save_path='per_file_activity_1_roc.png')