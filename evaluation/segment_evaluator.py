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
    
    def __init__(self, timeline_resolution_seconds=1.0, start_time_threshold_seconds=0, overlap_threshold=0.5):
        """
        Args:
            timeline_resolution_seconds: Time resolution for timeline (default 1.0 = per second)
            start_time_threshold_seconds: Minimum time from start to consider valid (default 0)
        """
        self.resolution = timeline_resolution_seconds
        self.start_time_threshold = start_time_threshold_seconds
        self.overlap_threshold = overlap_threshold
        
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

    def extract_segments(self, df: pd.DataFrame, confidence_threshold: float = 0.0) -> List[Dict]:
        """Extract segments from annotations"""
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
        
        # Macro-average
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
        
        # Micro-average
        total_TP = sum(m['TP_duration'] for m in per_class_metrics.values())
        total_FP = sum(m['FP_duration'] for m in per_class_metrics.values())
        total_FN = sum(m['FN_duration'] for m in per_class_metrics.values())
        
        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Weighted-average
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
                    print(f"⚠️ WARNING: Predictions and GT are in completely different time periods!")
            
            print(f"\n{'='*80}")
            print(f"ACTIVITY {activity_id} - OVERLAP THRESHOLD = {overlap_threshold}")
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
                    overlap_start = max(gt_seg['start_time'], pred_seg['start_time'])
                    overlap_end = min(gt_seg['end_time'], pred_seg['end_time'])
                    
                    if overlap_start < overlap_end:
                        overlap_duration = (overlap_end - overlap_start).total_seconds()
                        pred_duration = pred_seg['duration']
                        
                        if gt_duration is not None and gt_duration > 0:
                            overlap_ratio = overlap_duration / gt_duration
                            
                            confidence = pred_seg.get('confidence', 0.5)
                            if confidence is None:
                                confidence = 0.5
                            
                            # Store this overlap for debugging
                            overlaps_found.append({
                                'pred_idx': pred_idx,
                                'pred_start': pred_seg['start_time'],
                                'pred_end': pred_seg['end_time'],
                                'pred_duration': pred_duration,
                                'overlap_duration': overlap_duration,
                                'overlap_ratio': overlap_ratio,
                                'confidence': confidence,
                                'meets_threshold': overlap_ratio >= overlap_threshold
                            })
                            
                            if overlap_ratio >= overlap_threshold:
                                if overlap_ratio > best_overlap_ratio or \
                                (overlap_ratio == best_overlap_ratio and confidence > best_confidence):
                                    best_confidence = confidence
                                    best_overlap_ratio = overlap_ratio
                                    best_pred_idx = pred_idx
                
                # Print all overlaps for this GT segment
                if len(overlaps_found) > 0:
                    print(f"  Found {len(overlaps_found)} overlapping predictions:")
                    for i, ov in enumerate(overlaps_found[:5]):  # Show first 5
                        meets = "✓ MATCH" if ov['meets_threshold'] else "✗ below threshold"
                        print(f"    Pred {ov['pred_idx']}: {ov['pred_start'].strftime('%H:%M:%S')}->{ov['pred_end'].strftime('%H:%M:%S')} "
                            f"overlap={ov['overlap_duration']:.1f}s ratio={ov['overlap_ratio']:.4f} conf={ov['confidence']:.3f} {meets}")
                    if len(overlaps_found) > 5:
                        print(f"    ... and {len(overlaps_found) - 5} more")
                else:
                    print(f"  No overlapping predictions found")
                
                # Record result
                if best_overlap_ratio >= overlap_threshold:
                    print(f"  → BEST MATCH: Pred {best_pred_idx}, ratio={best_overlap_ratio:.4f}, conf={best_confidence:.3f} ✓ TP")
                    y_true.append(1)
                    y_scores.append(best_confidence)
                    matched_pred_indices.add(best_pred_idx)
                else:
                    print(f"  → NO MATCH (best ratio={best_overlap_ratio:.4f} < threshold={overlap_threshold}) ✗ FN")
                    y_true.append(0)
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
                        
            if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    print(f"  → AUC: {roc_auc:.4f}")
                    
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
                print(f"  → Not enough variation for AUC")
                roc_auc_results[activity_id] = {'auc': None}
        
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
            if idx < len(pred_df) - 1:
                next_timestamp = pred_df.iloc[idx + 1]['timestamp']
                next_elapsed = (next_timestamp - start_time).total_seconds()
                end_frame = int(next_elapsed / self.resolution)
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
            y_true = (gt_timeline == activity_id).astype(int)
            
            # Get confidence scores where this class was predicted
            y_scores = np.zeros(len(pred_confidence_timeline))
            for i in range(len(pred_confidence_timeline)):
                if pred_timeline[i] == activity_id:
                    y_scores[i] = pred_confidence_timeline[i]
            
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
                                aggregation='average') -> Dict:
        """
        Evaluate with both TRUE frame-level and segment-level ROC curves
        """
        # Parse files
        pred_df = self.parse_log_file(predicted_file)
        gt_df = self.parse_log_file(ground_truth_file)
        
        # Extract segments
        pred_segments = self.extract_segments(pred_df)
        gt_segments = self.extract_segments(gt_df)
        
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
        
        # Calculate standard segment-based metrics
        metrics = self.evaluate_segments(pred_segments, gt_segments)
        
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
        
        # Combine results
        results = {
            'per_class': metrics.get('per_class', {}),
            'macro_avg': metrics.get('macro_avg', {}),
            'micro_avg': metrics.get('micro_avg', {}),
            'weighted_avg': metrics.get('weighted_avg', {}),
            'frame_level': frame_roc,
            'segment_level': segment_roc,
            'metadata': {
                'timeline_start': start_time,
                'timeline_end': end_time,
                'total_duration_seconds': (end_time - start_time).total_seconds(),
                'resolution_seconds': self.resolution,
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
        
        if 'frame_level' not in results or 'segment_level' not in results:
            print("Both ROC types not available in results")
            return
        
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
            'macro_avg': cumulative_metrics.get('macro_avg', {}),
            'micro_avg': cumulative_metrics.get('micro_avg', {}),
            'weighted_avg': cumulative_metrics.get('weighted_avg', {}),
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
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                        "tab:purple", "tab:brown", "tab:pink", "tab:gray"])
        
        # Frame-level
        for activity_id, color in zip(activity_filter, colors):
            if activity_id in averaged_frame_roc:
                fpr = averaged_frame_roc[activity_id]['fpr']
                tpr = averaged_frame_roc[activity_id]['tpr']
                mean_auc = averaged_frame_roc[activity_id]['auc']
                std_auc = averaged_frame_roc[activity_id]['auc_std']
                
                ax1.plot(fpr, tpr, color=color, lw=2,
                        label=f"Activity {activity_id} (AUC={mean_auc:.2f}±{std_auc:.2f})")
        
        ax1.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'Frame-Level ROC (Averaged Across {len(all_fold_results)} Folds)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Segment-level
        colors = cycle(["tab:blue", "tab:green", "tab:red", "tab:orange", 
                        "tab:purple", "tab:brown", "tab:pink", "tab:gray"])
        for activity_id, color in zip(activity_filter, colors):
            if activity_id in averaged_segment_roc:
                fpr = averaged_segment_roc[activity_id]['fpr']
                tpr = averaged_segment_roc[activity_id]['tpr']
                mean_auc = averaged_segment_roc[activity_id]['auc']
                std_auc = averaged_segment_roc[activity_id]['auc_std']
                
                ax2.plot(fpr, tpr, color=color, lw=2,
                        label=f"Activity {activity_id} (AUC={mean_auc:.2f}±{std_auc:.2f})")
        
        ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'Segment-Level ROC (Averaged Across {len(all_fold_results)} Folds)')
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