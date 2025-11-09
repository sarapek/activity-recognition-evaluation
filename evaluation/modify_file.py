import random
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Tuple
import re

class SanityCheckModifier:
    """
    Modifies sensor data in controlled ways to test segment evaluation system.
    Creates various types of errors to validate evaluation metrics.
    Now includes confidence scores for ROC curve testing.
    Uses segment-based approach for continuous predictions.
    """
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        self.modifications_log = []
    
    def read_file(self, filepath: str) -> List[str]:
        """Read file and return lines"""
        with open(filepath, 'r') as f:
            return f.readlines()
    
    def _normalize_activity(self, activity: str) -> str:
        """Extract numeric activity ID from labels like '2-start', '2.1', etc."""
        match = re.match(r'(\d+)', str(activity))
        if match:
            return match.group(1)
        return str(activity)
    
    def _generate_confidence_score(self, line: str, parts: List[str], variant_name: str = 'unknown') -> float:
        """
        Generate confidence scores based on variant type
        
        Args:
            line: Input line (can be None)
            parts: Parsed line parts (can be None)
            variant_name: Name of the variant (e.g., 'perfect', 'label_confusion')
        
        Returns:
            Confidence score between 0 and 1
        """
        variant_lower = variant_name.lower()
        
        if 'perfect' in variant_lower:
            # Perfect predictions - HIGH confidence
            return np.random.uniform(0.85, 0.95)
        
        elif 'label_confusion' in variant_lower or 'random' in variant_lower:
            # Wrong labels - LOW confidence
            return np.random.uniform(0.15, 0.30)
        
        elif 'shift' in variant_lower or 'boundary' in variant_lower:
            # Timing errors - MEDIUM confidence
            return np.random.uniform(0.50, 0.70)
        
        elif 'false_positive' in variant_lower:
            # False positives - MEDIUM-LOW confidence
            return np.random.uniform(0.30, 0.50)
        
        elif 'missing' in variant_lower:
            # Missing segments - MEDIUM-HIGH confidence
            return np.random.uniform(0.65, 0.80)
        
        elif 'combined' in variant_lower:
            # Combined errors - MEDIUM confidence
            return np.random.uniform(0.40, 0.60)
        
        else:
            # Default
            return np.random.uniform(0.45, 0.65)
    
    def _extract_segments_from_lines(self, lines: List[str]) -> List[dict]:
        """
        Extract segments from ground truth lines
        Groups consecutive events with same activity into segments
        """
        segments = []
        current_segment = None
        
        for line in lines:
            parts = line.strip().split('\t')
            
            if len(parts) < 4 or not parts[3].strip():
                continue
            
            timestamp_str = parts[0]
            sensor = parts[1]
            activity_raw = parts[3]
            
            # CRITICAL: Skip error annotations
            if 'error' in activity_raw.lower():
                continue
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            
            # Extract numeric activity
            activity = self._normalize_activity(activity_raw)
            
            # Check if this continues the current segment or starts a new one
            if current_segment is None or current_segment['activity'] != activity:
                # Save previous segment
                if current_segment is not None:
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'activity': activity,
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'sensor': sensor,
                    'events': [{'timestamp': timestamp, 'sensor': sensor}]
                }
            else:
                # Extend current segment
                current_segment['end_time'] = timestamp
                current_segment['events'].append({'timestamp': timestamp, 'sensor': sensor})
        
        # Don't forget last segment
        if current_segment is not None:
            segments.append(current_segment)
        
        return segments
    
    def create_continuous_predictions_from_gt(self, gt_file: str, variant_name: str, 
                                      resolution_seconds: float = None) -> List[str]:
        """
        Generate predictions from ground truth by annotating ALL events between start/end markers
        Uses existing timestamps from GT file, doesn't create new ones
        """
        gt_lines = self.read_file(gt_file)
        
        # First pass: identify segment boundaries
        segment_boundaries = []
        current_activity = None
        segment_start = None
        
        for line in gt_lines:
            parts = line.strip().split()
            
            if len(parts) < 5:
                continue
            
            timestamp_str = f"{parts[0]} {parts[1]}"
            activity_raw = parts[4] if len(parts) > 4 else ""
            
            # Skip if no activity annotation
            if not activity_raw or 'error' in activity_raw.lower():
                continue
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            
            # Check for start/end markers
            if '-start' in activity_raw.lower():
                activity_id = self._normalize_activity(activity_raw.split('-start')[0])
                if current_activity is not None and segment_start is not None:
                    # Save previous segment
                    segment_boundaries.append({
                        'activity': current_activity,
                        'start': segment_start,
                        'end': timestamp
                    })
                current_activity = activity_id
                segment_start = timestamp
            
            elif '-end' in activity_raw.lower():
                activity_id = self._normalize_activity(activity_raw.split('-end')[0])
                if current_activity == activity_id and segment_start is not None:
                    segment_boundaries.append({
                        'activity': current_activity,
                        'start': segment_start,
                        'end': timestamp
                    })
                    current_activity = None
                    segment_start = None
        
        # Second pass: annotate ALL events that fall within segments
        predictions = []
        
        for line in gt_lines:
            parts = line.strip().split()
            
            if len(parts) < 4:
                continue
            
            timestamp_str = f"{parts[0]} {parts[1]}"
            sensor = parts[2]
            status = parts[3]
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            
            # Check which segment this event falls into
            activity = None
            for seg in segment_boundaries:
                if seg['start'] <= timestamp <= seg['end']:
                    activity = seg['activity']
                    break
            
            # Only create prediction if event is within a segment
            if activity is not None:
                confidence = self._generate_confidence_score(None, None, variant_name)
                predictions.append(f"{timestamp_str} {sensor} {sensor} {status} {activity} {confidence:.6f}")
        
        return predictions
    
    def create_continuous_predictions_with_time_shift(self, gt_file: str, variant_name: str,
                                                      shift_range: Tuple[float, float],
                                                      resolution_seconds: float = 1.0) -> List[str]:
        """Generate predictions with shifted timestamps"""
        gt_lines = self.read_file(gt_file)
        segments = self._extract_segments_from_lines(gt_lines)
        
        predictions = []
        
        for seg in segments:
            activity = seg['activity']
            start_time = seg['start_time']
            end_time = seg['end_time']
            sensor = seg['sensor']
            
            # Apply random shift to this segment
            shift = random.uniform(shift_range[0], shift_range[1])
            shifted_start = start_time + timedelta(seconds=shift)
            shifted_end = end_time + timedelta(seconds=shift)
            
            current_time = shifted_start
            while current_time <= shifted_end:
                confidence = self._generate_confidence_score(None, None, variant_name)
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{timestamp_str} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current_time += timedelta(seconds=resolution_seconds)
        
        return predictions
    
    def create_continuous_predictions_with_boundary_shift(self, gt_file: str, variant_name: str,
                                                          shift_range: Tuple[float, float],
                                                          resolution_seconds: float = 1.0) -> List[str]:
        """Generate predictions with shifted segment boundaries"""
        gt_lines = self.read_file(gt_file)
        segments = self._extract_segments_from_lines(gt_lines)
        
        predictions = []
        
        for seg in segments:
            activity = seg['activity']
            start_time = seg['start_time']
            end_time = seg['end_time']
            sensor = seg['sensor']
            
            # Shift start and end independently
            start_shift = random.uniform(shift_range[0], shift_range[1])
            end_shift = random.uniform(shift_range[0], shift_range[1])
            
            shifted_start = start_time + timedelta(seconds=start_shift)
            shifted_end = end_time + timedelta(seconds=end_shift)
            
            # Ensure end is after start
            if shifted_end <= shifted_start:
                shifted_end = shifted_start + timedelta(seconds=1.0)
            
            current_time = shifted_start
            while current_time <= shifted_end:
                confidence = self._generate_confidence_score(None, None, variant_name)
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{timestamp_str} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current_time += timedelta(seconds=resolution_seconds)
        
        return predictions
    
    def create_continuous_predictions_with_missing_segments(self, gt_file: str, variant_name: str,
                                                            missing_percent: float = 30,
                                                            resolution_seconds: float = 1.0) -> List[str]:
        """Generate predictions but randomly skip entire segments"""
        gt_lines = self.read_file(gt_file)
        segments = self._extract_segments_from_lines(gt_lines)
        
        # Randomly select segments to keep
        num_to_keep = int(len(segments) * (1 - missing_percent / 100))
        kept_segments = random.sample(segments, num_to_keep)
        
        predictions = []
        
        for seg in kept_segments:
            activity = seg['activity']
            start_time = seg['start_time']
            end_time = seg['end_time']
            sensor = seg['sensor']
            
            current_time = start_time
            while current_time <= end_time:
                confidence = self._generate_confidence_score(None, None, variant_name)
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{timestamp_str} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current_time += timedelta(seconds=resolution_seconds)
        
        # Sort by timestamp
        predictions.sort()
        
        return predictions
    
    def create_continuous_predictions_with_false_positives(self, gt_file: str, variant_name: str,
                                                       num_false_segments: int = 10,
                                                       resolution_seconds: float = 1.0) -> List[str]:
        """Generate predictions with additional false positive segments"""
        gt_lines = self.read_file(gt_file)
        segments = self._extract_segments_from_lines(gt_lines)
        
        all_activities = list(set(seg['activity'] for seg in segments))
        predictions = []
        
        # Generate predictions for REAL segments with HIGH confidence (correct predictions)
        for seg in segments:
            activity = seg['activity']
            start_time = seg['start_time']
            end_time = seg['end_time']
            sensor = seg['sensor']
            
            current_time = start_time
            while current_time <= end_time:
                # HIGH confidence for correct predictions
                confidence = np.random.uniform(0.85, 0.95)  # ← Changed!
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{timestamp_str} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current_time += timedelta(seconds=resolution_seconds)
        
        # Add FALSE positive segments with LOW confidence
        if len(segments) > 1:
            for _ in range(num_false_segments):
                seg_idx = random.randint(0, len(segments) - 2)
                gap_start = segments[seg_idx]['end_time'] + timedelta(seconds=1)
                gap_end = segments[seg_idx + 1]['start_time'] - timedelta(seconds=1)
                
                if gap_end > gap_start:
                    false_activity = random.choice(all_activities)
                    false_sensor = segments[seg_idx]['sensor']
                    
                    duration = min((gap_end - gap_start).total_seconds(), 5.0)
                    false_end = gap_start + timedelta(seconds=duration)
                    
                    current_time = gap_start
                    while current_time <= false_end:
                        # LOW confidence for false predictions
                        confidence = np.random.uniform(0.30, 0.50)  # ← Changed!
                        timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                        predictions.append(f"{timestamp_str} {false_sensor} {false_sensor} ON {false_activity} {confidence:.6f}")
                        current_time += timedelta(seconds=resolution_seconds)
        
        predictions.sort()
        return predictions
    
    def create_continuous_predictions_with_label_swap(self, gt_file: str, variant_name: str,
                                                  swap_percent: float = 40,
                                                  resolution_seconds: float = 1.0) -> List[str]:
        """Generate predictions with swapped activity labels"""
        gt_lines = self.read_file(gt_file)
        segments = self._extract_segments_from_lines(gt_lines)
        
        all_activities = list(set(seg['activity'] for seg in segments))
        
        if len(all_activities) < 2:
            return self.create_continuous_predictions_from_gt(gt_file, variant_name, resolution_seconds)
        
        # Randomly select segments to swap
        num_to_swap = int(len(segments) * (swap_percent / 100))
        segments_to_swap = random.sample(range(len(segments)), num_to_swap)
        
        predictions = []
        
        for idx, seg in enumerate(segments):
            # Determine if this segment is swapped
            is_swapped = idx in segments_to_swap
            
            if is_swapped:
                # Swap activity
                other_activities = [a for a in all_activities if a != seg['activity']]
                activity = random.choice(other_activities)
                # LOW confidence for incorrect predictions
                confidence_range = (0.15, 0.30)  # ← Changed!
            else:
                activity = seg['activity']
                # HIGH confidence for correct predictions
                confidence_range = (0.85, 0.95)  # ← Changed!
            
            start_time = seg['start_time']
            end_time = seg['end_time']
            sensor = seg['sensor']
            
            current_time = start_time
            while current_time <= end_time:
                confidence = np.random.uniform(*confidence_range)
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{timestamp_str} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current_time += timedelta(seconds=resolution_seconds)
        
        return predictions
    
    def create_sanity_check_variants(self, input_file: str, output_dir: str):
        """
        Create sanity check variants with continuous predictions
        Each variant tests different types of errors
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        variants = {
            'perfect': {
                'description': 'Perfect predictions with high confidence (0.85-0.95)',
                'generator': lambda: self.create_continuous_predictions_from_gt(
                    input_file, 'perfect', resolution_seconds=None
                )
            },
            'small_time_shift': {
                'description': 'Small timestamp shifts (1-3s), medium-high confidence (0.50-0.70)',
                'generator': lambda: self.create_continuous_predictions_with_time_shift(
                    input_file, 'small_time_shift', shift_range=(1.0, 3.0), resolution_seconds=1.0
                )
            },
            'large_time_shift': {
                'description': 'Large timestamp shifts (10-20s), medium confidence (0.50-0.70)',
                'generator': lambda: self.create_continuous_predictions_with_time_shift(
                    input_file, 'large_time_shift', shift_range=(10.0, 20.0), resolution_seconds=1.0
                )
            },
            'boundary_errors': {
                'description': 'Shifted segment boundaries (5-12s), medium confidence (0.50-0.70)',
                'generator': lambda: self.create_continuous_predictions_with_boundary_shift(
                    input_file, 'boundary_errors', shift_range=(5.0, 12.0), resolution_seconds=1.0
                )
            },
            'missing_segments': {
                'description': '30% of segments removed (false negatives), medium-high confidence (0.65-0.80)',
                'generator': lambda: self.create_continuous_predictions_with_missing_segments(
                    input_file, 'missing_segments', missing_percent=30, resolution_seconds=1.0
                )
            },
            'false_positives': {
                'description': '10 false segments added, low-medium confidence (0.30-0.50)',
                'generator': lambda: self.create_continuous_predictions_with_false_positives(
                    input_file, 'false_positives', num_false_segments=10, resolution_seconds=1.0
                )
            },
            'label_confusion': {
                'description': '40% of segment labels swapped, low confidence (0.15-0.30)',
                'generator': lambda: self.create_continuous_predictions_with_label_swap(
                    input_file, 'label_confusion', swap_percent=40, resolution_seconds=1.0
                )
            },
            'combined_errors': {
                'description': 'Multiple error types combined, medium confidence (0.40-0.60)',
                'generator': lambda: self._create_combined_errors(input_file)
            }
        }
        
        # Generate and write each variant
        summary = []
        for variant_name, variant_info in variants.items():
            print(f"Creating {variant_name}...")
            
            # Generate predictions
            predictions = variant_info['generator']()
            
            # Write to file
            output_file = os.path.join(output_dir, f"{base_name}_{variant_name}.txt")
            with open(output_file, 'w') as f:
                f.write('\n'.join(predictions))
            
            print(f"  Created {len(predictions)} predictions")
            
            summary.append(f"{variant_name:20s} -> {output_file}")
            summary.append(f"{'':20s}    {variant_info['description']}")
        
        # Write summary file
        summary_file = os.path.join(output_dir, "README.txt")
        with open(summary_file, 'w') as f:
            f.write("SANITY CHECK VARIANTS WITH CONFIDENCE SCORES\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Original file: {input_file}\n")
            f.write(f"Created {len(variants)} variants for testing\n\n")
            f.write("All variants include confidence scores for ROC curve testing.\n")
            f.write("Each variant has continuous predictions (every second annotated).\n")
            f.write("Confidence scores reflect prediction quality:\n")
            f.write("  - High (0.85-0.95): Perfect predictions\n")
            f.write("  - Medium (0.50-0.70): Timing errors\n")
            f.write("  - Low (0.15-0.30): Label errors\n\n")
            f.write("\n".join(summary))
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("Usage: Compare each variant against the original file\n")
            f.write("using the SegmentEvaluator to validate metrics and ROC curves.\n")
        
        print(f"\nCreated {len(variants)} sanity check variants in: {output_dir}")
        print(f"All files include continuous predictions with confidence scores")
        print(f"See {summary_file} for details")
        
        return variants
    
    def _create_combined_errors(self, gt_file: str) -> List[str]:
        """Create variant with multiple types of errors"""
        gt_lines = self.read_file(gt_file)
        segments = self._extract_segments_from_lines(gt_lines)
        
        all_activities = list(set(seg['activity'] for seg in segments))
        predictions = []
        
        for idx, seg in enumerate(segments):
            activity = seg['activity']
            start_time = seg['start_time']
            end_time = seg['end_time']
            sensor = seg['sensor']
            
            # Randomly apply different modifications
            modification = random.choice(['none', 'shift', 'boundary', 'label_swap', 'skip'])
            
            # Determine confidence based on modification type
            if modification == 'skip':
                continue  # Missing segment - no predictions
            
            elif modification == 'none':
                # Correct prediction - HIGH confidence
                confidence_range = (0.85, 0.95)
            
            elif modification in ['shift', 'boundary']:
                # Time shift - MEDIUM confidence (labels correct, timing wrong)
                confidence_range = (0.50, 0.70)
            
            elif modification == 'label_swap':
                # Wrong label - LOW confidence
                confidence_range = (0.15, 0.30)
                # Swap label
                if len(all_activities) > 1:
                    other_activities = [a for a in all_activities if a != activity]
                    activity = random.choice(other_activities)
            
            # Apply time modifications
            if modification == 'shift':
                shift = random.uniform(-3.0, 3.0)
                start_time += timedelta(seconds=shift)
                end_time += timedelta(seconds=shift)
            
            elif modification == 'boundary':
                start_shift = random.uniform(-2.0, 2.0)
                end_shift = random.uniform(-2.0, 2.0)
                start_time += timedelta(seconds=start_shift)
                end_time += timedelta(seconds=end_shift)
            
            # Generate predictions with appropriate confidence
            if end_time > start_time:
                current_time = start_time
                while current_time <= end_time:
                    confidence = np.random.uniform(*confidence_range)  # ← Use the range!
                    timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                    predictions.append(f"{timestamp_str} {sensor} {sensor} ON {activity} {confidence:.6f}")
                    current_time += timedelta(seconds=1.0)
        
        # Sort by timestamp
        predictions.sort()
        
        return predictions

# Example usage
if __name__ == "__main__":
    
    modifier = SanityCheckModifier(seed=42)
    
    # Create comprehensive sanity check variants with confidence scores
    input_file = "../data/P001.txt"
    variants = modifier.create_sanity_check_variants(input_file, "../SanityCheck")
    
    print("\n" + "="*70)
    print("SANITY CHECK FILES CREATED WITH CONTINUOUS PREDICTIONS")
    print("="*70)
    print("\nTo test your SegmentEvaluator with ROC curves:")
    print("\n  from segment_evaluator import SegmentEvaluator")
    print("  evaluator = SegmentEvaluator()")
    print("\n  # Test with ROC curves")
    print("  results = evaluator.evaluate_with_dual_roc(")
    print("      '../SanityCheck/P001_perfect.txt',")
    print("      '../data/P001.txt'")
    print("  )")
    print("\n  # Plot ROC curves")
    print("  evaluator.plot_dual_roc_curves(results, 'perfect_roc.png')")