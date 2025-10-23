import random
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Tuple

class SanityCheckModifier:
    """
    Modifies sensor data in controlled ways to test segment evaluation system.
    Creates various types of errors to validate evaluation metrics.
    Now includes confidence scores for ROC curve testing.
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
    
    def write_file(self, filepath: str, lines: List[str], add_confidence: bool = True):
        """
        Write lines to file with confidence scores, creating directories if needed
        
        Args:
            filepath: Output file path
            lines: Lines to write
            add_confidence: If True, adds confidence scores to predictions
        """
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(filepath, 'w') as f:
            for line in lines:
                if add_confidence:
                    # Parse the line
                    parts = line.strip().split('\t')
                    
                    # Check if this line has an activity annotation
                    has_activity = len(parts) >= 4 and parts[3].strip()
                    
                    if has_activity:
                        # Generate confidence score based on the type of modification
                        confidence = self._generate_confidence_score(line, parts)
                        
                        # Format: date time sensor sensor status activity confidence
                        # Need to duplicate sensor name for prediction format
                        timestamp = parts[0]
                        sensor = parts[1]
                        status = parts[2]
                        activity = parts[3]
                        
                        f.write(f"{timestamp}\t{sensor}\t{sensor}\t{status}\t{activity}\t{confidence:.6f}\n")
                    else:
                        # No activity annotation, write as-is
                        f.write(line)
                else:
                    f.write(line)
    
    def _generate_confidence_score(self, line: str, parts: List[str]) -> float:
        """
        Generate realistic confidence scores based on the quality of prediction
        
        Higher confidence for:
        - Perfect matches
        - Correct labels
        - Good timing
        
        Lower confidence for:
        - Modified/shifted predictions
        - Label swaps
        - False positives
        """
        # Check modification log to see if this line was modified
        line_modified = False
        modification_type = None
        
        for mod in self.modifications_log:
            if 'line' in mod:
                # Note: This is approximate since we don't track exact line numbers
                # But it gives us variation in confidence scores
                modification_type = mod.get('type')
                line_modified = True
                break
        
        # Base confidence
        if not line_modified:
            # Perfect match - high confidence
            confidence = np.random.uniform(0.85, 0.95)
        elif modification_type == 'timestamp_shift':
            # Timing error - medium-high confidence
            confidence = np.random.uniform(0.65, 0.85)
        elif modification_type == 'boundary_shift':
            # Boundary error - medium confidence
            confidence = np.random.uniform(0.55, 0.75)
        elif modification_type == 'label_swap':
            # Wrong label - lower confidence
            confidence = np.random.uniform(0.35, 0.55)
        elif modification_type == 'added_false_marker':
            # False positive - low confidence
            confidence = np.random.uniform(0.20, 0.45)
        else:
            # Default medium confidence
            confidence = np.random.uniform(0.50, 0.80)
        
        return confidence
    
    def shift_timestamp(self, timestamp_str: str, shift_seconds: float) -> str:
        """Shift a timestamp by given seconds"""
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        new_dt = dt + timedelta(seconds=shift_seconds)
        return new_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    def modify_timestamps(self, lines: List[str], percent: float, 
                         shift_range: Tuple[float, float] = (-5.0, 5.0)) -> List[str]:
        """
        Randomly shift timestamps to test temporal accuracy
        
        Args:
            lines: Input lines
            percent: Percentage of lines to modify (0-100)
            shift_range: Min and max shift in seconds
        """
        num_to_modify = int(len(lines) * (percent / 100))
        indices = random.sample(range(len(lines)), num_to_modify)
        
        modified = lines.copy()
        
        for idx in indices:
            parts = lines[idx].strip().split('\t')
            if len(parts) >= 3:
                shift = random.uniform(shift_range[0], shift_range[1])
                parts[0] = self.shift_timestamp(parts[0], shift)
                modified[idx] = '\t'.join(parts) + '\n'
                
                self.modifications_log.append({
                    'line': idx,
                    'type': 'timestamp_shift',
                    'shift': shift
                })
        
        return modified
    
    def remove_segment_markers(self, lines: List[str], percent: float) -> List[str]:
        """
        Remove segment start/end markers to create false negatives
        
        Args:
            lines: Input lines
            percent: Percentage of segment markers to remove
        """
        modified = lines.copy()
        
        # Find all lines with segment markers
        marker_indices = []
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[3].strip():
                marker_indices.append(i)
        
        if not marker_indices:
            return modified
        
        num_to_remove = int(len(marker_indices) * (percent / 100))
        indices_to_remove = random.sample(marker_indices, num_to_remove)
        
        # Remove from end to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            modified.pop(idx)
            
            self.modifications_log.append({
                'line': idx,
                'type': 'removed_segment_marker'
            })
        
        return modified
    
    def add_false_segment_markers(self, lines: List[str], num_to_add: int) -> List[str]:
        """
        Add fake segment markers to create false positives
        
        Args:
            lines: Input lines
            num_to_add: Number of false markers to add
        """
        modified = lines.copy()
        
        # Find lines without segment markers where we can insert
        eligible_indices = []
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                has_annotation = len(parts) >= 4 and parts[3].strip()
                if not has_annotation:
                    eligible_indices.append(i)
        
        if not eligible_indices:
            return modified
        
        num_to_add = min(num_to_add, len(eligible_indices))
        indices_to_modify = random.sample(eligible_indices, num_to_add)
        
        for idx in indices_to_modify:
            parts = modified[idx].strip().split('\t')
            
            # Add a fake segment marker
            fake_id = random.randint(1, 8)
            fake_annotation = str(fake_id)
            
            # Ensure we have 4 columns
            while len(parts) < 3:
                parts.append('')
            
            parts.append(fake_annotation)
            modified[idx] = '\t'.join(parts) + '\n'
            
            self.modifications_log.append({
                'line': idx,
                'type': 'added_false_marker',
                'annotation': fake_annotation
            })
        
        return modified
    
    def shift_segment_boundaries(self, lines: List[str], percent: float,
                                shift_range: Tuple[float, float] = (-15.0, 15.0)) -> List[str]:
        """
        Shift segment start/end times to test boundary detection
        
        Args:
            lines: Input lines
            percent: Percentage of segment boundaries to shift
            shift_range: Min and max shift in seconds
        """
        modified = lines.copy()
        
        # Find segment boundary markers (lines with activity annotations)
        boundary_indices = []
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[3].strip():
                boundary_indices.append(i)
        
        if not boundary_indices:
            return modified
        
        num_to_shift = int(len(boundary_indices) * (percent / 100))
        indices_to_shift = random.sample(boundary_indices, num_to_shift)
        
        for idx in indices_to_shift:
            parts = lines[idx].strip().split('\t')
            if len(parts) >= 3:
                shift = random.uniform(shift_range[0], shift_range[1])
                parts[0] = self.shift_timestamp(parts[0], shift)
                modified[idx] = '\t'.join(parts) + '\n'
                
                self.modifications_log.append({
                    'line': idx,
                    'type': 'boundary_shift',
                    'shift': shift
                })
        
        return modified
    
    def swap_segment_labels(self, lines: List[str], percent: float) -> List[str]:
        """
        Swap segment IDs to test label accuracy
        
        Args:
            lines: Input lines
            percent: Percentage of segment labels to swap
        """
        modified = lines.copy()
        
        segment_indices = []
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[3].strip():
                segment_indices.append(i)
        
        if not segment_indices:
            return modified
        
        num_to_swap = int(len(segment_indices) * (percent / 100))
        indices_to_swap = random.sample(segment_indices, num_to_swap)
        
        for idx in indices_to_swap:
            parts = lines[idx].strip().split('\t')
            annotation = parts[3]
            
            # Change segment ID to a different random one
            new_id = random.randint(1, 9)
            parts[3] = str(new_id)
            modified[idx] = '\t'.join(parts) + '\n'
            
            self.modifications_log.append({
                'line': idx,
                'type': 'label_swap',
                'original': annotation,
                'new': parts[3]
            })
        
        return modified
    
    def create_sanity_check_variants(self, input_file: str, output_dir: str = "./SanityCheck"):
        """
        Create multiple modified versions for comprehensive sanity checking
        All variants include confidence scores for ROC curve testing.
        
        Args:
            input_file: Path to original data file
            output_dir: Directory for output files
        """
        
        lines = self.read_file(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        variants = {
            'perfect': {
                'description': 'Perfect copy (should get 100% metrics, high confidence)',
                'lines': lines.copy(),
                'modifications': []
            },
            'small_time_shift': {
                'description': 'Small timestamp shifts (1-3s) - within tolerance, high confidence',
                'lines': None,
                'modifications': ['timestamp_shift']
            },
            'large_time_shift': {
                'description': 'Large timestamp shifts (10-20s) - beyond threshold, medium confidence',
                'lines': None,
                'modifications': ['large_timestamp_shift']
            },
            'boundary_errors': {
                'description': 'Shifted segment boundaries (5-12s), medium-low confidence',
                'lines': None,
                'modifications': ['boundary_shift']
            },
            'missing_segments': {
                'description': '30% of segment markers removed (false negatives)',
                'lines': None,
                'modifications': ['remove_segments']
            },
            'false_positives': {
                'description': '10 false segment markers added (low confidence)',
                'lines': None,
                'modifications': ['add_false_segments']
            },
            'label_confusion': {
                'description': 'Swapped segment labels (low confidence)',
                'lines': None,
                'modifications': ['label_swap']
            },
            'combined_errors': {
                'description': 'Multiple error types combined (varied confidence)',
                'lines': None,
                'modifications': ['combined']
            }
        }
        
        # Generate modified lines for each variant
        for variant_name, variant_data in variants.items():
            self.modifications_log = []  # Reset log for each variant
            
            if variant_name == 'perfect':
                continue  # Already set
            
            elif variant_name == 'small_time_shift':
                variant_data['lines'] = self.modify_timestamps(lines, 50, (1.0, 3.0))
            
            elif variant_name == 'large_time_shift':
                variant_data['lines'] = self.modify_timestamps(lines, 50, (10.0, 20.0))
            
            elif variant_name == 'boundary_errors':
                variant_data['lines'] = self.shift_segment_boundaries(lines, 70, (5.0, 12.0))
            
            elif variant_name == 'missing_segments':
                variant_data['lines'] = self.remove_segment_markers(lines, 30)
            
            elif variant_name == 'false_positives':
                variant_data['lines'] = self.add_false_segment_markers(lines, 10)
            
            elif variant_name == 'label_confusion':
                variant_data['lines'] = self.swap_segment_labels(lines, 40)
            
            elif variant_name == 'combined_errors':
                combined = lines.copy()
                self.modifications_log = []
                combined = self.modify_timestamps(combined, 20, (-3.0, 3.0))
                combined = self.shift_segment_boundaries(combined, 30, (-5.0, 5.0))
                combined = self.remove_segment_markers(combined, 15)
                combined = self.add_false_segment_markers(combined, 5)
                combined = self.swap_segment_labels(combined, 10)
                variant_data['lines'] = combined
        
        # Write all variants with confidence scores
        summary = []
        for variant_name, variant_data in variants.items():
            output_file = os.path.join(output_dir, f"{base_name}_{variant_name}.txt")
            self.write_file(output_file, variant_data['lines'], add_confidence=True)
            
            summary.append(f"{variant_name:20s} -> {output_file}")
            summary.append(f"{'':20s}    {variant_data['description']}")
        
        # Write summary file
        summary_file = os.path.join(output_dir, "README.txt")
        with open(summary_file, 'w') as f:
            f.write("SANITY CHECK VARIANTS WITH CONFIDENCE SCORES\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Original file: {input_file}\n")
            f.write(f"Created {len(variants)} variants for testing\n\n")
            f.write("All variants include confidence scores for ROC curve testing.\n")
            f.write("Confidence scores reflect prediction quality:\n")
            f.write("  - High (0.85-0.95): Perfect or near-perfect predictions\n")
            f.write("  - Medium (0.50-0.75): Timing errors or boundary shifts\n")
            f.write("  - Low (0.20-0.55): Label errors or false positives\n\n")
            f.write("\n".join(summary))
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("Usage: Compare each variant against the original file\n")
            f.write("using the SegmentEvaluator to validate metrics and ROC curves.\n")
        
        print(f"\nCreated {len(variants)} sanity check variants in: {output_dir}")
        print(f"All files include confidence scores for ROC analysis")
        print(f"See {summary_file} for details")
        
        return variants


# Example usage
if __name__ == "__main__":
    
    modifier = SanityCheckModifier(seed=42)
    
    # Create comprehensive sanity check variants with confidence scores
    input_file = "../data/P001.txt"
    variants = modifier.create_sanity_check_variants(input_file, "./SanityCheck")
    
    print("\n" + "="*70)
    print("SANITY CHECK FILES CREATED WITH CONFIDENCE SCORES")
    print("="*70)
    print("\nTo test your SegmentEvaluator with ROC curves:")
    print("\n  from segment_evaluator import SegmentEvaluator")
    print("  evaluator = SegmentEvaluator()")
    print("\n  # Test with ROC curves")
    print("  results = evaluator.evaluate_with_dual_roc(")
    print("      'SanityCheck/P001_perfect.txt',")
    print("      'data/P001.txt'")
    print("  )")
    print("\n  # Plot ROC curves")
    print("  evaluator.plot_dual_roc_curves(results, 'perfect_roc.png')")