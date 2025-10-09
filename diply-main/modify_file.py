import random
import os
from datetime import datetime, timedelta
from typing import List, Tuple

class SanityCheckModifier:
    """
    Modifies sensor data in controlled ways to test segment evaluation system.
    Creates various types of errors to validate evaluation metrics.
    """
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        self.modifications_log = []
    
    def read_file(self, filepath: str) -> List[str]:
        """Read file and return lines"""
        with open(filepath, 'r') as f:
            return f.readlines()
    
    def write_file(self, filepath: str, lines: List[str]):
        """Write lines to file, creating directories if needed"""
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
    
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
            if '-start' in line or '-end' in line:
                marker_indices.append(i)
        
        if not marker_indices:
            return modified
        
        num_to_remove = int(len(marker_indices) * (percent / 100))
        indices_to_remove = random.sample(marker_indices, num_to_remove)
        
        for idx in indices_to_remove:
            parts = lines[idx].strip().split('\t')
            if len(parts) >= 4:
                # Remove the annotation column
                parts[3] = ''
                if len(parts) > 4:
                    parts[4] = ''
                modified[idx] = '\t'.join(parts[:4]) + '\n'
                
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
        
        # Find lines without segment markers
        eligible_indices = []
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                has_annotation = len(parts) >= 4 and parts[3].strip()
                if not has_annotation or '-start' not in line and '-end' not in line:
                    eligible_indices.append(i)
        
        if not eligible_indices:
            return modified
        
        num_to_add = min(num_to_add, len(eligible_indices))
        indices_to_modify = random.sample(eligible_indices, num_to_add)
        
        for idx in indices_to_modify:
            parts = lines[idx].strip().split('\t')
            
            # Add a fake segment marker
            fake_id = random.randint(1, 8)
            fake_type = random.choice(['start', 'end'])
            fake_annotation = f"{fake_id}-{fake_type}"
            
            if len(parts) == 3:
                parts.append(fake_annotation)
            else:
                parts[3] = fake_annotation
            
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
        
        # Find segment boundary markers
        boundary_indices = []
        for i, line in enumerate(lines):
            if '-start' in line or '-end' in line or 'START' in line or 'STOP' in line:
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
            if len(parts) >= 4 and parts[3]:
                if '-start' in parts[3] or '-end' in parts[3]:
                    segment_indices.append(i)
        
        if not segment_indices:
            return modified
        
        num_to_swap = int(len(segment_indices) * (percent / 100))
        indices_to_swap = random.sample(segment_indices, num_to_swap)
        
        for idx in indices_to_swap:
            parts = lines[idx].strip().split('\t')
            annotation = parts[3]
            
            if '-' in annotation:
                prefix, suffix = annotation.split('-', 1)
                # Change segment ID
                new_id = random.randint(1, 9)
                parts[3] = f"{new_id}-{suffix}"
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
        
        Args:
            input_file: Path to original data file
            output_dir: Directory for output files
        """
        
        lines = self.read_file(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        variants = {
            'perfect': {
                'description': 'Perfect copy (should get 100% metrics)',
                'lines': lines.copy()
            },
            'small_time_shift': {
                'description': 'Small timestamp shifts (10-15s) - within tolerance',
                'lines': self.modify_timestamps(lines, 50, (10.0, 15.0))
            },
            'large_time_shift': {
                'description': 'Large timestamp shifts (15-25s) - beyond threshold',
                'lines': self.modify_timestamps(lines, 50, (15.0, 25.0))
            },
            'boundary_errors': {
                'description': 'Shifted segment boundaries (5-12s)',
                'lines': self.shift_segment_boundaries(lines, 70, (10.0, 17.0))
            },
            'missing_segments': {
                'description': '30% of segment markers removed',
                'lines': self.remove_segment_markers(lines, 30)
            },
            'false_positives': {
                'description': '10 false segment markers added',
                'lines': self.add_false_segment_markers(lines, 10)
            },
            'label_confusion': {
                'description': 'Swapped segment labels',
                'lines': self.swap_segment_labels(lines, 40)
            },
            'combined_errors': {
                'description': 'Multiple error types combined',
                'lines': lines.copy()
            }
        }
        
        # For combined errors, apply multiple modifications
        self.modifications_log = []
        combined = lines.copy()
        combined = self.modify_timestamps(combined, 20, (-3.0, 3.0))
        combined = self.shift_segment_boundaries(combined, 30, (-2.0, 2.0))
        combined = self.remove_segment_markers(combined, 15)
        combined = self.add_false_segment_markers(combined, 5)
        variants['combined_errors']['lines'] = combined
        
        # Write all variants
        summary = []
        for variant_name, variant_data in variants.items():
            output_file = os.path.join(output_dir, f"{base_name}_{variant_name}.txt")
            self.write_file(output_file, variant_data['lines'])
            
            summary.append(f"{variant_name:20s} -> {output_file}")
            summary.append(f"{'':20s}    {variant_data['description']}")
        
        # Write summary file
        summary_file = os.path.join(output_dir, "README.txt")
        with open(summary_file, 'w') as f:
            f.write("SANITY CHECK VARIANTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Original file: {input_file}\n")
            f.write(f"Created {len(variants)} variants for testing\n\n")
            f.write("\n".join(summary))
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("Usage: Compare each variant against the original file\n")
            f.write("using the SegmentEvaluator to validate metrics.\n")
        
        print(f"\nCreated {len(variants)} sanity check variants in: {output_dir}")
        print(f"See {summary_file} for details")
        
        return variants


# Example usage
if __name__ == "__main__":
    
    modifier = SanityCheckModifier(seed=42)
    
    # Create comprehensive sanity check variants
    input_file = "./RawData/P001.txt"
    variants = modifier.create_sanity_check_variants(input_file, "./SanityCheck")
    
    print("\n" + "="*70)
    print("SANITY CHECK FILES CREATED")
    print("="*70)
    print("\nTo test your SegmentEvaluator, run:")
    print("\n  from segment_evaluator import SegmentEvaluator")
    print("  evaluator = SegmentEvaluator()")
    print("\n  # Test perfect copy (should get F1=1.0, AUC=1.0)")
    print("  results = evaluator.evaluate(")
    print("      'SanityCheck/P001_perfect.txt',")
    print("      'RawData/P001.txt'")
    print("  )")
    print("  evaluator.print_report(results)")
    print("\n  # Test other variants to see how metrics respond")
    print("  # to different types of errors...")