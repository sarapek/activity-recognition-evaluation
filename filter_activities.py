import os
import re

def filter_activities(input_file, output_file, keep_activities=[str(i) for i in range(1, 9)]):
    """
    Filter data to only keep specified activities 1-8
    
    Rules:
    - Extracts activity number from annotations like '1-start', '1-end', '1.1', '1.2'
    - Labels all events between X-start and X-end as activity X
    - Keeps only activities 1-8
    - Maps activities 9-24 to Other_Activity
    - REMOVES completely unannotated lines (no annotation at all)
    """
    current_activity = None
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            words = line.strip().split()
            
            if len(words) < 4:  # Invalid line
                continue
            
            if len(words) < 5:  # No annotation - SKIP THIS LINE
                continue
            
            date, time, sensor, status, annotation = words[0], words[1], words[2], words[3], words[4]
            
            # Extract activity number from annotations
            # Patterns: '1-start', '1-end', '1.1', '1.2', '2-start', etc.
            match = re.match(r'^(\d+)', annotation)
            
            if match:
                activity_num = match.group(1)
                
                # Check if it's a start/end marker
                if '-start' in annotation.lower():
                    current_activity = activity_num
                elif '-end' in annotation.lower():
                    # Write this line, then clear current activity
                    if activity_num in keep_activities:
                        new_line = f"{date} {time} {sensor} {status} {activity_num}\n"
                        f_out.write(new_line)
                    current_activity = None
                    continue
                else:
                    # It's a point annotation like '1.1' or just '1'
                    current_activity = activity_num
                
                # Write the line if activity is in range 1-8
                if current_activity and current_activity in keep_activities:
                    new_line = f"{date} {time} {sensor} {status} {current_activity}\n"
                    f_out.write(new_line)
            else:
                # Annotation exists but doesn't match our pattern
                # Could be "Other_Activity" or something else
                # Skip these lines for cleaner training data
                continue

def filter_all_files(data_dir='./data', output_dir='./data_filtered'):
    """Filter all data files"""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'total_lines_in': 0,
        'total_lines_out': 0
    }
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Count lines before
            with open(input_path, 'r') as f:
                lines_in = sum(1 for _ in f)
            
            filter_activities(input_path, output_path)
            
            # Count lines after
            with open(output_path, 'r') as f:
                lines_out = sum(1 for _ in f)
            
            stats['total_files'] += 1
            stats['total_lines_in'] += lines_in
            stats['total_lines_out'] += lines_out
            
            print(f"✓ {filename}: {lines_in} → {lines_out} lines")
    
    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {stats['total_files']}")
    print(f"Total input lines: {stats['total_lines_in']}")
    print(f"Total output lines: {stats['total_lines_out']}")
    print(f"Lines removed: {stats['total_lines_in'] - stats['total_lines_out']}")
    print(f"Retention rate: {100 * stats['total_lines_out'] / stats['total_lines_in']:.1f}%")

if __name__ == "__main__":
    filter_all_files()