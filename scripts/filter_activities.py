import os
import re

def filter_activities(input_file, output_file, keep_activities=[str(i) for i in range(1, 25)]):
    """
    Filter data to keep only activities 1-24

    Rules:
    - Extracts activity number from annotations like '1-start', '1-end', '1.1', '1.2'
    - Labels all events between X-start and X-end as activity X
    - Keeps ONLY activities 1-24 (removes annotation errors like 41, 1001, etc.)
    - Removes "Other_Activity" and non-numeric annotations
    - REMOVES completely unannotated lines (no annotation at all)
    """
    current_activity = None

    with open(input_file, 'r', encoding='utf-8-sig') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            words = line.strip().split()

            if len(words) < 4:  # Invalid line (need at least date, time, sensor, status)
                continue

            date, time, sensor, status = words[0], words[1], words[2], words[3]
            annotation = words[4] if len(words) > 4 else None

            # Filter out battery sensors
            if sensor.startswith('BATP') or sensor.startswith('BATV'):
                continue

            # Filter out problematic sensors (SS, A sensors, unknown, 023)
            if sensor.startswith('SS') or sensor.startswith('A') or sensor in ['unknown', '023']:
                continue

            # Filter out lines with error- in any column (annotation or after)
            if any('error-' in word.lower() for word in words[4:]):
                continue

            # If there's an annotation, check for start/end markers
            if annotation:
                # Extract activity number from annotations
                # Patterns: '1-start', '1-end', '1.1', '1.2', '2-start', etc.
                match = re.match(r'^(\d+)', annotation)

                if match:
                    activity_num = match.group(1)

                    # Check if it's a start/end marker
                    if '-start' in annotation.lower():
                        current_activity = activity_num if activity_num in keep_activities else None
                        # Write the start event
                        if current_activity:
                            new_line = f"{date} {time} {sensor} {status} {current_activity}\n"
                            f_out.write(new_line)
                        continue
                    elif '-end' in annotation.lower():
                        # Write the end event, then clear current activity
                        if activity_num in keep_activities and current_activity == activity_num:
                            new_line = f"{date} {time} {sensor} {status} {activity_num}\n"
                            f_out.write(new_line)
                        current_activity = None
                        continue
                    else:
                        # It's a point annotation like '1.1' or just '1'
                        # This updates current activity for subsequent unlabeled events
                        if activity_num in keep_activities:
                            current_activity = activity_num

            # Write ALL events when inside an activity segment (labeled or not)
            if current_activity and current_activity in keep_activities:
                new_line = f"{date} {time} {sensor} {status} {current_activity}\n"
                f_out.write(new_line)

def filter_all_files(data_dir='./data_raw_normal', output_dir='./data_filtered_normal'):
    """Filter all data files - keeps all numbered activities"""
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
            with open(input_path, 'r', encoding='utf-8-sig') as f:
                lines_in = sum(1 for _ in f)

            filter_activities(input_path, output_path)

            # Count lines after
            with open(output_path, 'r', encoding='utf-8') as f:
                lines_out = sum(1 for _ in f)
            
            stats['total_files'] += 1
            stats['total_lines_in'] += lines_in
            stats['total_lines_out'] += lines_out
            
            print(f"[OK] {filename}: {lines_in} -> {lines_out} lines")
    
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