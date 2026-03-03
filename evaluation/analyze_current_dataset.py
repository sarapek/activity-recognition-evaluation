#!/usr/bin/env python3
"""
Analyze class distribution for the current dataset (sensor event data).
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

def parse_sensor_file(filepath):
    """Parse sensor file and return activity segments."""
    events = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 4:
                continue

            date = words[0]
            time = words[1]
            sensor_id = words[2]
            sensor_status = words[3]
            activity = words[4] if len(words) > 4 else "Other_Activity"

            # Parse datetime
            try:
                dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S.%f")
            except:
                try:
                    dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
                except:
                    continue

            events.append((dt, activity))

    # Group into segments (continuous sequences of same activity)
    segments = []
    if events:
        current_activity = events[0][1]
        segment_start = events[0][0]

        for i in range(1, len(events)):
            dt, activity = events[i]

            # If activity changes, end current segment
            if activity != current_activity:
                segment_end = events[i-1][0]
                duration = (segment_end - segment_start).total_seconds()
                segments.append((current_activity, segment_start, segment_end, duration))

                # Start new segment
                current_activity = activity
                segment_start = dt

        # Add final segment
        segment_end = events[-1][0]
        duration = (segment_end - segment_start).total_seconds()
        segments.append((current_activity, segment_start, segment_end, duration))

    return events, segments

def analyze_dataset(data_dir, activity_filter=None):
    """Analyze class distribution in the dataset."""

    # Statistics per activity
    activity_stats = defaultdict(lambda: {
        'num_events': 0,
        'num_segments': 0,
        'total_duration': 0,
        'num_files': 0,
        'durations': [],
        'file_list': []
    })

    total_files = 0
    total_events = 0
    files_processed = []

    # Process all sensor files
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.txt'):
            total_files += 1
            filepath = os.path.join(data_dir, filename)

            try:
                events, segments = parse_sensor_file(filepath)
                files_processed.append(filename)

                # Count events per activity
                event_activities = [act for dt, act in events]
                for act in event_activities:
                    activity_stats[act]['num_events'] += 1
                    total_events += 1

                # Count segments per activity
                file_activities = set()
                for activity, start, end, duration in segments:
                    if activity_filter and activity not in activity_filter and activity != "Other_Activity":
                        continue

                    activity_stats[activity]['num_segments'] += 1
                    activity_stats[activity]['total_duration'] += duration
                    activity_stats[activity]['durations'].append(duration)
                    file_activities.add(activity)

                # Count files per activity
                for activity in file_activities:
                    activity_stats[activity]['num_files'] += 1
                    if filename not in activity_stats[activity]['file_list']:
                        activity_stats[activity]['file_list'].append(filename)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return activity_stats, total_files, total_events

def create_visualizations(activity_stats, total_files, output_dir='.'):
    """Create visualization graphs."""

    # Filter out "Other_Activity" and sort numerically
    activities = sorted(
        [act for act in activity_stats.keys() if act != "Other_Activity"],
        key=lambda x: int(x) if x.isdigit() else 999
    )

    if not activities:
        print("No activities found to visualize!")
        return None

    # Prepare data
    num_events = [activity_stats[act]['num_events'] for act in activities]
    num_segments = [activity_stats[act]['num_segments'] for act in activities]
    total_durations = [activity_stats[act]['total_duration'] / 3600 for act in activities]  # Hours
    num_files = [activity_stats[act]['num_files'] for act in activities]
    mean_durations = [
        activity_stats[act]['total_duration'] / activity_stats[act]['num_segments']
        if activity_stats[act]['num_segments'] > 0 else 0
        for act in activities
    ]

    # Create figure with 5 subplots (2x3 grid)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Class Distribution Analysis - data_filtered_normal', fontsize=16, fontweight='bold')

    # 1. Number of events per activity
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(activities, num_events, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Activity ID', fontweight='bold')
    ax1.set_ylabel('Number of Sensor Events', fontweight='bold')
    ax1.set_title('Sensor Events per Activity')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # 2. Number of segments per activity
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(activities, num_segments, color='coral', alpha=0.7)
    ax2.set_xlabel('Activity ID', fontweight='bold')
    ax2.set_ylabel('Number of Segments', fontweight='bold')
    ax2.set_title('Segments per Activity')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # 3. Total duration per activity
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(activities, total_durations, color='mediumseagreen', alpha=0.7)
    ax3.set_xlabel('Activity ID', fontweight='bold')
    ax3.set_ylabel('Total Duration (hours)', fontweight='bold')
    ax3.set_title('Total Duration per Activity')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}h',
                ha='center', va='bottom', fontsize=9)

    # 4. Number of files per activity
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(activities, num_files, color='mediumpurple', alpha=0.7)
    ax4.set_xlabel('Activity ID', fontweight='bold')
    ax4.set_ylabel('Number of Files', fontweight='bold')
    ax4.set_title(f'Files per Activity (Total: {total_files} files)')
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # 5. Mean segment duration
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(activities, mean_durations, color='darkorange', alpha=0.7)
    ax5.set_xlabel('Activity ID', fontweight='bold')
    ax5.set_ylabel('Mean Segment Duration (seconds)', fontweight='bold')
    ax5.set_title('Mean Segment Duration per Activity')
    ax5.grid(axis='y', alpha=0.3)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}s',
                ha='center', va='bottom', fontsize=9)

    # 6. Events per segment (average)
    ax6 = fig.add_subplot(gs[1, 2])
    events_per_segment = [
        num_events[i] / num_segments[i] if num_segments[i] > 0 else 0
        for i in range(len(activities))
    ]
    bars6 = ax6.bar(activities, events_per_segment, color='teal', alpha=0.7)
    ax6.set_xlabel('Activity ID', fontweight='bold')
    ax6.set_ylabel('Events per Segment', fontweight='bold')
    ax6.set_title('Average Events per Segment')
    ax6.grid(axis='y', alpha=0.3)
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    # Save figure
    output_path = os.path.join(output_dir, 'class_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return output_path

def print_analysis(activity_stats, total_files, total_events):
    """Print detailed analysis."""

    print("\n" + "="*100)
    print("CLASS DISTRIBUTION ANALYSIS - data_filtered_normal")
    print("="*100)
    print(f"\nTotal files analyzed: {total_files}")
    print(f"Total sensor events: {total_events:,}")

    # Filter out "Other_Activity" for main stats
    activities = sorted(
        [act for act in activity_stats.keys() if act != "Other_Activity"],
        key=lambda x: int(x) if x.isdigit() else 999
    )

    print(f"Activities found: {len(activities)}")
    print("\n" + "-"*100)

    # Print header
    print(f"{'Act':<4} {'Events':<12} {'Segments':<10} {'Files':<8} {'Total Dur':<12} {'Mean Dur':<12} {'Evt/Seg':<10}")
    print("-"*100)

    total_activity_events = 0
    total_segments = 0
    total_duration = 0

    for activity in activities:
        stats = activity_stats[activity]
        num_events = stats['num_events']
        num_segments = stats['num_segments']
        num_files = stats['num_files']
        total_dur = stats['total_duration']
        mean_dur = total_dur / num_segments if num_segments > 0 else 0
        evt_per_seg = num_events / num_segments if num_segments > 0 else 0

        total_activity_events += num_events
        total_segments += num_segments
        total_duration += total_dur

        print(f"{activity:<4} {num_events:<12,} {num_segments:<10} {num_files:<8} "
              f"{total_dur:<12.0f}s {mean_dur:<12.1f}s {evt_per_seg:<10.1f}")

    # Print Other_Activity if exists
    if "Other_Activity" in activity_stats:
        stats = activity_stats["Other_Activity"]
        print("-"*100)
        print(f"{'Other':<4} {stats['num_events']:<12,} {stats['num_segments']:<10} {stats['num_files']:<8} "
              f"{stats['total_duration']:<12.0f}s")

    print("-"*100)
    print(f"{'Total':<4} {total_activity_events:<12,} {total_segments:<10}")
    print("\n" + "="*100)

    # Class balance analysis
    print("\nCLASS BALANCE ANALYSIS:")
    print("-"*100)

    for activity in activities:
        stats = activity_stats[activity]
        event_pct = (stats['num_events'] / total_activity_events) * 100 if total_activity_events > 0 else 0
        segment_pct = (stats['num_segments'] / total_segments) * 100 if total_segments > 0 else 0
        duration_pct = (stats['total_duration'] / total_duration) * 100 if total_duration > 0 else 0

        print(f"Activity {activity}: {event_pct:5.2f}% of events, {segment_pct:5.2f}% of segments, "
              f"{duration_pct:5.2f}% of duration")

    # Calculate imbalance ratios
    if activities:
        max_events = max(activity_stats[act]['num_events'] for act in activities)
        min_events = min(activity_stats[act]['num_events'] for act in activities)
        max_segments = max(activity_stats[act]['num_segments'] for act in activities)
        min_segments = min(activity_stats[act]['num_segments'] for act in activities)

        event_ratio = max_events / min_events if min_events > 0 else float('inf')
        segment_ratio = max_segments / min_segments if min_segments > 0 else float('inf')

        print(f"\nClass imbalance ratio (events): {event_ratio:.1f}:1")
        print(f"Class imbalance ratio (segments): {segment_ratio:.1f}:1")

    print("="*100)

def save_json_report(activity_stats, total_files, total_events, output_path='class_distribution_current.json'):
    """Save analysis as JSON."""

    activities = sorted(
        [act for act in activity_stats.keys() if act != "Other_Activity"],
        key=lambda x: int(x) if x.isdigit() else 999
    )

    report = {
        'dataset': 'data_filtered_normal',
        'total_files': total_files,
        'total_events': total_events,
        'num_activities': len(activities),
        'activities': {}
    }

    for activity in activities:
        stats = activity_stats[activity]
        report['activities'][activity] = {
            'num_events': stats['num_events'],
            'num_segments': stats['num_segments'],
            'num_files': stats['num_files'],
            'total_duration_seconds': stats['total_duration'],
            'mean_duration_seconds': stats['total_duration'] / stats['num_segments'] if stats['num_segments'] > 0 else 0,
            'min_duration_seconds': min(stats['durations']) if stats['durations'] else 0,
            'max_duration_seconds': max(stats['durations']) if stats['durations'] else 0,
            'std_duration_seconds': float(np.std(stats['durations'])) if stats['durations'] else 0,
            'events_per_segment': stats['num_events'] / stats['num_segments'] if stats['num_segments'] > 0 else 0
        }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nJSON report saved to: {output_path}")

if __name__ == '__main__':
    data_dir = './data_filtered_normal'

    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found!")
        exit(1)

    print("Analyzing dataset...")
    activity_stats, total_files, total_events = analyze_dataset(data_dir)

    # Print analysis
    print_analysis(activity_stats, total_files, total_events)

    # Create visualizations
    create_visualizations(activity_stats, total_files)

    # Save JSON report
    save_json_report(activity_stats, total_files, total_events)

    print("\nAnalysis complete!")
