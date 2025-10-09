import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np


def visualize_segments_timeline(
    predicted_segments: List[dict],
    ground_truth_segments: List[dict],
    segment_ids: List[str] = None,
    time_interval: Optional[Tuple[datetime, datetime]] = None,
    figsize: Tuple[int, int] = (16, 6),
    title: str = "Activity Recognition Timeline"
):
    """
    Visualize predicted vs ground truth segments on a single timeline.
    Each segment ID gets two rows: one for ground truth (top) and one for predicted (bottom).
    
    Args:
        predicted_segments: List of dicts with 'segment_id', 'start_time', 'end_time'
        ground_truth_segments: List of dicts with 'segment_id', 'start_time', 'end_time'
        segment_ids: List of segment IDs to visualize (default: ["1", "2", ..., "8"])
        time_interval: Tuple of (start_time, end_time) for x-axis. If None, uses data range
        figsize: Figure size as (width, height)
        title: Plot title
    """
    
    # Default segment IDs
    if segment_ids is None:
        # Exclude segment 6 by using a conditional in the comprehension
        segment_ids = [str(i) for i in range(1, 9) if i != 6]
    
    # Convert to DataFrame for easier manipulation
    pred_df = pd.DataFrame(predicted_segments) if predicted_segments else pd.DataFrame()
    gt_df = pd.DataFrame(ground_truth_segments) if ground_truth_segments else pd.DataFrame()
    
    # Filter by segment IDs
    if not pred_df.empty:
        pred_df = pred_df[pred_df['segment_id'].isin(segment_ids)]
    if not gt_df.empty:
        gt_df = gt_df[gt_df['segment_id'].isin(segment_ids)]
    
    # Determine time interval
    if time_interval is None:
        all_times = []
        if not pred_df.empty:
            all_times.extend(pred_df['start_time'].tolist())
            all_times.extend(pred_df['end_time'].tolist())
        if not gt_df.empty:
            all_times.extend(gt_df['start_time'].tolist())
            all_times.extend(gt_df['end_time'].tolist())
        
        if all_times:
            time_start = min(all_times)
            time_end = max(all_times)
            # Add some padding
            padding = (time_end - time_start).total_seconds() * 0.05
            time_start = time_start - timedelta(seconds=padding)
            time_end = time_end + timedelta(seconds=padding)
        else:
            time_start = datetime.now()
            time_end = time_start + timedelta(minutes=1)
    else:
        time_start, time_end = time_interval
    
    # Create figure - single axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for segments
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_ids)))
    color_map = {seg_id: colors[i] for i, seg_id in enumerate(segment_ids)}
    
    # Each segment ID gets 2 rows: GT on top, Pred on bottom
    num_rows = len(segment_ids) * 2
    
    # Setup axes
    ax.set_ylabel('Segment ID', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.5, num_rows - 0.5)
    
    # Create y-tick labels showing GT/Pred pairs
    yticks = []
    ytick_labels = []
    for i, seg_id in enumerate(segment_ids):
        # Ground truth row (top of pair)
        yticks.append(i * 2)
        ytick_labels.append(f'{seg_id} (GT)')
        # Predicted row (bottom of pair)
        yticks.append(i * 2 + 1)
        ytick_labels.append(f'{seg_id} (P)')
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(time_start, time_end)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    # Draw ground truth segments
    for _, row in gt_df.iterrows():
        seg_id = row['segment_id']
        if seg_id in segment_ids:
            seg_idx = segment_ids.index(seg_id)
            y_pos = seg_idx * 2  # Ground truth row
            start = row['start_time']
            end = row['end_time']
            
            rect = mpatches.Rectangle(
                (start, y_pos - 0.4),
                end - start,
                0.8,
                facecolor=color_map[seg_id],
                edgecolor='black',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add duration text if segment is wide enough
            if (end - start).total_seconds() > (time_end - time_start).total_seconds() * 0.05:
                mid_time = start + (end - start) / 2
                ax.text(mid_time, y_pos, f'{(end-start).total_seconds():.1f}s',
                       ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw predicted segments
    for _, row in pred_df.iterrows():
        seg_id = row['segment_id']
        if seg_id in segment_ids:
            seg_idx = segment_ids.index(seg_id)
            y_pos = seg_idx * 2 + 1  # Predicted row
            start = row['start_time']
            end = row['end_time']
            
            rect = mpatches.Rectangle(
                (start, y_pos - 0.4),
                end - start,
                0.8,
                facecolor=color_map[seg_id],
                edgecolor='red',
                linewidth=2,
                alpha=0.7,
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # Add duration text if segment is wide enough
            if (end - start).total_seconds() > (time_end - time_start).total_seconds() * 0.05:
                mid_time = start + (end - start) / 2
                ax.text(mid_time, y_pos, f'{(end-start).total_seconds():.1f}s',
                       ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Format x-axis
    time_range_seconds = (time_end - time_start).total_seconds()
    if time_range_seconds < 120:  # Less than 2 minutes
        ax.xaxis.set_major_locator(SecondLocator(interval=10))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    elif time_range_seconds < 600:  # Less than 10 minutes
        ax.xaxis.set_major_locator(SecondLocator(interval=30))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    elif time_range_seconds < 3600:  # Less than 1 hour
        ax.xaxis.set_major_locator(MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    else:
        ax.xaxis.set_major_locator(MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='black', linewidth=2, 
                      label='Ground Truth', alpha=0.7),
        mpatches.Patch(facecolor='gray', edgecolor='red', linewidth=2, 
                      linestyle='--', label='Predicted', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


# Alternative: Combined view (both on same axis)
def visualize_segments_combined(
    predicted_segments: List[dict],
    ground_truth_segments: List[dict],
    segment_ids: List[str] = None,
    time_interval: Optional[Tuple[datetime, datetime]] = None,
    figsize: Tuple[int, int] = (16, 6),
    title: str = "Activity Recognition: Ground Truth vs Predicted"
):
    """
    Visualize predicted and ground truth segments on the same timeline.
    Ground truth segments are shown as filled bars, predictions as outlined bars.
    
    Args:
        predicted_segments: List of dicts with 'segment_id', 'start_time', 'end_time'
        ground_truth_segments: List of dicts with 'segment_id', 'start_time', 'end_time'
        segment_ids: List of segment IDs to visualize (default: ["1", "2", ..., "8"])
        time_interval: Tuple of (start_time, end_time) for x-axis. If None, uses data range
        figsize: Figure size as (width, height)
        title: Plot title
    """
    
    # Default segment IDs
    if segment_ids is None:
        segment_ids = [str(i) for i in range(1, 9)]
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predicted_segments) if predicted_segments else pd.DataFrame()
    gt_df = pd.DataFrame(ground_truth_segments) if ground_truth_segments else pd.DataFrame()
    
    # Filter by segment IDs
    if not pred_df.empty:
        pred_df = pred_df[pred_df['segment_id'].isin(segment_ids)]
    if not gt_df.empty:
        gt_df = gt_df[gt_df['segment_id'].isin(segment_ids)]
    
    # Determine time interval
    if time_interval is None:
        all_times = []
        if not pred_df.empty:
            all_times.extend(pred_df['start_time'].tolist())
            all_times.extend(pred_df['end_time'].tolist())
        if not gt_df.empty:
            all_times.extend(gt_df['start_time'].tolist())
            all_times.extend(gt_df['end_time'].tolist())
        
        if all_times:
            time_start = min(all_times)
            time_end = max(all_times)
            padding = (time_end - time_start).total_seconds() * 0.05
            time_start = time_start - timedelta(seconds=padding)
            time_end = time_end + timedelta(seconds=padding)
        else:
            time_start = datetime.now()
            time_end = time_start + timedelta(minutes=1)
    else:
        time_start, time_end = time_interval
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_ids)))
    color_map = {seg_id: colors[i] for i, seg_id in enumerate(segment_ids)}
    
    # Setup axes
    ax.set_ylabel('Segment ID', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.5, len(segment_ids) - 0.5)
    ax.set_yticks(range(len(segment_ids)))
    ax.set_yticklabels(segment_ids)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(time_start, time_end)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    # Draw ground truth segments (filled, solid border)
    for _, row in gt_df.iterrows():
        seg_id = row['segment_id']
        if seg_id in segment_ids:
            y_pos = segment_ids.index(seg_id)
            start = row['start_time']
            end = row['end_time']
            
            rect = mpatches.Rectangle(
                (start, y_pos - 0.35),
                end - start,
                0.7,
                facecolor=color_map[seg_id],
                edgecolor='black',
                linewidth=2.5,
                alpha=0.6,
                label='Ground Truth' if _ == 0 else ''
            )
            ax.add_patch(rect)
    
    # Draw predicted segments (outlined, dashed border, lighter fill)
    for _, row in pred_df.iterrows():
        seg_id = row['segment_id']
        if seg_id in segment_ids:
            y_pos = segment_ids.index(seg_id)
            start = row['start_time']
            end = row['end_time']
            
            rect = mpatches.Rectangle(
                (start, y_pos - 0.3),
                end - start,
                0.6,
                facecolor='white',
                edgecolor='red',
                linewidth=2.5,
                linestyle='--',
                alpha=0.8,
                label='Predicted' if _ == 0 else ''
            )
            ax.add_patch(rect)
    
    # Format x-axis
    time_range_seconds = (time_end - time_start).total_seconds()
    if time_range_seconds < 120:
        ax.xaxis.set_major_locator(SecondLocator(interval=10))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    elif time_range_seconds < 600:
        ax.xaxis.set_major_locator(SecondLocator(interval=30))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    elif time_range_seconds < 3600:
        ax.xaxis.set_major_locator(MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    else:
        ax.xaxis.set_major_locator(MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', linewidth=2.5, 
                      label='Ground Truth', alpha=0.6),
        mpatches.Patch(facecolor='white', edgecolor='red', linewidth=2.5, 
                      linestyle='--', label='Predicted', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


# Example usage with the SegmentEvaluator class
if __name__ == "__main__":
    # Import the SegmentEvaluator (assumes segment_evaluator.py is in same directory or PYTHONPATH)
    from segment_evaluator import SegmentEvaluator
    
    # Initialize evaluator
    evaluator = SegmentEvaluator(
        time_tolerance_seconds=2.0,
        start_time_threshold_seconds=15.0
    )
    
    # Example 1: Visualize actual data from RawData and SanityCheck
    print("Loading and visualizing P001 data...")
    
    # Parse the log files
    gt_df = evaluator.parse_log_file('RawData/P001.txt')
    pred_df = evaluator.parse_log_file('SanityCheck/P001.txt')
    
    # Extract segments
    gt_segments = evaluator.extract_segments(gt_df)
    pred_segments = evaluator.extract_segments(pred_df)
    
    print(f"Ground truth segments: {len(gt_segments)}")
    print(f"Predicted segments: {len(pred_segments)}")
    
    # Visualize - two-panel view
    fig1, axes1 = visualize_segments_timeline(
        predicted_segments=pred_segments,
        ground_truth_segments=gt_segments,
        segment_ids=[str(i) for i in range(1, 9)],
        title="P001: Ground Truth vs Predicted (Two-Panel View)"
    )
    plt.savefig('p001_timeline_twopanel.png', dpi=150, bbox_inches='tight')
    print("Saved: p001_timeline_twopanel.png")
    
    # Visualize - combined view
    fig2, ax2 = visualize_segments_combined(
        predicted_segments=pred_segments,
        ground_truth_segments=gt_segments,
        segment_ids=[str(i) for i in range(1, 9)],
        title="P001: Ground Truth vs Predicted (Combined View)"
    )
    plt.savefig('p001_timeline_combined.png', dpi=150, bbox_inches='tight')
    print("Saved: p001_timeline_combined.png")
    
    # Optional: Show specific time interval
    if gt_segments:
        # Show first 2 minutes of activity
        start = gt_segments[0]['start_time']
        end = start + timedelta(minutes=2)
        
        fig3, ax3 = visualize_segments_combined(
            predicted_segments=pred_segments,
            ground_truth_segments=gt_segments,
            segment_ids=[str(i) for i in range(1, 9)],
            time_interval=(start, end),
            title="P001: First 2 Minutes Detail"
        )
        plt.savefig('p001_timeline_detail.png', dpi=150, bbox_inches='tight')
        print("Saved: p001_timeline_detail.png")
    
    plt.show()
    
    # Example 2: Quick test with sample data (if files don't exist)
    print("\n" + "="*60)
    print("Sample data visualization (for testing):")
    
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    sample_ground_truth = [
        {'segment_id': '1', 'start_time': base_time, 'end_time': base_time + timedelta(seconds=30)},
        {'segment_id': '2', 'start_time': base_time + timedelta(seconds=35), 'end_time': base_time + timedelta(seconds=60)},
        {'segment_id': '3', 'start_time': base_time + timedelta(seconds=65), 'end_time': base_time + timedelta(seconds=95)},
    ]
    
    sample_predicted = [
        {'segment_id': '1', 'start_time': base_time + timedelta(seconds=2), 'end_time': base_time + timedelta(seconds=32)},
        {'segment_id': '2', 'start_time': base_time + timedelta(seconds=38), 'end_time': base_time + timedelta(seconds=58)},
        {'segment_id': '4', 'start_time': base_time + timedelta(seconds=70), 'end_time': base_time + timedelta(seconds=90)},
    ]
    
    fig4, ax4 = visualize_segments_combined(sample_predicted, sample_ground_truth, title="Sample Data")
    plt.show()