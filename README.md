# Activity Recognition & Evaluation System

This repository uses the Activity Learner (AL) — Smart Home Edition, developed by Dr. Diane J. Cook (School of Electrical Engineering and Computer Science, Washington State University). While AL is employed here to learn and label activity models from ambient sensor data, the primary purpose of this repository is to evaluate and analyze the performance of the AL algorithm.

In other words, this project does not develop or modify AL itself; instead, it provides an experimental framework to assess how well the AL algorithm performs under different conditions and datasets.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training & Evaluation Pipeline](#training--evaluation-pipeline)
- [Evaluation Tools](#evaluation-tools)
- [File Structure](#file-structure)

---

## Overview

This system consists of three main components:

1. **Data Filtering** (`filter_activities.py`) - Prepares raw sensor data for training
2. **Activity Learning** (`al.py`) - Trains and tests activity recognition models (see `AL/README.md` for details)
3. **Evaluation Suite** - Validates model performance with comprehensive metrics

---

## Setup

### Required Packages

```bash
pip install numpy scikit-learn joblib matplotlib
```

Tested versions:
- `numpy==1.19.2`
- `scikit-learn==0.23.2`
- `joblib==0.17.0`

### Directory Structure

```
project/
├── data/                  # Raw CASAS data files
├── data_filtered/         # Filtered data (output from filter_activities.py)
├── data_spaces/           # Unfiltered data reshaped with spaces instead of tabs as delimeters
├── output/                # Timestamped run directories created by train_and_evaluate.py
│   ├── run_20241019_143022_data_filtered_thresh15s/
│   │   ├── predicted_0_P001.txt
│   │   ├── timeline_0_P001.png
│   │   ├── evaluation_by_activity.txt
│   │   └── roc_by_activity.png
│   └── run_20241020_091534_data_filtered_thresh15s/
│       └── ...
├── AL/
│   ├── al.py              # Main activity learning script
│   ├── README.md          # Detailed AL documentation
│   └── model/             # Trained models saved here
├── evaluation/
│   ├── segment_evaluator.py
│   ├── test_evaluation.py
│   └── modify_file.py      
├── filter_activities.py
└── train_and_evaluate.py
```

---

## Data Preparation

### Filter Activities with `filter_activities.py`

The filter script processes raw sensor data to extract only relevant activities (activity numbers 1-8), making the data suitable for use with `al.py`.

**What it does:**
- Filters sensor events to keep only activities 1-8
- Removes "Other_Activity" and unlabeled sensor events
- Handles both start/end markers and point annotations
- Creates cleaned training data in `data_filtered/` directory

**Usage:**

```bash
python filter_activities.py
```

**Input format** (raw data):
```
2008-02-27 00:07:46.898610 M003 ON 1
2008-02-27 00:10:32.539722 M004 ON Other_Activity
2008-02-27 00:12:15.268912 M005 OFF 2-start
2008-02-27 00:15:48.982341 M006 ON 2-end
```

**Output format** (filtered data):
```
2008-02-27 00:07:46.898610 M003 ON 1
2008-02-27 00:12:15.268912 M005 OFF 2
2008-02-27 00:15:48.982341 M006 ON 2
```

**Notes:**
- Only events with activity labels 1-8 are kept
- Lines without annotations or with "Other_Activity" are removed
- This creates cleaner training data for the AL model

---

## Training & Evaluation Pipeline

### Using `train_and_evaluate.py` - Complete Pipeline

The `train_and_evaluate.py` script provides an automated end-to-end workflow that handles training, annotation, and evaluation in a single run. This is the recommended way to work with the system.

#### Running the Complete Pipeline

```bash
python train_and_evaluate.py
```

**What it does:**
1. Automatically splits your data into training set (200 files) and test set (3 files)
2. Trains a new activity recognition model OR uses an existing trained model
3. Annotates each test file with predicted activity labels
4. Evaluates predictions against ground truth using comprehensive metrics
5. Generates detailed performance reports and optional visualizations

#### Interactive Workflow

When you run the script, you'll be prompted with several options:

**1. Model Selection**
```
Existing model found at ./AL/model/activity_model.pkl.gz
Do you want to use the existing model? (y/n):
```
- **Yes (y)**: Uses the pre-trained model (faster, good for testing)
- **No (n)**: Trains a new model from scratch on the training data

**2. Visualization Options**
```
Do you want to enable timeline visualization? (y/n):
```
- **Yes (y)**: Generates visual timeline plots comparing predictions vs ground truth
- **No (n)**: Skips visualization (faster processing)

#### Output Files

After running, the script creates a **timestamped run directory** for each execution:

```
./output/run_YYYYMMDD_HHMMSS_data_filtered_thresh15s/
```

The folder name includes:
- Timestamp of the run
- Name of the training data directory used
- Start error threshold value (in seconds)

**Inside each run directory:**

**Annotated Predictions:**
- `predicted_0_<filename>.txt` - Predictions for first test file
- `predicted_1_<filename>.txt` - Predictions for second test file
- `predicted_2_<filename>.txt` - Predictions for third test file

**Evaluation Reports:**
- `evaluation_by_activity.txt` - Consolidated metrics for each activity (1-8) across all test files
- Console output showing per-file evaluation metrics

**Visualizations** (if enabled):
- `timeline_0_<filename>.png` - Timeline for first test file
- `timeline_1_<filename>.png` - Timeline for second test file
- `timeline_2_<filename>.png` - Timeline for third test file
- `roc_by_activity.png` - ROC curves for all activities aggregated across test files

**Model:**
- Saved separately in `./AL/model/<model_name>.pkl.gz`

#### Example Output

```
========================================
Processing test file 1/3: P001.txt
========================================
✓ Cleared annotations
✓ Annotated file created with 1247 lines
✓ Saved predictions -> ./output/run_20241019_143022_data_filtered_thresh15s/predicted_0_P001.txt

Evaluating predictions...
  Ground truth segments: 45
  Predicted segments: 48

✓ Evaluation complete
✓ Saved visualization -> ./output/run_20241019_143022_data_filtered_thresh15s/timeline_0_P001.png

========================================
CONSOLIDATED EVALUATION BY ACTIVITY (ALL TEST FILES)
========================================

--- Activity 1 ---
  Total Predicted:     15
  Total Ground Truth:  13
  True Positives:      12
  False Positives:     3
  False Negatives:     1
  Precision:           0.8000
  Recall:              0.9231
  F1 Score:            0.8571
  AUC:                 0.8945
  Start Error (mean):  1.2s (±0.8s)
  End Error (mean):    1.5s
  Duration Err (mean): 0.3s

[... Activities 2-8 ...]

========================================

✓ Saved ROC curve by activity -> ./output/run_20241019_143022_data_filtered_thresh15s/roc_by_activity.png
✓ Saved consolidated report -> ./output/run_20241019_143022_data_filtered_thresh15s/evaluation_by_activity.txt
```

#### When to Use Different Modes

**Use existing model when:**
- Testing changes to evaluation metrics
- Comparing different test datasets
- Quick validation of new data
- You're satisfied with current model performance

**Train new model when:**
- First time setup
- Adding new training data
- Model performance is poor
- Changing feature extraction or classification parameters

### Manual Training & Testing

For more control over individual steps, you can use `al.py` directly. See `AL/README.md` for detailed documentation on:
- Training models with custom parameters
- Cross-validation
- Leave-one-out testing
- Annotating unlabeled data
- Activity and sensor filtering options

---

## Evaluation Tools

### 1. Sanity Check Testing (`test_evaluation.py`)

Validates that your evaluation metrics work correctly by testing against known modifications.

#### Running Sanity Checks

```bash
python ./evaluation/test_evaluation.py
```

**Interactive Options:**

1. **Create sanity check files?** (y/n)
   - Generates 8 test variants from ground truth data
   - Variants include: perfect copy, time shifts, missing segments, false positives, etc.

2. **Enable per-segment breakdown?** (y/n)
   - Shows detailed metrics for each activity segment
   - Useful for debugging specific detection failures

3. **Enable segment timeline visualization?** (y/n)
   - Creates visual timeline comparing predicted vs ground truth segments
   - Helps visualize timing errors and missed detections

4. **Start and end time tolerance in seconds:** (e.g., 2)
   - Sets the acceptable timing error threshold
   - Predictions within this tolerance are considered accurate

5. **Choose test mode:**
   - **Option 1**: Run full sanity check test suite (all 8 variants)
   - **Option 2**: Quick test single file
   - **Option 3**: Test with confidence scores (for SVM predictions)

#### Sanity Check Variants

The system creates 8 test files to validate evaluation metrics:

1. **perfect** - Exact copy (should get F1=1.0)
2. **small_time_shift** - ±1-5 second shifts
3. **large_time_shift** - ±10-30 second shifts
4. **boundary_errors** - Start/end time errors
5. **missing_segments** - 20% of segments removed
6. **false_positives** - Extra incorrect segments added
7. **label_confusion** - Some activity labels swapped
8. **combined_errors** - Multiple error types

**Output:**
- Detailed metrics for each variant
- Summary report comparing all variants
- Results saved to `./SanityCheck/evaluation_results.json`

### 2. Segment Evaluator (`segment_evaluator.py`)

The core evaluation engine that computes metrics for activity recognition.

#### Basic Usage

```python
from segment_evaluator import SegmentEvaluator

# Initialize evaluator
evaluator = SegmentEvaluator(
    time_tolerance_seconds=2.0,      # Accuracy threshold
    start_time_threshold_seconds=15.0 # Detection latency threshold
)

# Evaluate predictions
results = evaluator.evaluate(
    predicted_file='./output/predictions.txt',
    ground_truth_file='./data_filtered/ground_truth.txt',
    per_segment=True  # Include per-segment breakdown
)

# Print report
evaluator.print_report(results, show_per_segment=True)
```

#### Parameters Explained

**`time_tolerance_seconds`** (default: 2.0)
- Acceptable timing error for start/end times
- Predictions within ±2 seconds are considered temporally accurate
- Used to determine if timing is "within tolerance"

**`start_time_threshold_seconds`** (default: 15.0)
- Maximum acceptable detection delay
- If activity detected >15 seconds late, it's marked as "too late"
- Predictions that start early or within this threshold are valid detections

**`per_segment`** (boolean)
- If True, includes detailed per-segment analysis
- Shows which specific activities were detected/missed
- Useful for debugging model performance

**`confidence_scores`** (optional list)
- Confidence values for each predicted segment (0.0 to 1.0)
- Required for AUC calculation
- Used when testing SVM predictions with probability estimates

#### Metrics Reported

**Detection Metrics:**
- **F1 Score** - Harmonic mean of precision and recall
- **Precision** - Fraction of predictions that are correct
- **Recall (TPR)** - Fraction of ground truth segments detected
- **FPR** - False positive rate
- **AUC** - Area under ROC curve (requires confidence scores)

**Confusion Matrix:**
- **True Positives** - Correctly detected segments
- **False Positives** - Incorrect detections
- **False Negatives** - Missed segments
- **True Negatives** - Correctly identified non-events

**Temporal Accuracy:**
- **Start Time Error** - Mean/std/max/median error in start times
- **End Time Error** - Mean/std/max/median error in end times
- **Duration Error** - Mean/std/max/median error in segment duration

**Per-Segment Details** (if enabled):
- Status for each activity: detected, missed, false positive, too late
- Individual timing errors
- Whether timing is within tolerance

---

## Common Workflows

### Complete Pipeline from Raw Data (Recommended)

```bash
# Step 1: Filter raw data
python filter_activities.py

# Step 2: Run complete training and evaluation pipeline
python train_and_evaluate.py
# Choose options:
# - Train new model or use existing
# - Enable/disable visualization

# Results are saved in: ./output/run_YYYYMMDD_HHMMSS_data_filtered_thresh15s/
```

### Quick Testing with Sanity Checks

```bash
# Generate and test sanity check variants
cd evaluation
python test_evaluation.py

# Choose options:
# 1. y - Create sanity check files
# 2. y - Enable per-segment breakdown
# 3. y - Enable visualization
# 4. 2 - Set time tolerance to 2 seconds
# 5. 1 - Run full test suite
```

### Custom Training & Evaluation

For advanced use cases requiring fine-grained control:

```bash
# Step 1: Filter raw data
python filter_activities.py

# Step 2: Train model manually (see AL/README.md)
python al.py --mode TRAIN --data ./data_filtered/train_combined.txt --model custom_model

# Step 3: Annotate test data
python al.py --mode ANNOTATE --data ./test_data.txt --model custom_model

# Step 4: Evaluate predictions
python -c "
from evaluation.segment_evaluator import SegmentEvaluator
evaluator = SegmentEvaluator(time_tolerance_seconds=2.0, start_time_threshold_seconds=15.0)
results = evaluator.evaluate('./data.al', './test_data_with_labels.txt', per_segment=True)
evaluator.print_report(results)
"
```

---

## Understanding Evaluation Results

### Good Performance Indicators
- **F1 Score ≥ 0.90** - Strong overall performance
- **Recall ≥ 0.85** - Most activities detected
- **Start Error < 5s** - Timely detection
- **Within Tolerance > 80%** - Accurate timing

### Common Issues
- **Low Recall** - Model missing many activities (check for missing segments)
- **Low Precision** - Too many false positives (model over-predicting)
- **High Start Error** - Detection delays (adjust start_time_threshold)
- **Low AUC** - Poor confidence calibration (retrain with better features)

---

## File Formats

### Annotated Sensor Data Format
```
YYYY-MM-DD HH:MM:SS.mmmmmm SENSOR_ID STATE ACTIVITY_LABEL
```

Example:
```
2008-02-27 00:07:46.898610 M003 ON 1
2008-02-27 00:10:32.539722 M004 OFF 2
```

### Unannotated Sensor Data Format
```
YYYY-MM-DD HH:MM:SS.mmmmmm SENSOR_ID STATE
```

Example:
```
2008-02-27 00:07:46.898610 M003 ON
2008-02-27 00:10:32.539722 M004 OFF
```

---

## Troubleshooting

**Issue:** `filter_activities.py` removes too many lines
- **Solution:** Check that your activity labels are 1-8 (not strings)

**Issue:** `train_and_evaluate.py` fails to find training files
- **Solution:** Run `filter_activities.py` first to create `data_filtered/` directory

**Issue:** Evaluation shows F1=0.0
- **Solution:** Check that predicted and ground truth files have same activity labels

**Issue:** Timeline visualization not showing
- **Solution:** Install matplotlib: `pip install matplotlib`

**Issue:** Model training is very slow
- **Solution:** Use existing model or reduce training data size

---

## Additional Documentation

For detailed information about the Activity Learner (AL) algorithm, including advanced training options, cross-validation, and custom configuration, see `AL/README.md`.