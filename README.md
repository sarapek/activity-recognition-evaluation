# Activity Recognition & Evaluation System

This repository uses the Activity Learner (AL) - Smart Home Edition, developed by Dr. Diane J. Cook (School of Electrical Engineering and Computer Science, Washington State University). While AL is employed here to learn and label activity models from ambient sensor data, the primary purpose of this repository is to evaluate and analyze the performance of the AL algorithm.

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
│   ├── run_20241019_143022_5fold_overlap0.500/
│   │   ├── config.json
│   │   ├── cross_validation_summary.json
│   │   ├── roc_all_folds_averaged.png
│   │   ├── fold_1/
│   │   │   ├── fold_summary.json
│   │   │   ├── roc_fold1.png
│   │   │   ├── fold1_pred_0_P001.txt
│   │   │   └── timeline_P001.png
│   │   ├── fold_2/
│   │   └── ...
│   └── run_20241020_091534_5fold_overlap0.500/
│       └── ...
├── AL/
│   ├── al.py              # Main activity learning script
│   ├── README.md          # Detailed AL documentation
│   └── model/             # Trained models saved here
├── evaluation/
│   ├── segment_evaluator.py
│   ├── test_evaluation.py
│   ├── modify_file.py
│   └── visualize_segments_timeline.py
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

### Two Main Evaluation Scripts

This system provides two complementary evaluation scripts:

1. **`train_and_evaluate.py`** - For real model training and testing
2. **`test_evaluation.py`** - For validating that your evaluation metrics work correctly

---

### 1. Real Model Evaluation: `train_and_evaluate.py`

This script provides an automated end-to-end workflow for training activity recognition models and evaluating their real-world performance. Use this when you want to **assess actual model performance on real data**.

#### Basic Usage

```bash
python train_and_evaluate.py
```

#### What It Does

1. **Data Splitting**: Automatically creates stratified k-fold cross-validation splits
2. **Model Training**: Trains a separate model for each fold
3. **Prediction**: Annotates test files with predicted activity labels
4. **Evaluation**: Computes comprehensive metrics comparing predictions vs ground truth
5. **Visualization**: Optionally generates timeline plots and ROC curves
6. **Aggregation**: Combines results across all folds with mean ± std statistics

#### Interactive Options

When you run the script, you'll be prompted:

**1. Visualization**
```
Enable timeline visualization? (y/n):
```
- **Yes (y)**: Creates visual timeline plots comparing predictions vs ground truth
- **No (n)**: Text-only output (faster)

**2. Overlap Threshold**
```
Overlap threshold (0.0-1.0):
```
- Sets the minimum overlap ratio for segment matching
- Default: 0.5 (50% overlap required)
- Higher values = stricter matching criteria

#### Cross-Validation Process

The script performs **stratified 5-fold cross-validation**:

1. **Fold Creation**: Files are split into 5 folds, ensuring balanced activity distribution
2. **Training**: For each fold, 4 folds are used for training
3. **Testing**: Remaining fold is used for testing
4. **Evaluation**: Comprehensive metrics calculated for each fold
5. **Aggregation**: Results averaged across all folds

#### Output Structure

Creates a **timestamped run directory**:

```
./output/run_YYYYMMDD_HHMMSS_5fold_overlap0.500/
├── config.json                          # Run configuration
├── cross_validation_summary.json        # Aggregated metrics across all folds
├── roc_all_folds_averaged.png          # Averaged ROC curves
├── fold_1/
│   ├── fold_summary.json               # Fold 1 detailed results
│   ├── roc_fold1.png                   # Fold 1 ROC curves
│   ├── fold1_pred_0_P001.txt          # Predictions for test file 0
│   ├── fold1_pred_1_P002.txt          # Predictions for test file 1
│   └── timeline_P001.png               # Timeline visualization (if enabled)
├── fold_2/
│   └── ...
└── fold_5/
    └── ...
```

**Key Output Files:**

- **`config.json`**: Records all settings (n_folds, overlap_threshold, resolution, etc.)
- **`cross_validation_summary.json`**: Aggregate statistics (mean ± std) for all metrics across folds
- **`roc_all_folds_averaged.png`**: Frame-level and segment-level ROC curves averaged across all folds
- **`fold_N/fold_summary.json`**: Detailed per-class metrics for fold N
- **`fold_N/roc_foldN.png`**: ROC curves for fold N
- **`fold_N/fold1_pred_*.txt`**: Model predictions for each test file in fold N
- **`fold_N/timeline_*.png`**: Visual timelines (if enabled)

#### Example Output

```
======================================================================
FOLD 1/5
======================================================================
Training: 16 files (filtered)
Testing: 4 files (unfiltered)

[OK] Model trained successfully
[OK] Generated predictions for 4 files

Evaluating fold 1...
======================================================================
TIME-CONTINUOUS MULTI-CLASS EVALUATION REPORT
======================================================================

Timeline Duration: 86400 seconds
Resolution: 1.0s
Activity Classes: 1, 2, 3, 4, 5, 6, 7, 8

======================================================================
AGGREGATE METRICS
======================================================================

--- Macro-Average (simple average across classes) ---
Precision: 0.8234
Recall:    0.7891
F1 Score:  0.8059

--- Micro-Average (pooled TP/FP/FN across classes) ---
Precision: 0.8456
Recall:    0.8123
F1 Score:  0.8286
Total TP Duration: 12456.0s
Total FP Duration: 2234.0s
Total FN Duration: 2890.0s

======================================================================
PER-CLASS METRICS
======================================================================

--- Activity 1 ---
  Ground Truth Duration: 3456.0s
  Predicted Duration:    3234.0s
  TP Duration:           2890.0s
  FP Duration:           344.0s
  FN Duration:           566.0s
  Precision:             0.8937
  Recall:                0.8362
  F1 Score:              0.8640
  IoU:                   0.7602
  Frame-Level AUC:       0.9123
  Segment-Level AUC:     0.8845

[... Activities 2-8 ...]

======================================================================
CROSS-FOLD COMPARISON
======================================================================
Metric               Mean        Std        Min        Max
----------------------------------------------------------------------
Micro Precision    0.8456     0.0234     0.8123     0.8789
Micro Recall       0.8123     0.0345     0.7654     0.8567
Micro F1           0.8286     0.0289     0.7891     0.8678

======================================================================
PER-ACTIVITY METRICS ACROSS FOLDS
======================================================================

--- Activity 1 ---
  Precision:     0.8937 ± 0.0234
  Recall:        0.8362 ± 0.0345
  F1 Score:      0.8640 ± 0.0289
  Frame AUC:     0.9123 ± 0.0156
  Segment AUC:   0.8845 ± 0.0178

[... Activities 2-8 ...]

[OK] Cross-validation results saved:
  - Aggregated summary: ./cross_validation_summary.json
  - Per-fold summaries: ./fold_N/fold_summary.json
```

#### Metrics Explained

**Frame-Level ROC**: Evaluates detection at each time unit (1-second resolution)
- Treats each second as an independent classification decision
- Confidence varies within segments
- Good for assessing real-time detection capability

**Segment-Level ROC**: Evaluates detection of entire activity segments
- Based on segment overlap ratios
- One score per segment instance
- Good for assessing activity instance detection

**Time-Continuous Metrics**: Duration-based precision/recall/F1
- TP Duration: Time correctly classified
- FP Duration: Time incorrectly classified as activity
- FN Duration: Activity time missed

#### When to Use This Script

**Use `train_and_evaluate.py` when:**
- Training a new model architecture or features
- Evaluating model performance on real sensor data
- Comparing different training datasets or parameters
- Conducting robust cross-validation experiments
- Generating publication-ready performance metrics
- You need actual prediction results for downstream analysis

**Key Features:**
- **Stratified K-Fold Cross-Validation**: Ensures balanced activity representation in each fold
- **Dual ROC Curves**: Both frame-level (per-time-unit) and segment-level (overlap-based)
- **Comprehensive Metrics**: Precision, Recall, F1, IoU, AUC per activity class
- **Temporal Analysis**: Evaluates timing accuracy at 1-second resolution
- **Aggregated Statistics**: Mean ± std across all folds for robust estimates
- **Vertical ROC Averaging**: Proper statistical aggregation of ROC curves across folds

---

### 2. Evaluation Testing: `test_evaluation.py`

This script validates that your evaluation metrics are working correctly by testing them against **synthetic data with known properties**. Use this when you want to **verify your evaluation system** before trusting real model results.

#### Basic Usage

```bash
cd evaluation
python test_evaluation.py
```

#### What It Does

Creates 8 synthetic test variants from ground truth data, each with specific known errors:

1. **perfect** - Exact copy (should get F1 ≈ 1.0)
2. **small_time_shift** - ±1-5 second timing errors
3. **large_time_shift** - ±10-30 second timing errors
4. **boundary_errors** - Incorrect start/end times
5. **missing_segments** - 20% of activities removed (should show low recall)
6. **false_positives** - Extra incorrect detections added (should show low precision)
7. **label_confusion** - Some activity labels swapped
8. **combined_errors** - Multiple error types mixed

#### Interactive Options

**1. Create Sanity Check Files**
```
Create sanity check files? (y/n):
```
- **Yes**: Generates all 8 test variants in `./SanityCheck/` directory
- **No**: Uses existing files (faster if you've already created them)

**2. Timeline Visualization**
```
Enable segment timeline visualization? (y/n):
```
- **Yes**: Shows visual comparison for each variant
- **No**: Text-only metrics

**3. ROC Curves**
```
Generate ROC curves? (y/n):
```
- **Yes**: Generates ROC curves for each variant
- **No**: Skips ROC generation

**4. Test Mode Selection**
```
Choose test mode:
1. Run full sanity check test suite
2. Quick test single file
3. Test with confidence scores (SVM mode)
```
- **Option 1**: Tests all 8 variants (recommended for validation)
- **Option 2**: Tests one specific prediction file
- **Option 3**: Tests with confidence scores for AUC calculation

#### Example Output

```
======================================================================
SANITY CHECK EVALUATION TEST SUITE
======================================================================

======================================================================
Testing: PERFECT
======================================================================

TIME-CONTINUOUS MULTI-CLASS EVALUATION REPORT
----------------------------------------------------------------------
Activity Classes: 1, 2, 3, 4, 5, 6, 7, 8

AGGREGATE METRICS
----------------------------------------------------------------------
Macro F1:    1.0000
Micro F1:    1.0000
Precision:   1.0000
Recall:      1.0000

PER-CLASS METRICS
----------------------------------------------------------------------
--- Activity 1 ---
  Precision:     1.0000
  Recall:        1.0000
  F1 Score:      1.0000
  Frame AUC:     1.0000
  Segment AUC:   1.0000

[... Activities 2-8 ...]

======================================================================
Testing: MISSING_SEGMENTS
======================================================================

--- Activity 2 ---
  Precision:     1.0000
  Recall:        0.8000  # <- Expected: 20% segments removed
  F1 Score:      0.8889

======================================================================
Testing: FALSE_POSITIVES
======================================================================

--- Activity 3 ---
  Precision:     0.7500  # <- Expected: Extra false detections
  Recall:        1.0000
  F1 Score:      0.8571

======================================================================
SUMMARY REPORT - ALL VARIANTS
======================================================================

Variant              F1      Precision    Recall   Frame AUC  Seg AUC
--------------------------------------------------------------------------------
perfect            1.0000     1.0000     1.0000      1.000      1.000
small_time_shift   0.9856     0.9823     0.9889      0.995      0.992
large_time_shift   0.8234     0.8567     0.7923      0.876      0.854
boundary_errors    0.9123     0.9345     0.8912      0.923      0.918
missing_segments   0.8889     1.0000     0.8000      0.900      0.850
false_positives    0.8571     0.7500     1.0000      0.875      0.820
label_confusion    0.7234     0.7567     0.6923      0.745      0.728
combined_errors    0.6789     0.6912     0.6667      0.698      0.672

Detailed results saved to: ./SanityCheck/evaluation_results.json
```

#### Understanding Sanity Check Results

**What to Look For:**

- **perfect**: Should get F1 = 1.0, all metrics perfect
  - If not: Evaluation code has bugs
  
- **small_time_shift**: Should have high F1 (> 0.95) with slightly lower AUC
  - Tests tolerance to minor timing errors
  
- **missing_segments**: Should have high precision (1.0), lower recall (0.8)
  - Tests detection of segment removal
  
- **false_positives**: Should have low precision (< 0.8), high recall (1.0)
  - Tests detection of spurious segments
  
- **label_confusion**: Should have moderate F1 (0.6-0.8)
  - Tests handling of classification errors

#### When to Use This Script

**Use `test_evaluation.py` when:**
- **Initial Setup**: Verifying evaluation metrics work correctly before trusting them
- **Debugging**: Understanding why certain metrics behave unexpectedly
- **Development**: Testing changes to evaluation code
- **Documentation**: Demonstrating how metrics respond to specific error types
- **Validation**: Confirming evaluation thresholds (overlap, timing) are appropriate

**This script helps answer questions like:**
- "Does my evaluator correctly detect perfect predictions?" (perfect variant → F1=1.0?)
- "How sensitive are my metrics to timing errors?" (compare small vs large time shifts)
- "Can I distinguish between precision vs recall issues?" (missing_segments vs false_positives)
- "Are my ROC curves calculated correctly?" (all variants should have reasonable AUCs)
- "Is my overlap threshold appropriate?" (check segment_level AUC sensitivity)

---

### Comparison: When to Use Each Script

| Scenario | Use This Script | Why |
|----------|----------------|-----|
| Training a new model | `train_and_evaluate.py` | Real model performance on real data |
| Evaluating model improvements | `train_and_evaluate.py` | Compare actual predictions across versions |
| First-time setup | `test_evaluation.py` | Verify evaluation system works correctly |
| Debugging weird metrics | `test_evaluation.py` | Understand how metrics respond to known errors |
| Cross-validation experiments | `train_and_evaluate.py` | Robust performance estimates |
| Testing evaluation changes | `test_evaluation.py` | Validate metric calculations |
| Publication results | `train_and_evaluate.py` | Real model metrics for reporting |
| Understanding metric behavior | `test_evaluation.py` | Educational/diagnostic analysis |

**Recommended Workflow:**
1. **First**: Run `test_evaluation.py` to verify your evaluation system
2. **Then**: Run `train_and_evaluate.py` to get real model performance
3. **If metrics look weird**: Go back to `test_evaluation.py` to debug

---

### Manual Training & Testing

For more control over individual steps, you can use `al.py` directly. See `AL/README.md` for detailed documentation on:
- Training models with custom parameters
- Cross-validation
- Leave-one-out testing
- Annotating unlabeled data
- Activity and sensor filtering options

---

## Evaluation Tools

### Segment Evaluator (`segment_evaluator.py`)

The core evaluation engine that computes metrics for activity recognition.

#### Basic Usage

```python
from segment_evaluator import SegmentEvaluator

# Initialize evaluator
evaluator = SegmentEvaluator(
    timeline_resolution_seconds=1.0,  # Frame resolution
    start_time_threshold_seconds=0,   # Detection latency threshold
    overlap_threshold=0.5              # Segment overlap threshold
)

# Evaluate with dual ROC curves
results = evaluator.evaluate_with_dual_roc(
    predicted_file='./predictions.txt',
    ground_truth_file='./ground_truth.txt',
    aggregation='average'  # How to aggregate multiple events per frame
)

# Print report
evaluator.print_report(results, activity_filter=['1','2','3','4','5','6','7','8'])

# Plot ROC curves
evaluator.plot_dual_roc_curves(
    results,
    save_path='roc_curves.png',
    activity_filter=['1','2','3','4','5','6','7','8']
)
```

#### Parameters Explained

**`timeline_resolution_seconds`** (default: 1.0)
- Time resolution for frame-level analysis
- 1.0 = analyze predictions at 1-second intervals
- Lower values = finer temporal resolution but slower

**`start_time_threshold_seconds`** (default: 0)
- Maximum acceptable detection delay
- If activity detected >threshold seconds late, marked as "too late"
- 0 = no latency tolerance

**`overlap_threshold`** (default: 0.5)
- Minimum overlap ratio for segment matching
- 0.5 = require 50% overlap to consider segments matched
- Higher values = stricter matching

**`aggregation`** (default: 'average')
- How to handle multiple events in same time frame
- Options: 'average', 'max', 'median'
- Affects confidence scores in frame-level ROC

#### Metrics Reported

**Aggregate Metrics:**
- **Macro-Average**: Simple mean across activity classes
- **Micro-Average**: Pooled TP/FP/FN across all classes
- **Weighted-Average**: Weighted by ground truth duration

**Per-Class Metrics:**
- **TP/FP/FN Duration**: Time correctly/incorrectly classified
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union
- **Frame-Level AUC**: ROC curve at time-unit resolution
- **Segment-Level AUC**: ROC curve for segment detection

---

## Common Workflows

### Complete Pipeline from Raw Data (Recommended)

```bash
# Step 1: Filter raw data
python filter_activities.py

# Step 2: Validate evaluation system
cd evaluation
python test_evaluation.py
# Choose: Create files (y), Visualize (n), ROC (y), Mode 1
cd ..

# Step 3: Run cross-validation
python train_and_evaluate.py
# Choose: Visualize (y), Overlap threshold (0.5)

# Results are saved in: ./output/run_YYYYMMDD_HHMMSS_5fold_overlap0.500/
```

### Quick Testing with Sanity Checks

```bash
# Generate and test sanity check variants
cd evaluation
python test_evaluation.py

# Choose options:
# 1. y - Create sanity check files
# 2. y - Enable visualization
# 3. y - Generate ROC curves
# 4. 1 - Run full test suite
```

### Custom Evaluation

```python
from evaluation.segment_evaluator import SegmentEvaluator

# Initialize with custom settings
evaluator = SegmentEvaluator(
    timeline_resolution_seconds=0.5,  # Higher resolution
    overlap_threshold=0.7              # Stricter matching
)

# Evaluate multiple files
results = evaluator.evaluate_multiple_files(
    predicted_files=['pred1.txt', 'pred2.txt'],
    ground_truth_files=['gt1.txt', 'gt2.txt'],
    aggregation='max'
)

# Print report
evaluator.print_report(results, activity_filter=['1','2','3'])

# Plot aggregated ROC
evaluator.plot_dual_roc_curves(
    results,
    save_path='custom_roc.png',
    activity_filter=['1','2','3']
)
```

---

## Understanding Evaluation Results

### Good Performance Indicators
- **F1 Score >= 0.85** - Strong overall performance
- **Recall >= 0.80** - Most activities detected
- **Frame AUC >= 0.90** - Good temporal discrimination
- **Segment AUC >= 0.85** - Good segment detection

### Common Issues and Diagnosis

| Symptom | Likely Cause | Check This |
|---------|-------------|------------|
| Low Recall, High Precision | Missing segments | `missing_segments` sanity check |
| Low Precision, High Recall | Too many false positives | `false_positives` sanity check |
| Low Frame AUC, Good F1 | Confidence scores not calibrated | Check confidence distributions |
| Low Segment AUC, Good F1 | Overlap threshold too strict | Try lower overlap_threshold |
| All metrics poor | Model not trained properly | Check training data quality |

---

## File Formats

### Annotated Sensor Data Format (Ground Truth)
```
YYYY-MM-DD HH:MM:SS.mmmmmm SENSOR_ID STATE ACTIVITY_LABEL
```

Example:
```
2008-02-27 00:07:46.898610 M003 ON 1
2008-02-27 00:10:32.539722 M004 OFF 2
```

### Annotated Sensor Data with Confidence (Predictions)
```
YYYY-MM-DD HH:MM:SS.mmmmmm SENSOR_ID NEW_SENSOR STATE ACTIVITY_LABEL CONFIDENCE
```

Example:
```
2008-02-27 00:07:46.898610 M003 M003 ON 1 0.945
2008-02-27 00:10:32.539722 M004 M004 OFF 2 0.876
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
- **Solution:** Reduce number of training files or use existing model

**Issue:** Sanity checks failing (perfect != 1.0)
- **Solution:** Debug evaluation code, check overlap_threshold and resolution settings

**Issue:** ROC curves look weird
- **Solution:** Run `test_evaluation.py` to understand expected behavior

---

## Additional Documentation

For detailed information about the Activity Learner (AL) algorithm, including advanced training options, cross-validation, and custom configuration, see `AL/README.md`.