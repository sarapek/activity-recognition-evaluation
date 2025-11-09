SANITY CHECK VARIANTS WITH CONFIDENCE SCORES
======================================================================

Original file: ../data/P001.txt
Created 8 variants for testing

All variants include confidence scores for ROC curve testing.
Each variant has continuous predictions (every second annotated).
Confidence scores reflect prediction quality:
  - High (0.85-0.95): Perfect predictions
  - Medium (0.50-0.70): Timing errors
  - Low (0.15-0.30): Label errors

perfect              -> ../SanityCheck\P001_perfect.txt
                        Perfect predictions with high confidence (0.85-0.95)
small_time_shift     -> ../SanityCheck\P001_small_time_shift.txt
                        Small timestamp shifts (1-3s), medium-high confidence (0.50-0.70)
large_time_shift     -> ../SanityCheck\P001_large_time_shift.txt
                        Large timestamp shifts (10-20s), medium confidence (0.50-0.70)
boundary_errors      -> ../SanityCheck\P001_boundary_errors.txt
                        Shifted segment boundaries (5-12s), medium confidence (0.50-0.70)
missing_segments     -> ../SanityCheck\P001_missing_segments.txt
                        30% of segments removed (false negatives), medium-high confidence (0.65-0.80)
false_positives      -> ../SanityCheck\P001_false_positives.txt
                        10 false segments added, low-medium confidence (0.30-0.50)
label_confusion      -> ../SanityCheck\P001_label_confusion.txt
                        40% of segment labels swapped, low confidence (0.15-0.30)
combined_errors      -> ../SanityCheck\P001_combined_errors.txt
                        Multiple error types combined, medium confidence (0.40-0.60)

======================================================================
Usage: Compare each variant against the original file
using the SegmentEvaluator to validate metrics and ROC curves.
