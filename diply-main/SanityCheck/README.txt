SANITY CHECK VARIANTS
======================================================================

Original file: ./RawData/P001.txt
Created 8 variants for testing

perfect              -> ./SanityCheck\P001_perfect.txt
                        Perfect copy (should get 100% metrics)
small_time_shift     -> ./SanityCheck\P001_small_time_shift.txt
                        Small timestamp shifts (10-15s) - within tolerance
large_time_shift     -> ./SanityCheck\P001_large_time_shift.txt
                        Large timestamp shifts (15-25s) - beyond threshold
boundary_errors      -> ./SanityCheck\P001_boundary_errors.txt
                        Shifted segment boundaries (5-12s)
missing_segments     -> ./SanityCheck\P001_missing_segments.txt
                        30% of segment markers removed
false_positives      -> ./SanityCheck\P001_false_positives.txt
                        10 false segment markers added
label_confusion      -> ./SanityCheck\P001_label_confusion.txt
                        Swapped segment labels
combined_errors      -> ./SanityCheck\P001_combined_errors.txt
                        Multiple error types combined

======================================================================
Usage: Compare each variant against the original file
using the SegmentEvaluator to validate metrics.
