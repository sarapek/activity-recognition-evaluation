"""Quick script to run sanity check evaluation with ROC curves"""
import sys
sys.path.append('..')
from test_evaluation import run_all_sanity_tests
from modify_file import SanityCheckModifier

# Configuration
GROUND_TRUTH_FILE = "../data/P001.txt"
SANITY_CHECK_DIR = "../results/sanity_checks"
ACTIVITY_FILTER = ['1', '2', '3', '4', '5', '6', '7', '8']

# Regenerate sanity check files with activity filter
print("Regenerating sanity check files with activity filter...")
modifier = SanityCheckModifier(seed=42)
modifier.create_sanity_check_variants(
    GROUND_TRUTH_FILE,
    SANITY_CHECK_DIR,
    activity_filter=ACTIVITY_FILTER
)
print("[OK] Sanity check files generated\n")

# Run sanity check tests
print("Running sanity check evaluation with ROC curves...")
all_results = run_all_sanity_tests(
    GROUND_TRUTH_FILE,
    SANITY_CHECK_DIR,
    timeline_visualisation=False,
    start_time_threshold_seconds=2.0,
    show_roc=True,
    activity_filter=ACTIVITY_FILTER
)

print("\n[OK] Sanity check evaluation complete!")
