"""
Generate test files with diverse, deliberately bad ROC curves
These files will produce AUC values ranging from 0.0 to 0.6
to thoroughly test ROC/AUC calculation correctness.
"""

import random
import numpy as np
from datetime import datetime, timedelta
import os


class BadROCGenerator:
    """Generate prediction files with known bad ROC characteristics"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def read_gt_file(self, filepath):
        """Read ground truth file and extract segments"""
        from segment_evaluator import SegmentEvaluator
        evaluator = SegmentEvaluator()
        df = evaluator.parse_log_file(filepath)
        segments = evaluator.extract_segments(df)
        return segments

    def write_predictions(self, predictions, output_path):
        """Write predictions to file"""
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(pred + '\n')

    def scenario_1_inverted_confidence(self, gt_segments, output_dir):
        """
        Scenario 1: INVERTED CONFIDENCE (AUC < 0.5)
        - True positives get LOW confidence (0.1-0.3)
        - False positives get HIGH confidence (0.7-0.9)
        - This inverts the ROC curve → AUC < 0.5
        """
        print("\nScenario 1: Inverted Confidence")
        print("  Expected AUC: 0.0 - 0.3 (worse than random)")

        predictions = []

        # Detect 50% of GT segments with LOW confidence
        for i, seg in enumerate(gt_segments):
            if i % 2 == 0:  # Only half
                activity = seg['segment_id']
                start = seg['start_time']
                end = seg['end_time']
                sensor = 'M001'

                # Generate with LOW confidence (inverted!)
                confidence = np.random.uniform(0.1, 0.3)

                current = start
                while current <= end:
                    ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                    predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                    current += timedelta(seconds=1.0)

        # Add many false positives with HIGH confidence
        num_fps = len(gt_segments) * 2
        for i in range(num_fps):
            # Random activity
            activity = str(random.randint(1, 8))
            # Time outside GT segments
            base_time = datetime(2008, 1, 1, 12, 0, 0)  # Different time
            start = base_time + timedelta(minutes=i*5)
            end = start + timedelta(minutes=2)
            sensor = 'M001'

            # HIGH confidence for false positives (inverted!)
            confidence = np.random.uniform(0.7, 0.9)

            current = start
            while current <= end:
                ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current += timedelta(seconds=1.0)

        predictions.sort()
        output_path = os.path.join(output_dir, 'bad_roc_inverted_confidence.txt')
        self.write_predictions(predictions, output_path)
        print(f"  Created: {output_path}")
        return output_path

    def scenario_2_random_confidence(self, gt_segments, output_dir):
        """
        Scenario 2: RANDOM CONFIDENCE (AUC ≈ 0.5)
        - All predictions get random confidence
        - No correlation between confidence and correctness
        - Should produce AUC around 0.5 (random guessing)
        """
        print("\nScenario 2: Random Confidence")
        print("  Expected AUC: 0.4 - 0.6 (random classifier)")

        predictions = []

        # Detect 50% of GT with random confidence
        for i, seg in enumerate(gt_segments):
            if i % 2 == 0:
                activity = seg['segment_id']
                start = seg['start_time']
                end = seg['end_time']
                sensor = 'M001'

                # RANDOM confidence
                confidence = np.random.uniform(0.1, 0.9)

                current = start
                while current <= end:
                    ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                    predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                    current += timedelta(seconds=1.0)

        # Add false positives with random confidence
        num_fps = len(gt_segments)
        for i in range(num_fps):
            activity = str(random.randint(1, 8))
            base_time = datetime(2008, 1, 1, 12, 0, 0)
            start = base_time + timedelta(minutes=i*5)
            end = start + timedelta(minutes=2)
            sensor = 'M001'

            # RANDOM confidence for FPs too
            confidence = np.random.uniform(0.1, 0.9)

            current = start
            while current <= end:
                ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current += timedelta(seconds=1.0)

        predictions.sort()
        output_path = os.path.join(output_dir, 'bad_roc_random_confidence.txt')
        self.write_predictions(predictions, output_path)
        print(f"  Created: {output_path}")
        return output_path

    def scenario_3_all_false_positives(self, gt_segments, output_dir):
        """
        Scenario 3: ALL FALSE POSITIVES (AUC ≈ 0.0)
        - Predict segments that don't exist in GT
        - Miss all real GT segments
        - High confidence for all wrong predictions
        """
        print("\nScenario 3: All False Positives")
        print("  Expected AUC: 0.0 - 0.2 (all predictions wrong)")

        predictions = []

        # Don't detect ANY GT segments (all FN)
        # Instead, predict at wrong times with high confidence

        for i in range(len(gt_segments)):
            activity = str(random.randint(1, 8))
            # Different time period from GT
            base_time = datetime(2008, 1, 1, 18, 0, 0)
            start = base_time + timedelta(minutes=i*10)
            end = start + timedelta(minutes=3)
            sensor = 'M001'

            # High confidence for all wrong predictions
            confidence = np.random.uniform(0.7, 0.95)

            current = start
            while current <= end:
                ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current += timedelta(seconds=1.0)

        predictions.sort()
        output_path = os.path.join(output_dir, 'bad_roc_all_false_positives.txt')
        self.write_predictions(predictions, output_path)
        print(f"  Created: {output_path}")
        return output_path

    def scenario_4_wrong_labels_high_confidence(self, gt_segments, output_dir):
        """
        Scenario 4: WRONG LABELS + HIGH CONFIDENCE (AUC ≈ 0.1)
        - Predict at correct times but WRONG activity labels
        - Use high confidence for wrong labels
        - Tests label confusion with bad confidence calibration
        """
        print("\nScenario 4: Wrong Labels + High Confidence")
        print("  Expected AUC: 0.1 - 0.3 (systematic label confusion)")

        predictions = []
        all_activities = sorted(list(set(seg['segment_id'] for seg in gt_segments)))

        for seg in gt_segments:
            # Predict at CORRECT time but WRONG activity
            correct_activity = seg['segment_id']
            wrong_activities = [a for a in all_activities if a != correct_activity]

            if wrong_activities:
                wrong_activity = random.choice(wrong_activities)
            else:
                wrong_activity = str(int(correct_activity) + 1)

            start = seg['start_time']
            end = seg['end_time']
            sensor = 'M001'

            # HIGH confidence for WRONG label (bad calibration)
            confidence = np.random.uniform(0.8, 0.95)

            current = start
            while current <= end:
                ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{ts} {sensor} {sensor} ON {wrong_activity} {confidence:.6f}")
                current += timedelta(seconds=1.0)

        predictions.sort()
        output_path = os.path.join(output_dir, 'bad_roc_wrong_labels.txt')
        self.write_predictions(predictions, output_path)
        print(f"  Created: {output_path}")
        return output_path

    def scenario_5_mostly_low_confidence_fps(self, gt_segments, output_dir):
        """
        Scenario 5: MOSTLY FALSE POSITIVES + LOW CONFIDENCE (AUC ≈ 0.3)
        - Many false positives with low confidence
        - Few true positives with medium confidence
        - Tests FP-heavy scenario
        """
        print("\nScenario 5: Many Low-Confidence False Positives")
        print("  Expected AUC: 0.2 - 0.4 (many false alarms)")

        predictions = []

        # Detect only 20% of GT with medium confidence
        for i, seg in enumerate(gt_segments):
            if i % 5 == 0:  # Only 20%
                activity = seg['segment_id']
                start = seg['start_time']
                end = seg['end_time']
                sensor = 'M001'

                # Medium confidence for the few TPs
                confidence = np.random.uniform(0.5, 0.7)

                current = start
                while current <= end:
                    ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                    predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                    current += timedelta(seconds=1.0)

        # Add MANY false positives with LOW confidence
        num_fps = len(gt_segments) * 5  # 5x more FPs than GT
        for i in range(num_fps):
            activity = str(random.randint(1, 8))
            base_time = datetime(2008, 1, 1, 15, 0, 0)
            start = base_time + timedelta(minutes=i*2)
            end = start + timedelta(minutes=1)
            sensor = 'M001'

            # LOW confidence for FPs
            confidence = np.random.uniform(0.1, 0.4)

            current = start
            while current <= end:
                ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current += timedelta(seconds=1.0)

        predictions.sort()
        output_path = os.path.join(output_dir, 'bad_roc_many_fps.txt')
        self.write_predictions(predictions, output_path)
        print(f"  Created: {output_path}")
        return output_path

    def scenario_6_bimodal_confidence(self, gt_segments, output_dir):
        """
        Scenario 6: BIMODAL CONFIDENCE DISTRIBUTION (AUC ≈ 0.6)
        - TPs and FPs both have bimodal confidence
        - Some TPs low, some high; same for FPs
        - Tests if ROC handles complex distributions
        """
        print("\nScenario 6: Bimodal Confidence Distribution")
        print("  Expected AUC: 0.5 - 0.7 (complex distribution)")

        predictions = []

        # Detect 60% of GT with bimodal confidence
        for i, seg in enumerate(gt_segments):
            if i % 5 != 4:  # Skip 20%
                activity = seg['segment_id']
                start = seg['start_time']
                end = seg['end_time']
                sensor = 'M001'

                # Bimodal: either very low or very high
                if random.random() < 0.5:
                    confidence = np.random.uniform(0.1, 0.3)  # Low mode
                else:
                    confidence = np.random.uniform(0.7, 0.9)  # High mode

                current = start
                while current <= end:
                    ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                    predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                    current += timedelta(seconds=1.0)

        # Add FPs with bimodal distribution too
        num_fps = len(gt_segments) * 2
        for i in range(num_fps):
            activity = str(random.randint(1, 8))
            base_time = datetime(2008, 1, 1, 14, 0, 0)
            start = base_time + timedelta(minutes=i*3)
            end = start + timedelta(minutes=1)
            sensor = 'M001'

            # Bimodal for FPs too
            if random.random() < 0.5:
                confidence = np.random.uniform(0.1, 0.3)
            else:
                confidence = np.random.uniform(0.7, 0.9)

            current = start
            while current <= end:
                ts = current.strftime('%Y-%m-%d %H:%M:%S.%f')
                predictions.append(f"{ts} {sensor} {sensor} ON {activity} {confidence:.6f}")
                current += timedelta(seconds=1.0)

        predictions.sort()
        output_path = os.path.join(output_dir, 'bad_roc_bimodal.txt')
        self.write_predictions(predictions, output_path)
        print(f"  Created: {output_path}")
        return output_path

    def generate_all_scenarios(self, gt_file, output_dir):
        """Generate all bad ROC test scenarios"""
        print("="*70)
        print("GENERATING BAD ROC TEST FILES")
        print("="*70)
        print(f"\nGround truth: {gt_file}")
        print(f"Output directory: {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Read GT segments
        gt_segments = self.read_gt_file(gt_file)
        print(f"\nFound {len(gt_segments)} segments in ground truth")

        # Generate all scenarios
        scenarios = []
        scenarios.append(self.scenario_1_inverted_confidence(gt_segments, output_dir))
        scenarios.append(self.scenario_2_random_confidence(gt_segments, output_dir))
        scenarios.append(self.scenario_3_all_false_positives(gt_segments, output_dir))
        scenarios.append(self.scenario_4_wrong_labels_high_confidence(gt_segments, output_dir))
        scenarios.append(self.scenario_5_mostly_low_confidence_fps(gt_segments, output_dir))
        scenarios.append(self.scenario_6_bimodal_confidence(gt_segments, output_dir))

        print("\n" + "="*70)
        print(f"CREATED {len(scenarios)} BAD ROC TEST FILES")
        print("="*70)
        print("\nExpected AUC ranges:")
        print("  1. Inverted Confidence:        0.0 - 0.3")
        print("  2. Random Confidence:          0.4 - 0.6")
        print("  3. All False Positives:        0.0 - 0.2")
        print("  4. Wrong Labels:               0.1 - 0.3")
        print("  5. Many Low-Conf FPs:          0.2 - 0.4")
        print("  6. Bimodal Distribution:       0.5 - 0.7")
        print("\nThese will thoroughly test ROC/AUC calculation!")
        print("="*70)

        return scenarios


def test_bad_roc_files(gt_file, test_files):
    """Test the generated files and report AUC values"""
    from segment_evaluator import SegmentEvaluator

    print("\n" + "="*70)
    print("TESTING BAD ROC FILES")
    print("="*70)

    evaluator = SegmentEvaluator()

    results = []
    for test_file in test_files:
        filename = os.path.basename(test_file)
        print(f"\nTesting: {filename}")
        print("-" * 70)

        try:
            result = evaluator.evaluate_with_dual_roc(
                test_file,
                gt_file,
                aggregation='average',
                activity_filter=['1', '2', '3', '4', '5', '6', '7', '8']
            )

            # Get average frame-level AUC
            if 'frame_level' in result:
                aucs = [v.get('auc') for v in result['frame_level'].values()
                       if v.get('auc') is not None]
                if aucs:
                    avg_auc = np.mean(aucs)
                    print(f"  Average Frame-Level AUC: {avg_auc:.4f}")
                    results.append((filename, avg_auc))
                else:
                    print(f"  Frame-Level AUC: None (no variation)")
                    results.append((filename, None))

            # Get aggregate metrics
            f1 = result['micro_avg']['f1_score']
            precision = result['micro_avg']['precision']
            recall = result['micro_avg']['recall']
            print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((filename, None))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF BAD ROC TESTS")
    print("="*70)
    print(f"\n{'File':<40} {'AUC':<10} {'Quality'}")
    print("-" * 70)
    for filename, auc_val in results:
        if auc_val is not None:
            if auc_val < 0.3:
                quality = "Terrible (as expected!)"
            elif auc_val < 0.5:
                quality = "Poor (as expected!)"
            elif auc_val < 0.7:
                quality = "Fair (as expected!)"
            else:
                quality = "UNEXPECTED - too high!"

            print(f"{filename:<40} {auc_val:<10.4f} {quality}")
        else:
            print(f"{filename:<40} {'None':<10} No variation")

    print("="*70)
    print("\nIf AUC values match expected ranges, ROC calculation is CORRECT!")
    print("="*70)


if __name__ == "__main__":
    # Configuration
    GT_FILE = "../data/P001.txt"
    OUTPUT_DIR = "../results/bad_roc_tests"

    # Generate test files
    generator = BadROCGenerator(seed=42)
    test_files = generator.generate_all_scenarios(GT_FILE, OUTPUT_DIR)

    # Test them
    test_bad_roc_files(GT_FILE, test_files)
