#!/usr/bin/env python3
"""
Count all unique sensors per type in the dataset.
"""

import os
import sys
from collections import defaultdict

def count_sensors_by_type(data_dir='./data_filtered_normal'):
    """
    Count unique sensors grouped by type (prefix).

    Args:
        data_dir: Directory containing dataset files
    """
    all_sensors = set()

    # Collect all unique sensor IDs from all files
    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if len(words) >= 3:
                    sensor_id = words[2]
                    all_sensors.add(sensor_id)

    # Group sensors by type (prefix)
    sensors_by_type = defaultdict(list)
    for sensor in sorted(all_sensors):
        # Extract prefix (letters before numbers)
        prefix = ''
        for char in sensor:
            if char.isalpha():
                prefix += char
            else:
                break

        if not prefix:  # Handle edge cases like '023'
            prefix = 'numeric'

        sensors_by_type[prefix].append(sensor)

    # Print results
    print("\n" + "="*80)
    print("SENSOR COUNT BY TYPE")
    print("="*80)
    print(f"{'Type':<15} {'Count':<10} {'Sensors'}")
    print("-"*80)

    total_sensors = 0
    for sensor_type in sorted(sensors_by_type.keys()):
        sensors = sensors_by_type[sensor_type]
        count = len(sensors)
        total_sensors += count
        sensor_list = ', '.join(sensors)
        print(f"{sensor_type:<15} {count:<10} {sensor_list}")

    print("-"*80)
    print(f"{'TOTAL':<15} {total_sensors:<10}")
    print("="*80)

    return sensors_by_type

if __name__ == '__main__':
    data_dir = './data_filtered_normal'

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found!")
        sys.exit(1)

    count_sensors_by_type(data_dir)
