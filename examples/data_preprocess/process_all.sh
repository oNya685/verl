#!/bin/bash
# Batch processing script for all datasets
# This script processes both training and test datasets

set -e  # Exit on error

echo "=========================================="
echo "Starting batch data processing..."
echo "=========================================="

# Process training datasets (each script has os.chdir('..') inside)
echo ""
echo ">>> Processing training datasets..."
python3 dapo.py --local_dir ./data/dapo
python3 math_eval.py --local_dir ./data/math

# Process test datasets (each script has os.chdir('..') inside)
echo ""
echo ">>> Processing test datasets..."
python3 test_aime2024.py --local_dir ./data/aime24
python3 test_aime2025.py --local_dir ./data/aime25
python3 test_amc.py --local_dir ./data/amc
python3 test_math500.py --local_dir ./data/math500
python3 test_minervamath.py --local_dir ./data/minerva
python3 test_olympiad.py --local_dir ./data/olympiad

echo ""
echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Training:"
echo "    - data/dapo/train.parquet"
echo "    - data/math/train.parquet"
echo "  Testing:"
echo "    - data/aime24/test.parquet"
echo "    - data/aime25/test.parquet"
echo "    - data/amc/test.parquet"
echo "    - data/math500/test.parquet"
echo "    - data/minerva/test.parquet"
echo "    - data/olympiad/test.parquet"
