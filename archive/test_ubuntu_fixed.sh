#!/bin/bash
# Quick test script for fixed bond data logger on Ubuntu

echo "🧪 Testing Fixed Bond Data Logger..."
echo "==================================="

# Activate virtual environment
if [ -f "bond_env/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source bond_env/bin/activate
else
    echo "❌ Virtual environment not found. Run ubuntu_setup_fixed.sh first!"
    exit 1
fi

# Check Python packages
echo "🔍 Checking Python packages..."
python3 -c "
try:
    import requests, pandas as pd, numpy as np
    from datetime import datetime, timedelta
    import json, time, os, logging, re, glob
    print('✅ All required packages available')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    exit(1)
"

# Check if fixed logger files exist
missing_files=()
for file in "bond_data_logger_fixed.py" "run_logger_fixed.py" "test_fixed_logger.py"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing fixed logger files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please copy the fixed files from Windows to Ubuntu:"
    echo "1. From WSL: cp /mnt/d/Downloads/QRM/*.py ."
    echo "2. Or use scp/rsync to transfer files"
    exit 1
fi

# Run the test
echo "🚀 Running test with fixed logger..."
python test_fixed_logger.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
    echo "📄 Check test_bond_data.csv for sample output"
else
    echo ""
    echo "❌ Test failed. Check error messages above."
    exit 1
fi
