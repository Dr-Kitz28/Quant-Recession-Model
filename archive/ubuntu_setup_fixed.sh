#!/bin/bash
# Updated Ubuntu setup script for Fixed Bond Data Logger

echo "🐧 Setting up FIXED Bond Data Logger in Ubuntu..."
echo "================================================="

# Step 1: Update system and install Python
echo "📦 Installing Python and dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl

# Step 2: Create virtual environment
echo "🔧 Creating virtual environment 'bond_env'..."
if [ -d "bond_env" ]; then
    echo "⚠️  Removing existing bond_env..."
    rm -rf bond_env
fi

python3 -m venv bond_env
source bond_env/bin/activate

# Step 3: Install required packages
echo "📚 Installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install requests beautifulsoup4 pandas numpy

# Step 4: Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/fbil_par_yields logs

# Step 5: Test package installation
echo "🧪 Testing package installation..."
python3 -c "
import requests, pandas as pd, numpy as np
from datetime import datetime, timedelta
import json, time, os, logging, re, glob
print('✅ All core packages installed successfully!')
print(f'✅ Python version: {__import__('sys').version}')
print(f'✅ Pandas version: {pd.__version__}')
print(f'✅ Requests version: {requests.__version__}')
"

echo ""
echo "✅ Ubuntu setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the fixed Python files to Ubuntu:"
echo "   - bond_data_logger_fixed.py" 
echo "   - run_logger_fixed.py"
echo "   - test_fixed_logger.py"
echo "   - config.py"
echo ""
echo "2. Optional: Place FBIL Par Yield files in data/fbil_par_yields/"
echo ""
echo "3. Test the setup:"
echo "   source bond_env/bin/activate"
echo "   python test_fixed_logger.py"
echo ""
echo "4. Run full collection:"
echo "   python run_logger_fixed.py 2024-08-01 2024-08-07"
echo ""
echo "📍 Current directory: $(pwd)"
echo "📦 Virtual environment: bond_env/bin/activate"
