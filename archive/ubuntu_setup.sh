#!/bin/bash
# Ubuntu setup script for Bond Data Logger

echo "🐧 Setting up Bond Data Logger in Ubuntu..."
echo "============================================"

# Step 1: Update system and install Python
echo "📦 Installing Python and dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Step 2: Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv bond_env
source bond_env/bin/activate

# Step 3: Install required packages
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install requests beautifulsoup4 pandas numpy

# Step 4: Create directory structure
mkdir -p logs data

echo "✅ Ubuntu setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your Python files: bond_data_logger.py, run_logger.py, config.py"
echo "2. Activate environment: source bond_env/bin/activate"
echo "3. Run logger: python run_logger.py"
echo ""
echo "🎯 Quick test: python -c 'import requests, pandas; print(\"✅ All packages ready!\")'"
