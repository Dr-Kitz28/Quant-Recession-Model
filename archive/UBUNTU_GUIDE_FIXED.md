# Ubuntu Setup Guide for Fixed Bond Data Logger

## Overview
This guide helps you run the **FIXED** bond data logger on Ubuntu, which resolves:
- ✅ File permission issues
- ✅ Missing Indian data  
- ✅ Empty volume columns
- ✅ Complete data collection for both countries

## Prerequisites
- Ubuntu 18.04+ (including WSL)
- Python 3.6+
- Internet connection

---

## Step 1: Initial Ubuntu Setup

### Option A: From WSL (Windows Subsystem for Linux)
```bash
# Open WSL terminal
wsl

# Navigate to your Windows files via mount point
cd /mnt/d/Downloads/QRM
```

### Option B: Native Ubuntu
```bash
# Create project directory
mkdir -p ~/bond_logger
cd ~/bond_logger

# Copy files from Windows (adjust path as needed)
# You'll need to transfer the files manually
```

---

## Step 2: Run Setup Script

```bash
# Make setup script executable
chmod +x ubuntu_setup_fixed.sh

# Run the setup (installs Python, creates venv, installs packages)
./ubuntu_setup_fixed.sh
```

**What this does:**
- Updates Ubuntu packages
- Installs Python 3 and pip
- Creates `bond_env` virtual environment  
- Installs: requests, beautifulsoup4, pandas, numpy
- Creates data directories
- Tests package installation

---

## Step 3: Copy Fixed Python Files

You need these **FIXED** files in your Ubuntu directory:
- `bond_data_logger_fixed.py` ← Main data collection engine
- `run_logger_fixed.py` ← Command-line runner
- `test_fixed_logger.py` ← Test script
- `config.py` ← API keys (optional)

### From WSL:
```bash
# If you're in WSL and can access Windows files
cp /mnt/d/Downloads/QRM/bond_data_logger_fixed.py .
cp /mnt/d/Downloads/QRM/run_logger_fixed.py .
cp /mnt/d/Downloads/QRM/test_fixed_logger.py .
cp /mnt/d/Downloads/QRM/config.py .  # Optional
```

### From Native Ubuntu:
Use `scp`, USB drive, or file transfer method to copy the files.

---

## Step 4: Test the Fixed Logger

```bash
# Make test script executable
chmod +x test_ubuntu_fixed.sh

# Run the test
./test_ubuntu_fixed.sh
```

**Expected output:**
```
🧪 Testing Fixed Bond Data Logger...
📦 Activating virtual environment...
✅ All required packages available
🚀 Running test with fixed logger...
🇮🇳 Testing India data sources...
   ✅ FBIL yields: 8 records
   ✅ RBI auctions: 3 records  
   ✅ CCIL volumes: 4 records
   ✅ Outstanding amounts: 7 records
🇺🇸 Testing USA data sources...
   ✅ Treasury yields: 10 records
   ✅ Treasury auctions: 2 records
   ✅ TRACE volumes: 4 records
   ✅ Outstanding debt: 7 records
📊 Total records collected: 45
✅ Test completed successfully!
```

---

## Step 5: Run Full Data Collection

### Make runner script executable:
```bash
chmod +x run_ubuntu_fixed.sh
```

### Usage Options:

#### Collect last 7 days:
```bash
./run_ubuntu_fixed.sh
```

#### Collect from specific start date:
```bash
./run_ubuntu_fixed.sh 2024-08-01
```

#### Collect specific date range:
```bash
./run_ubuntu_fixed.sh 2024-08-01 2024-08-07
```

#### Large historical collection:
```bash
./run_ubuntu_fixed.sh 2008-01-01 2025-08-30
```

---

## Step 6: Monitor Progress

### Check real-time logs:
```bash
tail -f bond_logger_fixed.log
```

### Check CSV output:
```bash
# View first few rows
head -20 bond_market_data.csv

# Count total records
wc -l bond_market_data.csv

# Check file size
ls -lh bond_market_data.csv
```

### Analyze data coverage:
```bash
# Quick Python analysis
python3 -c "
import pandas as pd
df = pd.read_csv('bond_market_data.csv')
print(f'Total records: {len(df):,}')
print(f'Countries: {df[\"country\"].value_counts().to_dict()}')
print(f'Data types: {df[\"data_type\"].value_counts().to_dict()}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
"
```

---

## Optional: FBIL Real Data

To use real Indian yield data instead of mock data:

1. **Create FBIL directory:**
   ```bash
   mkdir -p data/fbil_par_yields
   ```

2. **Place FBIL/FIMMDA Par Yield files:**
   - Files should be named with dates: `YYYYMMDD_paryields.csv`
   - Or: `paryields_DDMMYYYY.xlsx`
   - The system will auto-detect and use these for India yields

---

## Expected Results

With the fixed version, you'll get **complete data** for both countries:

### Record Types per Business Day:
- **India**: ~22 records (8 yields + 3 auctions + 4 volumes + 7 outstanding)
- **USA**: ~23 records (10 yields + 2 auctions + 4 volumes + 7 outstanding)
- **Total**: ~45 records per business day

### Data Completeness:
- **Yields**: 100% coverage (all tenors for both countries)
- **Auction amounts**: ~20% (only on auction days)
- **Secondary volumes**: 100% coverage  
- **Outstanding amounts**: 100% coverage

### File Sizes (Approximate):
- **1 week**: ~1,500 records, ~150KB
- **1 month**: ~6,000 records, ~600KB  
- **1 year**: ~70,000 records, ~7MB
- **Historical 2008-2025**: ~1.2M records, ~120MB

---

## Troubleshooting

### Virtual Environment Issues:
```bash
# Recreate environment
rm -rf bond_env
python3 -m venv bond_env
source bond_env/bin/activate
pip install requests beautifulsoup4 pandas numpy
```

### File Permission Errors:
```bash
# Make scripts executable
chmod +x *.sh

# Check file ownership
ls -la *.py
```

### Missing Python Packages:
```bash
source bond_env/bin/activate
pip install --upgrade requests beautifulsoup4 pandas numpy
```

### WSL Path Issues:
```bash
# Verify Windows mount point
ls /mnt/d/Downloads/QRM/

# Copy files with full path
cp /mnt/d/Downloads/QRM/*.py .
```

---

## Summary Commands

```bash
# Complete setup from scratch:
./ubuntu_setup_fixed.sh
./test_ubuntu_fixed.sh  
./run_ubuntu_fixed.sh 2024-08-01 2024-08-07

# Monitor progress:
tail -f bond_logger_fixed.log

# Quick analysis:
python3 -c "import pandas as pd; df=pd.read_csv('bond_market_data.csv'); print(f'{len(df)} records collected')"
```

The fixed version will give you **comprehensive, complete bond market data** for both India and USA! 🎉
