#!/bin/bash
# Ubuntu runner for bond data logger

echo "🚀 Starting Bond Data Logger..."
echo "=============================="

# Activate virtual environment
if [ -f "bond_env/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source bond_env/bin/activate
else
    echo "⚠️  Virtual environment not found. Using system Python..."
fi

# Check if config exists
if [ ! -f "config.py" ]; then
    echo "❌ config.py not found! Please copy your config file first."
    exit 1
fi

# Check if main logger exists  
if [ ! -f "bond_data_logger.py" ]; then
    echo "❌ bond_data_logger.py not found! Please copy your Python files first."
    exit 1
fi

# Run with date arguments if provided, otherwise default
if [ $# -eq 0 ]; then
    echo "📅 Collecting last 7 days of data..."
    python run_logger.py
elif [ $# -eq 2 ]; then
    echo "📅 Collecting data from $1 to $2..."
    python run_logger.py "$1" "$2"
else
    echo "📖 Usage: $0 [start_date] [end_date]"
    echo "   Example: $0 2024-01-01 2024-08-30"
    echo "   Default: Last 7 days"
    exit 1
fi

echo "✅ Data collection complete!"
echo "📊 Check bond_market_data.csv for results"
echo "📝 Check bond_logger.log for details"
