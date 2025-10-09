"""
Bond Data Configuration
Set your API keys and data sources here
"""

# API Keys (add your own)
FRED_API_KEY = "f5d7266bd56754a5ad95c8d55b35b040"  # Get from https://fred.stlouisfed.org/docs/api/api_key.html

# Data Sources Configuration
INDIA_SOURCES = {
    'yields': 'https://rbi.org.in/Scripts/ReferenceRateDisplay.aspx',
    'auctions': 'https://rbi.org.in/Scripts/AuctionResultDisplay.aspx', 
    'volumes': 'https://www.ccil.co.in/api/statistics/tradingsummary',
    'outstanding': 'https://www.ccil.co.in/api/statistics/outstanding'
}

USA_SOURCES = {
    'yields': 'https://api.stlouisfed.org/fred',
    'auctions': 'https://api.fiscaldata.treasury.gov/services/api/v1',
    'volumes': 'https://api.finra.org/trace/aggregate',
    'outstanding': 'https://api.fiscaldata.treasury.gov/services/api/v1'
}

# Standard tenor buckets for both countries
STANDARD_TENORS = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']

# Collection settings
COLLECTION_SETTINGS = {
    'start_date': '2020-01-01',  # Start from this date
    'rate_limit_seconds': 1,     # Delay between requests
    'retry_attempts': 3,         # Retry failed requests
    'weekend_skip': True         # Skip weekends
}

# Optional toggles
# If False, skip India HTTP calls (use only local drop-ins if present)
INDIA_HTTP_ENABLED = True

# Do not attempt CCIL online endpoints before this date (local files still used if present)
# CCIL OTC outright reporting on NDS-OM is consistent from Apr-2013
CCIL_ONLINE_START_DATE = '2013-04-01'

# Enable FINRA TRACE volumes for US (optional; may require proper headers/auth in some environments)
ENABLE_FINRA_API = True
