"""
Simple runner script for bond data collection
Usage: python run_logger.py [start_date] [end_date]
"""
import sys
from datetime import datetime, timedelta
from bond_data_logger import BondDataLogger, IndiaDataLogger, USADataLogger
import logging

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bond_logger.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=7)  # Default: last week
    
    if len(sys.argv) >= 3:
        end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d') 
    else:
        end_date = datetime.now()
    
    logger.info(f"🚀 Starting Bond Data Collection")
    logger.info(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize loggers
    main_logger = BondDataLogger("bond_market_data.csv")
    india_logger = IndiaDataLogger(main_logger)
    usa_logger = USADataLogger(main_logger)
    
    total_records = 0
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            logger.info(f"📊 Collecting data for {current_date.strftime('%Y-%m-%d')}")
            
            daily_records = []
            
            try:
                # India data
                logger.info("🇮🇳 Collecting India data (FBIL/FIMMDA → CCIL ZCYC → RBI fallback)...")
                # Prefer FBIL/FIMMDA Par Yield files for constant-maturity series
                in_yields = india_logger.get_fbil_par_yields(current_date)
                if not in_yields:
                    # Use CCIL ZCYC local fallback if present (extends history)
                    in_yields = india_logger.get_ccil_zcyc_yields(current_date)
                if not in_yields:
                    # Fallback to RBI reference if neither FBIL nor CCIL ZCYC available
                    in_yields = india_logger.get_rbi_yields(current_date)
                daily_records.extend(in_yields)

                daily_records.extend(india_logger.get_ccil_volumes(current_date))
                daily_records.extend(india_logger.get_rbi_auctions(current_date))
                daily_records.extend(india_logger.get_outstanding_amounts(current_date))
                # Optional: CCIL indicative yields for corroboration (not appended to final CSV)
                try:
                    _ = india_logger.get_ccil_indicative_yields(current_date)
                except Exception:
                    pass
                
                # USA data  
                logger.info("🇺🇸 Collecting USA data...")
                daily_records.extend(usa_logger.get_treasury_yields(current_date))
                daily_records.extend(usa_logger.get_treasury_auctions(current_date))
                daily_records.extend(usa_logger.get_trace_volumes(current_date))
                daily_records.extend(usa_logger.get_outstanding_debt(current_date))
                
                # Log collected data
                if daily_records:
                    main_logger.log_data(daily_records)
                    total_records += len(daily_records)
                    logger.info(f"✅ Logged {len(daily_records)} records")
                else:
                    logger.info("ℹ️  No data available for this date")
                
            except Exception as e:
                logger.error(f"❌ Error processing {current_date.strftime('%Y-%m-%d')}: {e}")
            
            # Rate limiting
            import time
            time.sleep(1)
        
        current_date += timedelta(days=1)
    
    logger.info(f"🎉 Collection completed! Total records: {total_records}")
    logger.info(f"📄 Output file: bond_market_data.csv")
    
    # Show summary
    try:
        import pandas as pd
        df = pd.read_csv("bond_market_data.csv")
        
        logger.info(f"📊 Final dataset summary:")
        logger.info(f"   - Total rows: {len(df):,}")
        logger.info(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"   - Countries: {', '.join(df['country'].unique())}")
        logger.info(f"   - Data types: {', '.join(df['data_type'].unique())}")
        logger.info(f"   - Tenors: {', '.join(sorted(df['tenor'].unique()))}")
        
        # Coverage by country and data type
        coverage = df.groupby(['country', 'data_type']).size().unstack(fill_value=0)
        logger.info(f"📈 Data coverage by type:")
        logger.info(f"\n{coverage}")
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()
