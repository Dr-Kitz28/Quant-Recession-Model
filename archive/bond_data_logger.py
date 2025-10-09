"""
Bond Market Data Logger - India & USA
Scrapes and logs yield, auction, secondary volume, and outstanding data
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import time
import os
from typing import Dict, List, Optional
import logging
import re
import glob
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BondDataLogger:
    def __init__(self, output_file: str = "bond_market_data.csv"):
        self.output_file = output_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(output_file):
            self.initialize_csv()
    
    def initialize_csv(self):
        """Initialize CSV with proper headers"""
        headers = [
            'date', 'country', 'tenor', 'data_type',
            'yield_pct', 'auction_amount_mn', 'secondary_volume_mn', 
            'outstanding_amount_mn', 'bid_to_cover', 'auction_type',
            'source', 'timestamp'
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Initialized {self.output_file} with headers")
    
    def log_data(self, data_records: List[Dict]):
        """Append data records to CSV"""
        if not data_records:
            return
        
        df = pd.DataFrame(data_records)
        
        # Add timestamp
        df['timestamp'] = datetime.now().isoformat()
        
        # Append to existing CSV
        df.to_csv(self.output_file, mode='a', header=False, index=False)
        logger.info(f"Logged {len(data_records)} records to {self.output_file}")

class IndiaDataLogger:
    def __init__(self, main_logger: BondDataLogger):
        self.logger = main_logger
        self.base_urls = {
            'rbi': 'https://rbi.org.in',
            'ccil': 'https://www.ccil.co.in',
            'fimmda': 'https://www.fimmda.org'
        }
        # Config toggles
        try:
            import config
            self.india_http_enabled = getattr(config, 'INDIA_HTTP_ENABLED', True)
            self.ccil_online_start = datetime.strptime(getattr(config, 'CCIL_ONLINE_START_DATE', '2018-01-01'), '%Y-%m-%d').date()
        except Exception:
            self.india_http_enabled = True
            self.ccil_online_start = datetime(2018, 1, 1).date()
        # Local data folders for manual drop-in (robust fallback)
        self.local_dirs = {
            'fbil_par_yields': Path('data/fbil_par_yields'),
            'ccil_zcyc': Path('data/ccil_zcyc'),
            'ccil_indicative': Path('data/ccil_indicative_yields'),
            'ccil_outstanding': Path('data/ccil_outstanding'),
            'ccil_trading': Path('data/ccil_trading_summary')
        }
        # RBI auctions local snapshots
        self.local_dirs['rbi_auctions'] = Path('data/rbi_auctions')
        # Ensure directories exist
        for p in self.local_dirs.values():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def get_fbil_par_yields(self, date: datetime) -> List[Dict]:
        """Parse FBIL/FIMMDA daily Par Yield file for standard tenors.

        Strategy:
        - Prefer local file drop-in under data/fbil_par_yields (csv/xlsx/xls)
        - Attempt to infer tenors from columns or rows (wide or long format)
        - Target tenors: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y
        - Return long-format records with data_type='yield', source='FBIL_PAR'
        """
        recs: List[Dict] = []

        # 1) Try local file first (recommended)
        f = self._find_local_file(self.local_dirs['fbil_par_yields'], date)
        if f is None:
            logger.info(
                "FBIL par-yield file not found locally for %s. "
                "Drop daily files into data/fbil_par_yields/ to enable offline parsing.",
                date.strftime('%Y-%m-%d')
            )
            return recs

        try:
            df = self._read_tabular_file(f)
            if df is None or df.empty:
                return recs

            # Normalize columns
            df_cols = {c: self._norm(c) for c in df.columns}
            norm_to_orig = {self._norm(c): c for c in df.columns}

            # Two common shapes:
            # A) Wide: one row per date, columns are tenors (e.g., '3M', '6M', '1Y', ...)
            # B) Long: columns like ['tenor','par yield','date']

            target_tenors = ['3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','15Y','20Y','30Y']

            # Try wide format first
            tenor_cols = []
            for t in target_tenors:
                c = self._find_tenor_col(df.columns, t)
                if c:
                    tenor_cols.append((t, c))

            # Identify date column if present
            date_col = None
            for cand in ['date', 'asofdate', 'valuationdate']:
                n = self._norm(cand)
                if n in df_cols.values():
                    date_col = norm_to_orig[n]
                    break
                # Try direct presence
                for col in df.columns:
                    if self._norm(col) == n:
                        date_col = col
                        break

            effective_row = None
            if tenor_cols:
                if date_col and date_col in df.columns:
                    # pick exact date row if available
                    mask = pd.to_datetime(df[date_col], errors='coerce').dt.date == date.date()
                    if mask.any():
                        effective_row = df.loc[mask].iloc[0]
                    else:
                        # fallback last row
                        effective_row = df.iloc[-1]
                else:
                    # No date column, assume last row is the latest
                    effective_row = df.iloc[-1]

                for t, col in tenor_cols:
                    val = self.safe_float(effective_row[col])
                    if val is not None:
                        recs.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'country': 'India',
                            'tenor': t,
                            'data_type': 'yield',
                            'yield_pct': val,
                            'auction_amount_mn': None,
                            'secondary_volume_mn': None,
                            'outstanding_amount_mn': None,
                            'bid_to_cover': None,
                            'auction_type': None,
                            'source': 'FBIL_PAR'
                        })
                return recs

            # Try long format
            tenor_col = self._find_col(df.columns, ['tenor','bucket','maturity'])
            yield_col = self._find_col(df.columns, ['paryield','par_yield','yield','par_rate'])
            if tenor_col and yield_col:
                subset = df[[tenor_col, yield_col]].copy()
                for _, row in subset.iterrows():
                    std = self._map_tenor(row[tenor_col])
                    if std in target_tenors:
                        val = self.safe_float(row[yield_col])
                        if val is not None:
                            recs.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'country': 'India',
                                'tenor': std,
                                'data_type': 'yield',
                                'yield_pct': val,
                                'auction_amount_mn': None,
                                'secondary_volume_mn': None,
                                'outstanding_amount_mn': None,
                                'bid_to_cover': None,
                                'auction_type': None,
                                'source': 'FBIL_PAR'
                            })
                return recs

        except Exception as e:
            logger.error(f"Error parsing FBIL Par Yield file for {date:%Y-%m-%d}: {e}")

        return recs

    def get_ccil_zcyc_yields(self, date: datetime) -> List[Dict]:
        """Parse CCIL ZCYC (or older curve) from local files under data/ccil_zcyc.

        Accepts CSV/XLS/XLSX. Supports both wide (columns are tenors) and long (maturity/tenor + yield) shapes.
        Returns records with data_type='yield' and source='CCIL_ZCYC_LOCAL'.
        """
        recs: List[Dict] = []
        f = self._find_local_file(self.local_dirs['ccil_zcyc'], date)
        if f is None:
            return recs
        try:
            df = self._read_tabular_file(f)
            if df is None or df.empty:
                return recs
            # Try wide: columns include labels like 1M, 3M, 6M, 1Y, ...
            target = ['1M','3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','15Y','20Y','30Y']
            found_cols = []
            for t in target:
                c = self._find_tenor_col(df.columns, t)
                if c:
                    found_cols.append((t, c))
            date_col = self._find_col(df.columns, ['date','asof','valuationdate'])
            if found_cols:
                row = df.iloc[-1] if not date_col else (df.loc[pd.to_datetime(df[date_col], errors='coerce').dt.date == date.date()].iloc[0] if (pd.to_datetime(df[date_col], errors='coerce').dt.date == date.date()).any() else df.iloc[-1])
                for t, c in found_cols:
                    val = self.safe_float(row[c])
                    if val is not None:
                        recs.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'country': 'India',
                            'tenor': t,
                            'data_type': 'yield',
                            'yield_pct': val,
                            'auction_amount_mn': None,
                            'secondary_volume_mn': None,
                            'outstanding_amount_mn': None,
                            'bid_to_cover': None,
                            'auction_type': None,
                            'source': 'CCIL_ZCYC_LOCAL'
                        })
                return recs
            # Try long: maturity/tenor + yield
            tenor_col = self._find_col(df.columns, ['tenor','bucket','maturity'])
            ycol = self._find_col(df.columns, ['yield','par_yield','zero','rate'])
            if tenor_col and ycol:
                for _, r in df.iterrows():
                    t_raw = r[tenor_col]
                    t_std = self._map_tenor(t_raw)
                    if not t_std and pd.notna(t_raw):
                        # try numeric days/years
                        s = str(t_raw).strip().lower()
                        m = None
                        if s.endswith('d'):
                            try:
                                d = float(re.sub(r'[^0-9\.]','', s))
                                if 20 <= d <= 45:
                                    m = '1M'
                                elif 60 <= d <= 120:
                                    m = '3M'
                                elif 150 <= d <= 220:
                                    m = '6M'
                                elif 330 <= d <= 400:
                                    m = '1Y'
                            except Exception:
                                pass
                        if m is None:
                            try:
                                # years like 1, 2, 5
                                y = float(re.sub(r'[^0-9\.]','', s))
                                if y >= 0.08 and y < 0.2:
                                    m = '1M'
                                elif abs(y-0.25) < 0.05:
                                    m = '3M'
                                elif abs(y-0.5) < 0.1:
                                    m = '6M'
                                else:
                                    yi = int(round(y))
                                    if yi in [1,2,3,5,7,10,15,20,30]:
                                        m = f"{yi}Y"
                            except Exception:
                                pass
                        t_std = t_std or m
                    if t_std in ['1M','3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','15Y','20Y','30Y']:
                        val = self.safe_float(r[ycol])
                        if val is not None:
                            recs.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'country': 'India',
                                'tenor': t_std,
                                'data_type': 'yield',
                                'yield_pct': val,
                                'auction_amount_mn': None,
                                'secondary_volume_mn': None,
                                'outstanding_amount_mn': None,
                                'bid_to_cover': None,
                                'auction_type': None,
                                'source': 'CCIL_ZCYC_LOCAL'
                            })
            return recs
        except Exception as e:
            logger.error(f"Error parsing CCIL ZCYC for {date:%Y-%m-%d}: {e}")
            return recs

    def get_ccil_indicative_yields(self, date: datetime) -> List[Dict]:
        """Optional corroboration: CCIL tenor-wise indicative yields.

        Uses local CSV/XLSX drop-ins under data/ccil_indicative_yields if available.
        This is for methodology alignment; treat as secondary source.
        """
        recs: List[Dict] = []
        f = self._find_local_file(self.local_dirs['ccil_indicative'], date)
        if f is None:
            return recs
        try:
            df = self._read_tabular_file(f)
            if df is None or df.empty:
                return recs
            # Expect columns like Tenor, Yield or buckets with yields
            tenor_col = self._find_col(df.columns, ['tenor','bucket'])
            ycol = self._find_col(df.columns, ['yield','avg_yield','indicative'])
            if tenor_col and ycol:
                for _, row in df.iterrows():
                    std = self._map_tenor(row[tenor_col])
                    if std:
                        val = self.safe_float(row[ycol])
                        if val is not None:
                            recs.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'country': 'India',
                                'tenor': std,
                                'data_type': 'yield',
                                'yield_pct': val,
                                'auction_amount_mn': None,
                                'secondary_volume_mn': None,
                                'outstanding_amount_mn': None,
                                'bid_to_cover': None,
                                'auction_type': None,
                                'source': 'CCIL_INDICATIVE'
                            })
        except Exception as e:
            logger.error(f"Error parsing CCIL indicative yields for {date:%Y-%m-%d}: {e}")
        return recs
    
    def get_rbi_yields(self, date: datetime) -> List[Dict]:
        """Get RBI reference yields for date"""
        records = []
        
        try:
            # RBI Daily Reference Rate API (template)
            api_url = f"{self.base_urls['rbi']}/Scripts/ReferenceRateDisplay.aspx"
            params = {
                'Date': date.strftime('%d/%m/%Y'),
                'format': 'json'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                # Parse response (adjust based on actual RBI API structure)
                data = response.json() if 'json' in response.headers.get('content-type', '') else {}
                
                # Common Indian tenors
                tenor_mapping = {
                    '91D': '3M', '182D': '6M', '364D': '1Y',
                    '2Y': '2Y', '5Y': '5Y', '10Y': '10Y', '30Y': '30Y'
                }
                
                for tenor_key, standard_tenor in tenor_mapping.items():
                    if tenor_key in data:
                        records.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'country': 'India',
                            'tenor': standard_tenor,
                            'data_type': 'yield',
                            'yield_pct': float(data[tenor_key]),
                            'auction_amount_mn': None,
                            'secondary_volume_mn': None,
                            'outstanding_amount_mn': None,
                            'bid_to_cover': None,
                            'auction_type': None,
                            'source': 'RBI'
                        })
            
        except Exception as e:
            logger.error(f"Error fetching RBI yields for {date}: {e}")
        
        return records
    
    def get_ccil_volumes(self, date: datetime) -> List[Dict]:
        """Get CCIL secondary market volumes (tenor-wise). Returns data_type='volume'.

        Priority:
        1) Local drop-in under data/ccil_trading_summary (CSV/XLSX) with date in filename.
        2) Attempt HTTP endpoint (if available).
        """
        records: List[Dict] = []

        # 1) Local file fallback (recommended for reliability)
        f = self._find_local_file(self.local_dirs['ccil_trading'], date)
        if f is not None:
            try:
                df = self._read_tabular_file(f)
                if df is not None and not df.empty:
                    # Find tenor and volume columns
                    tenor_col = self._find_col(df.columns, ['tenor','bucket','maturity'])
                    vol_col = self._find_col(df.columns, ['volume','turnover','amount'])
                    ycol = self._find_col(df.columns, ['yield','avg_yield'])
                    if tenor_col and vol_col:
                        for _, row in df.iterrows():
                            std = self.map_ccil_tenor(row[tenor_col]) or self._map_tenor(row[tenor_col])
                            if std:
                                records.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'country': 'India',
                                    'tenor': std,
                                    'data_type': 'volume',
                                    'yield_pct': self.safe_float(row[ycol]) if ycol else None,
                                    'auction_amount_mn': None,
                                    'secondary_volume_mn': (self.safe_float(row[vol_col]) or 0.0) * 10.0,  # crore -> mn
                                    'outstanding_amount_mn': None,
                                    'bid_to_cover': None,
                                    'auction_type': None,
                                    'source': 'CCIL_LOCAL'
                                })
                        return records
            except Exception as e:
                logger.error(f"Error parsing local CCIL trading summary for {date:%Y-%m-%d}: {e}")

        # 2) HTTP attempt (best-effort; sites may change)
        # Skip if HTTP disabled or date before known online availability
        if not self.india_http_enabled or date.date() < self.ccil_online_start:
            return records
        # One-time lightweight connectivity check to avoid repeated DNS errors
        if not hasattr(self, '_ccil_online_ok'):
            try:
                ping = self.logger.session.get(self.base_urls['ccil'], timeout=5)
                self._ccil_online_ok = (ping.status_code < 500)
            except Exception:
                self._ccil_online_ok = False
        if not self._ccil_online_ok:
            return records
        try:
            api_url = f"{self.base_urls['ccil']}/api/statistics/tradingsummary"
            params = {
                'date': date.strftime('%Y-%m-%d'),
                'market': 'government_securities'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'tenorWise' in data:
                    for tenor, volume_data in data['tenorWise'].items():
                        # Map CCIL tenors to standard
                        standard_tenor = self.map_ccil_tenor(tenor)
                        
                        if standard_tenor:
                            records.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'country': 'India',
                                'tenor': standard_tenor,
                                'data_type': 'volume',
                                'yield_pct': volume_data.get('avgYield'),
                                'auction_amount_mn': None,
                                'secondary_volume_mn': float(volume_data.get('volume', 0)) * 10,  # Convert crore to million
                                'outstanding_amount_mn': None,
                                'bid_to_cover': None,
                                'auction_type': None,
                                'source': 'CCIL'
                            })
        
        except Exception as e:
            logger.error(f"Error fetching CCIL volumes for {date}: {e}")
        
        return records
    
    def get_rbi_auctions(self, date: datetime) -> List[Dict]:
        """Get RBI auction results"""
        records = []
        
        try:
            # 1) Local file fallback (CSV/XLSX/HTML) under data/rbi_auctions
            f = self._find_local_file(self.local_dirs['rbi_auctions'], date)
            if f is not None:
                parsed = self._parse_rbi_auction_local(f, date)
                if parsed:
                    return parsed

            # 2) HTTP best-effort (only on common auction weekdays)
            # Check if it's an auction day (typically Wed/Thu)
            if date.weekday() not in [2, 3]:  # Wed=2, Thu=3
                return records
            
            api_url = f"{self.base_urls['rbi']}/Scripts/AuctionResultDisplay.aspx"
            params = {
                'Date': date.strftime('%d/%m/%Y'),
                'SecurityType': 'all'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                # Parse auction results (structure depends on RBI format)
                # This is a template - adjust based on actual response
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for auction tables
                tables = soup.find_all('table')
                
                for table in tables:
                    rows = table.find_all('tr')[1:]  # Skip header
                    
                    for row in rows:
                        cols = row.find_all(['td', 'th'])
                        
                        if len(cols) >= 6:
                            tenor = self.extract_tenor_from_security_name(cols[0].text.strip())
                            
                            if tenor:
                                records.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'country': 'India',
                                    'tenor': tenor,
                                    'data_type': 'auction',
                                    'yield_pct': self.safe_float(cols[4].text),
                                    'auction_amount_mn': self.safe_float(cols[2].text) * 10,  # Crore to million
                                    'secondary_volume_mn': None,
                                    'outstanding_amount_mn': None,
                                    'bid_to_cover': self.safe_float(cols[3].text),
                                    'auction_type': 'primary',
                                    'source': 'RBI'
                                })
        
        except Exception as e:
            logger.error(f"Error fetching RBI auctions for {date}: {e}")
        
        return records

    def _parse_rbi_auction_local(self, path: Path, date: datetime) -> List[Dict]:
        """Parse local RBI auction files (CSV/XLSX/HTML). Returns list of auction records.

        Expected columns (any reasonable subset):
        - Security / Security Name / Description
        - Accepted / Total Accepted / Accepted Amount (optionally with 'crore' in header)
        - Bid to Cover / BC Ratio
        - Cut-off Yield / High Yield / Yield
        - Auction Type (optional)
        """
        recs: List[Dict] = []
        try:
            ext = path.suffix.lower()
            if ext in ['.csv', '.txt', '.xlsx', '.xls']:
                df = self._read_tabular_file(path)
                if df is None or df.empty:
                    return recs
                # Column detection
                sec_col = self._find_col(df.columns, ['security','security_name','description','security_desc','name','security issued'])
                amt_col = self._find_col(df.columns, ['accepted','accepted_amount','total accepted','total_accepted','amount accepted','notified amount accepted','sale amount'])
                bcr_col = self._find_col(df.columns, ['bid_to_cover','bid to cover','bc_ratio','bcr','bid cover'])
                yld_col = self._find_col(df.columns, ['cut-off','cut_off','cut off','cutoff','cut-off yield','cut off yield','high_yield','high yield','yield'])
                typ_col = self._find_col(df.columns, ['auction_type','type'])

                if not sec_col or not amt_col:
                    return recs

                # Determine unit multiplier for amount (crore -> mn)
                amt_multiplier = 1.0
                if any(k in self._norm(amt_col) for k in ['crore','cr']):
                    amt_multiplier = 10.0
                else:
                    # Also check separate unit column if present
                    unit_col = self._find_col(df.columns, ['unit','units'])
                    if unit_col is not None:
                        # If most entries mention crore, use 10
                        units = (df[unit_col].astype(str).str.lower()).fillna('')
                        if (units.str.contains('crore|cr')).mean() > 0.5:
                            amt_multiplier = 10.0

                for _, row in df.iterrows():
                    sec = str(row[sec_col]) if sec_col in row else ''
                    tenor = self.extract_tenor_from_security_name(sec)
                    if not tenor:
                        continue
                    amount = self.safe_float(row[amt_col])
                    if amount is None:
                        continue
                    bidcov = self.safe_float(row[bcr_col]) if bcr_col and bcr_col in row else None
                    yval = self.safe_float(row[yld_col]) if yld_col and yld_col in row else None
                    atyp = str(row[typ_col]).strip() if typ_col and typ_col in row else 'primary'

                    recs.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'country': 'India',
                        'tenor': tenor,
                        'data_type': 'auction',
                        'yield_pct': yval,
                        'auction_amount_mn': (amount or 0.0) * amt_multiplier,
                        'secondary_volume_mn': None,
                        'outstanding_amount_mn': None,
                        'bid_to_cover': bidcov,
                        'auction_type': atyp,
                        'source': 'RBI_LOCAL'
                    })
                return recs

            if ext in ['.html', '.htm']:
                try:
                    from bs4 import BeautifulSoup
                except Exception:
                    return recs
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    html = fh.read()
                soup = BeautifulSoup(html, 'html.parser')
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    if not rows or len(rows) < 2:
                        continue
                    headers = [c.get_text(strip=True) for c in rows[0].find_all(['th','td'])]
                    def idx(keys):
                        def norm(x):
                            return re.sub(r'[^a-z0-9]','', x.lower())
                        for i,h in enumerate(headers):
                            nh = norm(h)
                            for k in keys:
                                if norm(k) in nh:
                                    return i
                        return -1
                    i_sec = idx(['security','name','description'])
                    i_amt = idx(['accepted','total accepted','amount accepted'])
                    i_bcr = idx(['bid to cover','bid_to_cover','bc ratio','bcr'])
                    i_yld = idx(['cut-off','cut off','high yield','yield'])

                    # multiplier based on header mentioning crore
                    amt_multiplier = 10.0 if i_amt >=0 and re.search(r'crore|cr', headers[i_amt], re.I) else 1.0

                    for r in rows[1:]:
                        cols = [c.get_text(strip=True) for c in r.find_all(['td','th'])]
                        if i_sec < 0 or i_amt < 0 or len(cols) <= max(i_sec, i_amt):
                            continue
                        sec = cols[i_sec]
                        tenor = self.extract_tenor_from_security_name(sec)
                        if not tenor:
                            continue
                        amount = self.safe_float(cols[i_amt])
                        if amount is None:
                            continue
                        bidcov = self.safe_float(cols[i_bcr]) if i_bcr >=0 and i_bcr < len(cols) else None
                        yval = self.safe_float(cols[i_yld]) if i_yld >=0 and i_yld < len(cols) else None
                        recs.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'country': 'India',
                            'tenor': tenor,
                            'data_type': 'auction',
                            'yield_pct': yval,
                            'auction_amount_mn': (amount or 0.0) * amt_multiplier,
                            'secondary_volume_mn': None,
                            'outstanding_amount_mn': None,
                            'bid_to_cover': bidcov,
                            'auction_type': 'primary',
                            'source': 'RBI_LOCAL_HTML'
                        })
                return recs
        except Exception as e:
            logger.error(f"Error parsing RBI local auction file {path}: {e}")
        return recs
    
    def get_outstanding_amounts(self, date: datetime) -> List[Dict]:
        """Get outstanding debt amounts (monthly/periodic data)"""
        records = []
        
        # Fetch less frequently to avoid hammering site (approx month-begin)
        if date.day > 5:
            return records
        
        # Skip online CCIL if disabled or before availability or connectivity down
        if not self.india_http_enabled or date.date() < self.ccil_online_start:
            return records
        if not hasattr(self, '_ccil_online_ok'):
            try:
                ping = self.logger.session.get(self.base_urls['ccil'], timeout=5)
                self._ccil_online_ok = (ping.status_code < 500)
            except Exception:
                self._ccil_online_ok = False
        if not self._ccil_online_ok:
            return records
        try:
            api_url = f"{self.base_urls['ccil']}/api/statistics/outstanding"
            params = {
                'date': date.strftime('%Y-%m-%d'),
                'type': 'government_securities'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'tenorWise' in data:
                    for tenor, outstanding_data in data['tenorWise'].items():
                        standard_tenor = self.map_ccil_tenor(tenor)
                        
                        if standard_tenor:
                            records.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'country': 'India',
                                'tenor': standard_tenor,
                                'data_type': 'outstanding',
                                'yield_pct': None,
                                'auction_amount_mn': None,
                                'secondary_volume_mn': None,
                                'outstanding_amount_mn': float(outstanding_data.get('amount', 0)) * 10,
                                'bid_to_cover': None,
                                'auction_type': None,
                                'source': 'CCIL'
                            })
        
        except Exception as e:
            logger.error(f"Error fetching outstanding amounts for {date}: {e}")
        
        return records
    
    def map_ccil_tenor(self, ccil_tenor: str) -> Optional[str]:
        """Map CCIL tenor names to standard buckets"""
        mapping = {
            '91D': '3M', '3M': '3M', '0.25Y': '3M', '0_25Y': '3M',
            '182D': '6M', '6M': '6M', '0.5Y': '6M', '0_5Y': '6M',
            '364D': '1Y', '1Y': '1Y', '12M': '1Y',
            '2Y': '2Y', '3Y': '3Y', '5Y': '5Y',
            '7Y': '7Y', '10Y': '10Y', '15Y': '15Y',
            '20Y': '20Y', '30Y': '30Y'
        }
        clean = self._norm(ccil_tenor).upper()
        # normalize forms like '91 DAY', '91-D', '91 DAYS'
        if re.match(r'^91\D*D(AY|AYS)?$', clean):
            return '3M'
        if re.match(r'^182\D*D(AY|AYS)?$', clean):
            return '6M'
        if re.match(r'^364\D*D(AY|AYS)?$', clean):
            return '1Y'
        return mapping.get(clean)
    
    def extract_tenor_from_security_name(self, security_name: str) -> Optional[str]:
        """Extract tenor from security name"""
        # Common patterns in Indian security names
        patterns = [
            (r'91.*?day', '3M'),
            (r'182.*?day', '6M'), 
            (r'364.*?day', '1Y'),
            (r'(\d+).*?year', lambda m: f"{m.group(1)}Y")
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, security_name, re.IGNORECASE)
            if match:
                if callable(replacement):
                    return replacement(match)
                return replacement
        
        return None
    
    def safe_float(self, text: str) -> Optional[float]:
        """Safely convert text to float"""
        try:
            clean_text = str(text).replace(',', '').replace('%', '').strip()
            return float(clean_text) if clean_text else None
        except:
            return None

    # ===== Helpers for local file parsing =====
    def _find_local_file(self, base_dir: Path, date: datetime) -> Optional[Path]:
        """Find a local file matching the date in various naming patterns."""
        if not base_dir.exists():
            return None
        ymd = date.strftime('%Y%m%d')
        y_m_d = date.strftime('%Y-%m-%d')
        dmy = date.strftime('%d%m%Y')
        patterns = [f"**/*{ymd}*.*", f"**/*{y_m_d}*.*", f"**/*{dmy}*.*"]
        for pat in patterns:
            matches = list(base_dir.glob(pat))
            if matches:
                # pick the latest file variant
                return sorted(matches)[-1]
        return None

    def _read_tabular_file(self, path: Path) -> Optional[pd.DataFrame]:
        ext = path.suffix.lower()
        if ext in ['.csv', '.txt']:
            try:
                return pd.read_csv(path)
            except Exception:
                # try semicolon
                try:
                    return pd.read_csv(path, sep=';')
                except Exception:
                    return None
        if ext in ['.xlsx', '.xls']:
            # try all sheets, pick the widest
            try:
                xl = pd.ExcelFile(path)
                best = None
                best_cols = 0
                for sheet in xl.sheet_names:
                    try:
                        df = xl.parse(sheet)
                        if df.shape[1] > best_cols and not df.empty:
                            best = df
                            best_cols = df.shape[1]
                    except Exception:
                        continue
                return best
            except Exception:
                return None
        return None

    def _norm(self, s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    def _find_col(self, cols, keys) -> Optional[str]:
        for c in cols:
            n = self._norm(c)
            for k in keys:
                if self._norm(k) in n:
                    return c
        return None

    def _find_tenor_col(self, cols, tenor: str) -> Optional[str]:
        """Find a column matching a standard tenor label."""
        patterns = {
            '3M': [r'(^|\b)3m(\b|$)', r'0[_\.]?25y', r'91d'],
            '6M': [r'(^|\b)6m(\b|$)', r'0[_\.]?5y', r'182d'],
            '1Y': [r'(^|\b)1y(\b|$)', r'12m', r'364d'],
            '2Y': [r'(^|\b)2y(\b|$)'],
            '3Y': [r'(^|\b)3y(\b|$)'],
            '5Y': [r'(^|\b)5y(\b|$)'],
            '7Y': [r'(^|\b)7y(\b|$)'],
            '10Y': [r'(^|\b)10y(\b|$)'],
            '15Y': [r'(^|\b)15y(\b|$)'],
            '20Y': [r'(^|\b)20y(\b|$)'],
            '30Y': [r'(^|\b)30y(\b|$)']
        }
        pats = patterns.get(tenor, [])
        for c in cols:
            n = self._norm(c)
            for p in pats:
                if re.search(p, n):
                    return c
        return None

    def _map_tenor(self, raw) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip().upper()
        s = s.replace('MONTHS', 'M').replace('MONTH', 'M').replace('YEARS', 'Y').replace('YEAR', 'Y')
        s = re.sub(r'\s+', '', s)
        # Normalize numeric forms
        rep = {
            '91D': '3M', '182D': '6M', '364D': '1Y',
            '12M': '1Y', '1YR': '1Y', '2YR': '2Y', '3YR': '3Y', '5YR': '5Y', '7YR': '7Y',
            '10YR': '10Y', '15YR': '15Y', '20YR': '20Y', '30YR': '30Y'
        }
        if s in rep:
            return rep[s]
        # Direct matches like '3M','6M','1Y','2Y',...
        if re.fullmatch(r'(3|6)M', s) or re.fullmatch(r'(1|2|3|5|7|10|15|20|30)Y', s):
            return s
        # Forms like '0.25Y', '0.5Y'
        if s in ['0.25Y', '0_25Y']:
            return '3M'
        if s in ['0.5Y', '0_5Y']:
            return '6M'
        # Text like '3 MONTH', '6 MONTHS', '1 YEAR'
        if re.match(r'^3M(ONTH)?S?$', s):
            return '3M'
        if re.match(r'^6M(ONTH)?S?$', s):
            return '6M'
        if re.match(r'^1Y(EAR)?S?$', s):
            return '1Y'
        m = re.match(r'^(\d+)Y(EAR)?S?$', s)
        if m:
            return f"{m.group(1)}Y"
        return None

class USADataLogger:
    def __init__(self, main_logger: BondDataLogger):
        self.logger = main_logger
        self.base_urls = {
            'treasury': 'https://api.fiscaldata.treasury.gov/services/api/v1',
            'fred': 'https://api.stlouisfed.org/fred',
            'finra': 'https://api.finra.org'
        }
        # Load FRED API key from config if present
        try:
            import config
            self.fred_api_key = getattr(config, 'FRED_API_KEY', None)
            # Optional flag to enable FINRA API attempts
            self.enable_finra_api = bool(getattr(config, 'ENABLE_FINRA_API', False))
        except Exception:
            self.fred_api_key = None
            self.enable_finra_api = False
        # one-time warn flags
        self._finra_warned = False
    
    def get_treasury_yields(self, date: datetime) -> List[Dict]:
        """Get US Treasury constant maturity yields from FRED"""
        records = []
        
    # FRED series IDs for Treasury yields (H.15 constant maturity)
        tenor_series = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO', 
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        try:
            for tenor, series_id in tenor_series.items():
                api_url = f"{self.base_urls['fred']}/series/observations"
                params = {
                    'series_id': series_id,
                    'observation_start': date.strftime('%Y-%m-%d'),
                    'observation_end': date.strftime('%Y-%m-%d'),
                    'api_key': self.fred_api_key,
                    'file_type': 'json'
                }
                
                if self.fred_api_key:
                    response = self.logger.session.get(api_url, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'observations' in data and data['observations']:
                            obs = data['observations'][0]
                            
                            if obs['value'] != '.':  # FRED uses '.' for missing data
                                records.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'country': 'USA',
                                    'tenor': tenor,
                                    'data_type': 'yield',
                                    'yield_pct': float(obs['value']),
                                    'auction_amount_mn': None,
                                    'secondary_volume_mn': None,
                                    'outstanding_amount_mn': None,
                                    'bid_to_cover': None,
                                    'auction_type': None,
                                    'source': 'FRED'
                                })
        
        except Exception as e:
            logger.error(f"Error fetching US Treasury yields for {date}: {e}")
        
        return records
    
    def get_treasury_auctions(self, date: datetime) -> List[Dict]:
        """Get US Treasury auction results"""
        records = []
        
        try:
            # Treasury FiscalData Auctions dataset
            # https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/auctions
            api_url = f"{self.base_urls['treasury']}/fiscal_service/v2/accounting/od/auctions"
            params = {
                'filter': f'record_date:eq:{date.strftime("%Y-%m-%d")}',
                'page[number]': 1,
                'page[size]': 500,
                'format': 'json',
                'fields': 'record_date,security_type_desc,security_term,auction_type,high_yield,bc_ratio,total_accepted'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    for auction in data['data']:
                        tenor = self.extract_tenor_from_security_term(auction.get('security_term', '') or auction.get('security_type_desc',''))
                        
                        if tenor:
                            # total_accepted is reported in dollars -> convert to millions
                            amt = self.safe_float(auction.get('total_accepted'))
                            amt_mn = (amt / 1_000_000.0) if amt is not None else None
                            records.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'country': 'USA',
                                'tenor': tenor,
                                'data_type': 'auction',
                                'yield_pct': self.safe_float(auction.get('high_yield')),
                                'auction_amount_mn': amt_mn,
                                'secondary_volume_mn': None,
                                'outstanding_amount_mn': None,
                                'bid_to_cover': self.safe_float(auction.get('bc_ratio')),
                                'auction_type': (auction.get('auction_type') or 'primary'),
                                'source': 'Treasury_FiscalData'
                            })
        
        except Exception as e:
            logger.error(f"Error fetching US Treasury auctions for {date}: {e}")
        
        return records
    
    def get_trace_volumes(self, date: datetime) -> List[Dict]:
        """Get FINRA TRACE Treasury volume data (aggregate). Returns data_type='volume'."""
        records = []
        
        try:
            # Only attempt if explicitly enabled (FINRA endpoints can require auth/headers)
            if not self.enable_finra_api:
                if not self._finra_warned:
                    logger.info("Skipping FINRA TRACE volumes (API disabled). Set ENABLE_FINRA_API=True in config.py to enable.")
                    self._finra_warned = True
                return records
            # FINRA TRACE Treasury aggregate statistics
            api_url = f"{self.base_urls['finra']}/trace/aggregate"
            params = {
                'date': date.strftime('%Y-%m-%d'),
                'asset_class': 'treasury'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # TRACE typically provides aggregate volumes, not by tenor
                if 'aggregate_volume' in data:
                    records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'country': 'USA',
                        'tenor': 'ALL',  # Aggregate across all tenors
                        'data_type': 'volume',
                        'yield_pct': None,
                        'auction_amount_mn': None,
                        'secondary_volume_mn': float(data['aggregate_volume']),
                        'outstanding_amount_mn': None,
                        'bid_to_cover': None,
                        'auction_type': None,
                        'source': 'FINRA_TRACE'
                    })
        
        except Exception as e:
            logger.error(f"Error fetching TRACE volumes for {date}: {e}")
        
        return records
    
    def get_outstanding_debt(self, date: datetime) -> List[Dict]:
        """Get US outstanding debt by security type"""
        records = []
        
        # Only fetch monthly (first business day of month)
        if date.day > 5:
            return records
        
        try:
            # Use FiscalData v2 endpoint; get latest on/before date and convert dollars to millions
            api_url = f"{self.base_urls['treasury']}/fiscal_service/v2/accounting/od/debt_to_penny"
            params = {
                'filter': f'record_date:le:{date.strftime("%Y-%m-%d")}',
                'sort': '-record_date',
                'page[size]': 1,
                'fields': 'record_date,tot_pub_debt_out_amt,total_public_debt_outstanding',
                'format': 'json'
            }
            
            response = self.logger.session.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    row = data['data'][0]
                    amt = None
                    for k in ['tot_pub_debt_out_amt', 'total_public_debt_outstanding', 'total_debt', 'debt_outstanding_amt']:
                        v = row.get(k)
                        if v is not None:
                            try:
                                amt = float(str(v).replace(',', ''))
                                break
                            except Exception:
                                continue
                    if amt is not None:
                        records.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'country': 'USA',
                            'tenor': 'ALL',
                            'data_type': 'outstanding',
                            'yield_pct': None,
                            'auction_amount_mn': None,
                            'secondary_volume_mn': None,
                            'outstanding_amount_mn': amt / 1_000_000.0,
                            'bid_to_cover': None,
                            'auction_type': None,
                            'source': 'Treasury_FiscalData'
                        })
        
        except Exception as e:
            logger.error(f"Error fetching US outstanding debt for {date}: {e}")
        
        return records
    
    def extract_tenor_from_security_term(self, term: str) -> Optional[str]:
        """Extract tenor from security term description"""
        import re
        
        patterns = [
            (r'3.*?month', '3M'),
            (r'6.*?month', '6M'),
            (r'1.*?year', '1Y'),
            (r'(\d+).*?year', lambda m: f"{m.group(1)}Y")
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, term, re.IGNORECASE)
            if match:
                if callable(replacement):
                    return replacement(match)
                return replacement
        
        return None
    
    def safe_float(self, text: str) -> Optional[float]:
        """Safely convert text to float"""
        try:
            clean_text = str(text).replace(',', '').replace('$', '').strip()
            return float(clean_text) if clean_text else None
        except:
            return None

def main():
    """Main logging function"""
    logger.info("Starting Bond Market Data Logger")
    
    # Initialize main logger
    main_logger = BondDataLogger("bond_market_data.csv")
    
    # Initialize country-specific loggers
    india_logger = IndiaDataLogger(main_logger)
    usa_logger = USADataLogger(main_logger)
    
    # Date range for historical collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:
            logger.info(f"Processing {current_date.strftime('%Y-%m-%d')}")
            
            all_records = []
            
            # Collect India data
            all_records.extend(india_logger.get_rbi_yields(current_date))
            all_records.extend(india_logger.get_ccil_volumes(current_date))
            all_records.extend(india_logger.get_rbi_auctions(current_date))
            all_records.extend(india_logger.get_outstanding_amounts(current_date))
            
            # Collect USA data
            all_records.extend(usa_logger.get_treasury_yields(current_date))
            all_records.extend(usa_logger.get_treasury_auctions(current_date))
            all_records.extend(usa_logger.get_trace_volumes(current_date))
            all_records.extend(usa_logger.get_outstanding_debt(current_date))
            
            # Log all collected data
            if all_records:
                main_logger.log_data(all_records)
                logger.info(f"Logged {len(all_records)} records for {current_date.strftime('%Y-%m-%d')}")
            
            # Rate limiting
            time.sleep(1)
        
        current_date += timedelta(days=1)
    
    logger.info("Bond Market Data Logger completed")

if __name__ == "__main__":
    main()
