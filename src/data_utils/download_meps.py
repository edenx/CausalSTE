"""
MEPS Full Year Consolidated Data Downloader
Downloads FYC files from AHRQ and processes them into pandas DataFrames

This script tries multiple download methods and file formats to ensure success.
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
import time
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# MEPS Full Year Consolidated file information
# PUF Numbers: HC-216, HC-224, HC-233, HC-243 (official format with hyphen)
# URL paths use lowercase without hyphen: h216, h224, h233, h243
MEPS_FYC_FILES = {
    2019: {
        'puf_number': 'HC-216',
        'file_code': 'h216',
        'panel_info': 'Panels 23-24',
        'base_url': 'https://meps.ahrq.gov/data_files/pufs/h216',
        'formats': {
            'csv': 'h216csv.zip',
            'dta': 'h216dta.zip',
            'xlsx': 'h216xlsx.zip',
            'sas7bdat': 'h216v9.zip',
        },
        'expected_file': 'h216.csv'
    },
    2020: {
        'puf_number': 'HC-224',
        'file_code': 'h224',
        'panel_info': 'Panels 24-25',
        'base_url': 'https://meps.ahrq.gov/data_files/pufs/h224',
        'formats': {
            'csv': 'h224csv.zip',
            'dta': 'h224dta.zip',
            'xlsx': 'h224xlsx.zip',
            'sas7bdat': 'h224v9.zip',
        },
        'expected_file': 'h224.csv'
    },
    2021: {
        'puf_number': 'HC-233',
        'file_code': 'h233',
        'panel_info': 'Panels 25-26',
        'base_url': 'https://meps.ahrq.gov/data_files/pufs/h233',
        'formats': {
            'csv': 'h233csv.zip',
            'dta': 'h233dta.zip',
            'xlsx': 'h233xlsx.zip',
            'sas7bdat': 'h233v9.zip',
        },
        'expected_file': 'h233.csv'
    },
    2022: {
        'puf_number': 'HC-243',
        'file_code': 'h243',
        'panel_info': 'Panels 26-27',
        'base_url': 'https://meps.ahrq.gov/data_files/pufs/h243',
        'formats': {
            'csv': 'h243csv.zip',
            'dta': 'h243dta.zip',
            'xlsx': 'h243xlsx.zip',
            'sas7bdat': 'h243v9.zip',
        },
        'expected_file': 'h243.csv'
    }
}


class MEPSDownloader:
    """Handles downloading MEPS Full Year Consolidated data"""
    
    def __init__(self, output_dir: str = 'data/meps_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_with_requests(self, url: str, output_path: Path, 
                               timeout: int = 60) -> bool:
        """Download using requests library with custom headers"""
        try:
            print(f"    Attempting download with requests library...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, 
                                   stream=True, verify=True)
            response.raise_for_status()
            
            # Download in chunks
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r    Progress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Requests method failed: {str(e)}")
            return False
        except Exception as e:
            print(f"    ✗ Unexpected error: {str(e)}")
            return False
    
    def download_with_urllib(self, url: str, output_path: Path) -> bool:
        """Download using urllib with custom headers"""
        try:
            print(f"    Attempting download with urllib...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            req = Request(url, headers=headers)
            
            with urlopen(req, timeout=60) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            
            return True
            
        except Exception as e:
            print(f"    ✗ Urllib method failed: {str(e)}")
            return False
    
    def download_with_urlretrieve(self, url: str, output_path: Path) -> bool:
        """Download using urlretrieve (simplest method)"""
        try:
            print(f"    Attempting download with urlretrieve...")
            urlretrieve(url, output_path)
            return True
        except Exception as e:
            print(f"    ✗ Urlretrieve method failed: {str(e)}")
            return False
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Try multiple download methods"""
        
        # Check if file already exists
        if output_path.exists():
            print(f"    ✓ File already exists: {output_path.name}")
            return True
        
        print(f"    Downloading from: {url}")
        
        # Try different download methods in order
        methods = [
            self.download_with_requests,
            self.download_with_urllib,
            self.download_with_urlretrieve
        ]
        
        for method in methods:
            try:
                if method(url, output_path):
                    if output_path.exists() and output_path.stat().st_size > 0:
                        print(f"    ✓ Successfully downloaded: {output_path.name}")
                        return True
            except:
                continue
        
        print(f"    ✗ All download methods failed for {url}")
        return False
    
    def extract_zip(self, zip_path: Path, extract_dir: Path) -> List[Path]:
        """Extract zip file and return list of extracted files"""
        try:
            print(f"    Extracting {zip_path.name}...")
            extracted_files = []
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in the archive
                file_list = zip_ref.namelist()
                
                # Extract all files
                zip_ref.extractall(extract_dir)
                
                # Return paths to extracted files
                for filename in file_list:
                    extracted_path = extract_dir / filename
                    if extracted_path.exists():
                        extracted_files.append(extracted_path)
                
                print(f"    ✓ Extracted {len(extracted_files)} file(s)")
                return extracted_files
                
        except Exception as e:
            print(f"    ✗ Extraction failed: {str(e)}")
            return []
    
    def download_year(self, year: int, preferred_format: str = 'csv') -> Optional[Path]:
        """
        Download MEPS FYC data for a specific year
        
        Args:
            year: Year to download (2019-2022)
            preferred_format: Preferred file format ('csv', 'dta', 'xlsx', 'sas7bdat')
        
        Returns:
            Path to the downloaded data file, or None if failed
        """
        
        if year not in MEPS_FYC_FILES:
            print(f"✗ Year {year} not available. Choose from: {list(MEPS_FYC_FILES.keys())}")
            return None
        
        info = MEPS_FYC_FILES[year]
        print(f"\n{'='*70}")
        print(f"Downloading MEPS {year} Full Year Consolidated Data")
        print(f"PUF: {info['puf_number']} ({info['panel_info']})")
        print(f"File Code: {info['file_code']}")
        print(f"{'='*70}")
        
        # Try preferred format first, then fallback to others
        format_order = [preferred_format] + [f for f in info['formats'].keys() if f != preferred_format]
        
        for fmt in format_order:
            if fmt not in info['formats']:
                continue
            
            zip_filename = info['formats'][fmt]
            url = f"{info['base_url']}/{zip_filename}"
            zip_path = self.output_dir / zip_filename
            
            print(f"\n  Trying format: {fmt.upper()}")
            
            # Download the zip file
            if not self.download_file(url, zip_path):
                continue
            
            # Extract the zip file
            extracted_files = self.extract_zip(zip_path, self.output_dir)
            
            if not extracted_files:
                continue
            
            # Find the data file
            for extracted_file in extracted_files:
                ext = extracted_file.suffix.lower()
                if ext in ['.csv', '.dta', '.xlsx', '.sas7bdat']:
                    print(f"  ✓ Successfully prepared: {extracted_file.name}")
                    return extracted_file
        
        print(f"\n✗ Failed to download {year} data in any format")
        return None
    
    def download_all_years(self, years: List[int] = [2019, 2020, 2021, 2022],
                          preferred_format: str = 'csv') -> Dict[int, Path]:
        """Download MEPS FYC data for multiple years"""
        
        results = {}
        
        print("\n" + "="*70)
        print("MEPS FULL YEAR CONSOLIDATED DATA DOWNLOADER")
        print("="*70)
        print(f"Years to download: {years}")
        print(f"Preferred format: {preferred_format.upper()}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("="*70)
        
        for year in years:
            try:
                data_file = self.download_year(year, preferred_format)
                if data_file:
                    results[year] = data_file
                time.sleep(1)  # Be nice to the server
            except KeyboardInterrupt:
                print("\n\nDownload interrupted by user")
                break
            except Exception as e:
                print(f"\n✗ Error downloading {year}: {str(e)}")
                continue
        
        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Successfully downloaded: {len(results)}/{len(years)} years")
        
        for year, path in results.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {year}: {path.name} ({size_mb:.1f} MB)")
        
        if len(results) < len(years):
            failed_years = [y for y in years if y not in results]
            print(f"\nFailed to download: {failed_years}")
        
        print("="*70 + "\n")
        
        return results


def load_meps_to_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Load MEPS data file into pandas DataFrame
    
    Supports: CSV, Stata (.dta), Excel (.xlsx), SAS (.sas7bdat)
    """
    
    ext = file_path.suffix.lower()
    
    try:
        print(f"Loading {file_path.name} into pandas DataFrame...")
        
        if ext == '.csv':
            df = pd.read_csv(file_path, low_memory=False)
            
        elif ext == '.dta':
            df = pd.read_stata(file_path)
            
        elif ext == '.xlsx':
            df = pd.read_excel(file_path)
            
        elif ext == '.sas7bdat':
            try:
                # Try pyreadstat first (best for SAS)
                import pyreadstat
                df, meta = pyreadstat.read_sas7bdat(file_path)
            except ImportError:
                print("  Note: pyreadstat not available, trying pandas...")
                df = pd.read_sas(file_path)
        
        else:
            print(f"✗ Unsupported file format: {ext}")
            return None
        
        print(f"✓ Loaded successfully: {len(df):,} rows × {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"✗ Error loading file: {str(e)}")
        return None


def process_meps_dataframes(dataframes: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Process and combine MEPS DataFrames from multiple years
    
    - Adds YEAR column
    - Creates binary MEDICARE variable
    - Cleans negative values (MEPS missing codes)
    - Combines into single DataFrame
    """
    
    print("\n" + "="*70)
    print("PROCESSING MEPS DATA")
    print("="*70)
    
    processed_dfs = []
    
    for year, df in dataframes.items():
        print(f"\nProcessing {year} data...")
        print(f"  Initial: {len(df):,} observations, {len(df.columns)} variables")
        
        # Add year column
        df['YEAR'] = year
        
        # Create binary Medicare variable
        if 'MCREV' in df.columns:
            df['MEDICARE'] = np.where(df['MCREV'] == 1, 1, 0)
            df.loc[df['MCREV'] < 0, 'MEDICARE'] = np.nan
            print(f"  ✓ Created MEDICARE from MCREV")
        elif 'MCREVER' in df.columns:
            df['MEDICARE'] = np.where(df['MCREVER'] == 1, 1, 0)
            df.loc[df['MCREVER'] < 0, 'MEDICARE'] = np.nan
            print(f"  ✓ Created MEDICARE from MCREVER")
        elif 'AGE' in df.columns or 'AGELAST' in df.columns:
            age_col = 'AGE' if 'AGE' in df.columns else 'AGELAST'
            df['MEDICARE'] = (df[age_col] >= 65).astype(int)
            print(f"  ✓ Created MEDICARE from {age_col} >= 65 (proxy)")
        else:
            print(f"  ⚠ Warning: Could not create MEDICARE variable")
        
        # Clean negative values (MEPS missing codes)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        protected_cols = ['YEAR', 'MEDICARE', 'DUPERSID', 'DUID', 'PID']
        
        n_replaced = 0
        for col in numeric_cols:
            if col not in protected_cols:
                mask = df[col] < 0
                n_replaced += mask.sum()
                df.loc[mask, col] = np.nan
        
        print(f"  ✓ Replaced {n_replaced:,} negative values with NaN")
        
        # Remove observations with missing Medicare status
        if 'MEDICARE' in df.columns:
            before = len(df)
            df = df[df['MEDICARE'].notna()].copy()
            after = len(df)
            if before - after > 0:
                print(f"  ✓ Removed {before-after:,} obs with missing Medicare status")
        
        print(f"  ✓ Final: {len(df):,} observations")
        processed_dfs.append(df)
    
    # Combine all years
    print(f"\nCombining {len(processed_dfs)} years...")
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    
    print(f"✓ Combined dataset: {len(combined_df):,} total observations")
    
    # Print Medicare distribution
    if 'MEDICARE' in combined_df.columns:
        medicare_dist = combined_df['MEDICARE'].value_counts()
        print(f"\nMedicare distribution:")
        print(f"  Control (0): {medicare_dist.get(0, 0):,} ({medicare_dist.get(0, 0)/len(combined_df)*100:.1f}%)")
        print(f"  Treated (1): {medicare_dist.get(1, 0):,} ({medicare_dist.get(1, 0)/len(combined_df)*100:.1f}%)")
    
    print("="*70 + "\n")
    
    return combined_df


def main():
    """Main execution function"""
    
    # Initialize downloader
    downloader = MEPSDownloader(output_dir='data/meps_data')
    
    # Download data (try CSV format first, which is easiest for pandas)
    years = [2019, 2020, 2021, 2022]
    downloaded_files = downloader.download_all_years(years, preferred_format='dta')  # gen_meps reads the Stata files
    
    if not downloaded_files:
        print("\n✗ No files were downloaded successfully.")
        print("\nPossible reasons:")
        print("  1. Network restrictions blocking access to meps.ahrq.gov")
        print("  2. Server temporarily unavailable")
        print("  3. URLs may have changed")
        print("\nAlternative: Manually download files from:")
        print("  https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp")
        return None
    
    # Load into pandas DataFrames
    print("\n" + "="*70)
    print("LOADING DATA INTO PANDAS DATAFRAMES")
    print("="*70)
    
    dataframes = {}
    for year, file_path in downloaded_files.items():
        df = load_meps_to_dataframe(file_path)
        if df is not None:
            dataframes[year] = df
        time.sleep(0.5)
    
    if not dataframes:
        print("\n✗ No data could be loaded into DataFrames")
        return None
    
    # Process and combine
    combined_df = process_meps_dataframes(dataframes)
    
    # Save processed data
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_csv = output_dir / 'meps_fyc_2019_2022.csv'
    output_pkl = output_dir / 'meps_fyc_2019_2022.pkl'
    
    print("Saving processed data...")
    combined_df.to_csv(output_csv, index=False)
    print(f"✓ Saved CSV: {output_csv}")
    
    combined_df.to_pickle(output_pkl)
    print(f"✓ Saved pickle: {output_pkl}")
    
    # Print column names for reference
    print(f"\nAvailable columns ({len(combined_df.columns)}):")
    print("First 30 columns:")
    for i, col in enumerate(combined_df.columns[:30]):
        print(f"  {i+1:2d}. {col}")
    if len(combined_df.columns) > 30:
        print(f"  ... and {len(combined_df.columns) - 30} more")
    
    return combined_df

