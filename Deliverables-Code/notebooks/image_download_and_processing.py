# %% [markdown]
# # Image Download and Processing Pipeline
# 
# This notebook handles:
# 1. Download images from Google Drive URLs
# 2. Auto-orient images to portrait
# 3. Interactive image selection and processing markup
# 4. Automated image processing based on selections
# 5. Final review and confirmation interface
#
# **Environment Setup:**
# - Local: Create `.venv` and run this notebook
# - Runpod: Clone repo and run requirements install cell

# %% [markdown] 
# ## Cell Block 1: Setup & Authentication

# %%
# Cell 1: Imports and configuration
import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

def find_project_root():
    """Find the project root directory (where .gitattributes and .gitignore files are located)."""
    current = Path.cwd()
    
    # Walk up to find .gitattributes or .gitignore files (project root indicators)
    while current.parent != current:
        if (current / ".gitattributes").exists() or (current / ".gitignore").exists() or (current / ".git").exists():
            return current
        current = current.parent
    
    # If nothing found, use current directory
    return Path.cwd()

# Requirements installation function
def upgrade_pip():
    """Upgrade pip to the latest version."""
    print("🔧 Upgrading pip to latest version...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            print("✅ Pip upgraded successfully!")
            return True
        else:
            print("⚠️  Pip upgrade failed, but continuing with installation...")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"⚠️  Pip upgrade failed: {e}")
        return False

def install_requirements():
    """Install requirements from the requirements file."""
    # Find project root first
    project_root = find_project_root()
    print(f"📁 Project root: {project_root}")
    
    # Upgrade pip first
    upgrade_pip()
    
    # Look for requirements file in Deliverables-Code subdirectory
    requirements_path = project_root / "Deliverables-Code" / "requirements" / "requirements_image_download_and_processing.txt"
    
    if requirements_path.exists():
        print(f"📦 Installing requirements from: {requirements_path}")
        print("⏳ Installing packages... (this may take a few minutes)")
        
        try:
            # Read requirements to show what's being installed
            with open(requirements_path, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            print(f"📋 Found {len(requirements)} packages to install:")
            for req in requirements[:5]:  # Show first 5 packages
                package_name = req.split('==')[0].split('>=')[0].split('<=')[0]
                print(f"   • {package_name}")
            if len(requirements) > 5:
                print(f"   • ... and {len(requirements) - 5} more packages")
            
            print("🔧 Running pip install...")
            
            # Run pip install with proper encoding for Windows
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print("✅ Requirements installed successfully!")
                return True
            else:
                print("❌ Pip installation failed. Analyzing error...")
                print(f"🔍 Return code: {result.returncode}")
                
                # Show stderr output for debugging
                if result.stderr:
                    print("📝 Error details:")
                    error_lines = result.stderr.split('\n')
                    for line in error_lines[-10:]:  # Show last 10 lines of error
                        if line.strip():
                            print(f"   {line}")
                
                # Try individual package installation with Windows-specific handling
                print("\n🔄 Attempting individual package installation...")
                failed_packages = []
                successful_packages = []
                
                # Define problematic packages that often need special handling on Windows
                problematic_packages = {'numpy', 'pandas', 'Pillow', 'scikit-image', 'opencv-python'}
                
                for req in requirements:
                    package_name = req.split('==')[0].split('>=')[0].split('<=')[0]
                    print(f"   Installing {package_name}...", end=" ")
                    
                    # For problematic packages, try without version constraints first
                    if package_name in problematic_packages:
                        # Try without version constraints and force binary wheels
                        individual_result = subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            "--only-binary=all", "--no-build-isolation", package_name
                        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
                        
                        if individual_result.returncode != 0:
                            # Fallback: try with just the package name
                            individual_result = subprocess.run([
                                sys.executable, "-m", "pip", "install", package_name
                            ], capture_output=True, text=True, encoding='utf-8', errors='replace')
                    else:
                        # For other packages, try normal installation
                        individual_result = subprocess.run([
                            sys.executable, "-m", "pip", "install", req
                        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
                    
                    if individual_result.returncode == 0:
                        print("✓")
                        successful_packages.append(req)
                    else:
                        print("✗")
                        failed_packages.append(req)
                
                # Report results
                print(f"\n📊 Installation Results:")
                print(f"   ✅ Successful: {len(successful_packages)} packages")
                print(f"   ❌ Failed: {len(failed_packages)} packages")
                
                if failed_packages:
                    print(f"\n🚨 Failed packages:")
                    for pkg in failed_packages:
                        print(f"   • {pkg}")
                    
                    print(f"\n💡 Windows-specific troubleshooting suggestions:")
                    print(f"   1. Install Microsoft Visual C++ Build Tools")
                    print(f"   2. Try installing packages without version constraints:")
                    for pkg in failed_packages:
                        base_name = pkg.split('==')[0].split('>=')[0].split('<=')[0]
                        print(f"      pip install {base_name}")
                    print(f"   3. For numpy/pandas, try: pip install --only-binary=all numpy pandas")
                    print(f"   4. For Pillow, try: pip install --upgrade Pillow")
                    print(f"   5. Consider using conda instead: conda install numpy pandas pillow")
                
                return len(failed_packages) == 0
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running pip: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading requirements file: {e}")
            return False
    else:
        print(f"❌ Requirements file not found at: {requirements_path}")
        print("💡 Please install dependencies manually:")
        print("   pip install google-auth google-auth-oauthlib google-api-python-client")
        print("   pip install Pillow opencv-python pandas numpy ipywidgets tqdm")
        return False

# Install requirements (uncomment to run)
print("🔧 To install requirements, uncomment and run the next line:")
print("install_requirements()")

def install_failed_packages():
    """Install the packages that commonly fail on Windows."""
    print("🔧 Installing commonly problematic packages with Windows-specific approaches...")
    
    # Packages that failed and their Windows-friendly installation commands
    packages_to_try = [
        ("numpy", ["--only-binary=all", "numpy"]),
        ("pandas", ["--only-binary=all", "pandas"]),
        ("Pillow", ["--upgrade", "Pillow"]),
        ("scikit-image", ["--only-binary=all", "scikit-image"])
    ]
    
    successful = []
    failed = []
    
    for package_name, install_args in packages_to_try:
        print(f"   Installing {package_name}...", end=" ")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + install_args, 
            capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print("✓")
                successful.append(package_name)
            else:
                print("✗")
                failed.append(package_name)
        except Exception as e:
            print(f"✗ (Error: {e})")
            failed.append(package_name)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {len(successful)} packages")
    print(f"   ❌ Still failed: {len(failed)} packages")
    
    if failed:
        print(f"\n🚨 Still failing: {', '.join(failed)}")
        print(f"💡 Manual installation commands:")
        for pkg in failed:
            print(f"   pip install {pkg}")
    
    return len(failed) == 0

print("🔧 If some packages failed, try: install_failed_packages()")

# %%
# Cell 1b: Install Requirements (uncomment to run)
# install_requirements()

# %%
# Cell 2: Import all required libraries
print("📦 Loading required libraries...")

# Core data science libraries
try:
    print("   • Loading pandas & numpy...", end=" ")
    import pandas as pd
    import numpy as np
    print("✓")
except ImportError as e:
    print(f"✗ (pandas/numpy: {e})")

# Image processing libraries  
try:
    print("   • Loading PIL & OpenCV...", end=" ")
    from PIL import Image, ImageEnhance
    import cv2
    print("✓")
except ImportError as e:
    print(f"✗ (PIL/OpenCV: {e})")

# Progress and UI libraries
try:
    print("   • Loading tqdm & ipywidgets...", end=" ")
    from tqdm import tqdm
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    print("✓")
except ImportError as e:
    print(f"✗ (tqdm/ipywidgets: {e})")

# Google API libraries
try:
    print("   • Loading Google APIs...", end=" ")
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    print("✓")
except ImportError as e:
    print(f"✗ (Google APIs: {e})")

# Utility libraries
try:
    print("   • Loading utilities...", end=" ")
    import io
    import requests
    import shutil
    import base64
    print("✓")
except ImportError as e:
    print(f"✗ (utilities: {e})")

print("✅ Import process complete!")

# Check if any imports failed
import_errors = []
missing_packages = []

try:
    pandas_test = pd.DataFrame()
    numpy_test = np.array([1,2,3])
except NameError:
    import_errors.append("pandas/numpy")
    missing_packages.extend(["pandas", "numpy"])

try:
    pil_test = Image.new('RGB', (1, 1))
    cv2_test = cv2.__version__
except NameError:
    import_errors.append("PIL/OpenCV")
    missing_packages.extend(["Pillow", "opencv-python"])

try:
    tqdm_test = tqdm
    widgets_test = widgets
except NameError:
    import_errors.append("tqdm/ipywidgets")
    missing_packages.extend(["tqdm", "ipywidgets"])

try:
    google_test = Request
except NameError:
    import_errors.append("Google APIs")
    missing_packages.extend(["google-auth", "google-auth-oauthlib", "google-api-python-client"])

if import_errors:
    print(f"❌ Some imports failed: {', '.join(import_errors)}")
    print("🔧 Please run the requirements installation cell above.")
    print("💡 Or manually install missing packages:")
    for package in set(missing_packages):
        print(f"   pip install {package}")
else:
    print("🎉 All libraries loaded successfully!")

# %%
# Cell 3: Configuration and directory setup
# Configuration
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly',
          'https://www.googleapis.com/auth/drive.readonly']

# Find project root and set up directories
project_root = find_project_root()
print(f"🔍 Project root: {project_root}")
print(f"📂 Current working directory: {Path.cwd()}")

# Set up paths - project files are in Deliverables-Code subdirectory
deliverables_dir = project_root / "Deliverables-Code"
BASE_DIR = deliverables_dir / "data" / "images"
CREDENTIALS_BASE = deliverables_dir

RAW_DOWNLOAD_DIR = BASE_DIR / "0_raw_download" 
CURATED_DIR = BASE_DIR / "1_curated"
METADATA_DIR = BASE_DIR / "metadata"
DISPLAY_CACHE_DIR = BASE_DIR / "display_cache"

# Create directories
for dir_path in [RAW_DOWNLOAD_DIR, CURATED_DIR, METADATA_DIR, DISPLAY_CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Display settings
REVIEW_SIZE = (600, 800)
COMPARISON_SIZE = (500, 667)

print("✅ Configuration complete!")
print(f"📁 Data directory: {BASE_DIR}")
print(f"🔑 Credentials directory: {CREDENTIALS_BASE}")
print(f"🏗️ Deliverables directory: {deliverables_dir}")

# %%
# Cell 4: Google Drive authentication
def authenticate_google_drive():
    """Authenticate with Google Drive API."""
    creds = None
    token_path = CREDENTIALS_BASE / "token.json"
    credentials_path = CREDENTIALS_BASE / "gdrive_oauth.json"
    
    print(f"🔍 Looking for credentials at: {credentials_path}")
    
    # Load existing token
    if token_path.exists():
        print("📋 Loading existing token...")
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    
    # If there are no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"OAuth credentials file not found: {credentials_path}\n"
                    "Please ensure gdrive_oauth.json is in the correct location."
                )
            
            print("🆕 Getting new authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        print("💾 Saving credentials...")
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds), build('sheets', 'v4', credentials=creds)

def load_sheets_config():
    """Load Google Sheets configuration from gdrive_oauth.json."""
    credentials_path = CREDENTIALS_BASE / "gdrive_oauth.json"
    
    try:
        with open(credentials_path, 'r') as f:
            config = json.load(f)
        
        sheets_config = config.get('sheets_config', {})
        if sheets_config:
            print("📋 Loaded sheets configuration from gdrive_oauth.json:")
            print(f"   Spreadsheet ID: {sheets_config.get('spreadsheet_id', 'Not set')}")
            print(f"   Sheet Name: {sheets_config.get('sheet_name', 'Not set')}")
            print(f"   URL Column: {sheets_config.get('url_column', 'Not set')}")
            return sheets_config
        else:
            print("⚠️  No sheets_config found in gdrive_oauth.json")
            return {}
    except Exception as e:
        print(f"❌ Error loading sheets config: {e}")
        return {}

# Authenticate
try:
    drive_service, sheets_service = authenticate_google_drive()
    sheets_config = load_sheets_config()
    print("✅ Google Drive authentication successful!")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    print("💡 Make sure gdrive_oauth.json is in the correct location")
    drive_service, sheets_service = None, None
    sheets_config = {}

# %%
# Cell 5: Load existing metadata or create new CSV files
def load_or_create_metadata():
    """Load existing processing log or create new one."""
    metadata_path = METADATA_DIR / "processing_log.csv"
    
    if metadata_path.exists():
        try:
            # Check if file is empty or corrupted
            if metadata_path.stat().st_size == 0:
                print("⚠️  Metadata file is empty, creating new one...")
                # File exists but is empty, recreate it
                metadata_path.unlink()  # Delete empty file
            else:
                df = pd.read_csv(metadata_path)
                if len(df.columns) == 0 or len(df) == 0:
                    print("⚠️  Metadata file is corrupted, creating new one...")
                    metadata_path.unlink()  # Delete corrupted file
                else:
                    print(f"✅ Loaded existing metadata: {len(df)} records")
                    return df, metadata_path
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"⚠️  Metadata file is corrupted ({e}), creating new one...")
            metadata_path.unlink()  # Delete corrupted file
        except Exception as e:
            print(f"⚠️  Error reading metadata file ({e}), creating new one...")
            metadata_path.unlink()  # Delete problematic file
    
    # Create new metadata file with headers
    print("✅ Creating new metadata file...")
    columns = [
        'filename', 'work_order_number', 'downloaded', 'rotated', 'rotation_angle', 
        'hand_selected', 'brightness_needed', 'brightness_direction', 'brightness_amount',
        'contrast_needed', 'contrast_direction', 'contrast_amount',
        'color_needed', 'color_direction', 'color_amount', 'notes',
        'moved_to_curated', 'processed_version_created', 'final_choice', 'processing_complete'
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(metadata_path, index=False)
    print("✅ New metadata file created successfully")
    
    return df, metadata_path

def load_or_create_ground_truth():
    """Load existing ground truth CSV or create new one."""
    ground_truth_path = METADATA_DIR / "ground_truth.csv"
    
    if ground_truth_path.exists():
        try:
            # Check if file is empty or corrupted
            if ground_truth_path.stat().st_size == 0:
                print("⚠️  Ground truth file is empty, creating new one...")
                # File exists but is empty, recreate it
                ground_truth_path.unlink()  # Delete empty file
            else:
                df = pd.read_csv(ground_truth_path)
                if len(df.columns) == 0 or len(df) == 0:
                    print("⚠️  Ground truth file is corrupted, creating new one...")
                    ground_truth_path.unlink()  # Delete corrupted file
                else:
                    print(f"✅ Loaded existing ground truth: {len(df)} records")
                    return df, ground_truth_path
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"⚠️  Ground truth file is corrupted ({e}), creating new one...")
            ground_truth_path.unlink()  # Delete corrupted file
        except Exception as e:
            print(f"⚠️  Error reading ground truth file ({e}), creating new one...")
            ground_truth_path.unlink()  # Delete problematic file
    
    # Create new ground truth file with headers
    print("✅ Creating new ground truth file...")
    columns = [
        'filename', 'work_order_number', 'total', 'date', 'name', 
        'drive_url', 'download_status', 'file_id'
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(ground_truth_path, index=False)
    print("✅ New ground truth file created successfully")
    
    return df, ground_truth_path

metadata_df, metadata_path = load_or_create_metadata()
ground_truth_df, ground_truth_path = load_or_create_ground_truth()

# %% [markdown]
# ## Cell Block 2: Enhanced Download & Initial Processing

# %%
# Cell 6: Enhanced download with filtering and ground truth creation
def extract_file_id_from_url(url):
    """Extract Google Drive file ID from various URL formats."""
    if not url or 'drive.google.com' not in url:
        return None
    
    # Handle different URL formats
    if '/file/d/' in url:
        return url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    else:
        return None

def validate_required_fields(row_data, row_num):
    """Validate that required fields are populated."""
    # Check Work Order Number (column G, index 3)
    work_order = row_data[3] if len(row_data) > 3 and row_data[3] else None
    if not work_order or str(work_order).strip() == '':
        return False, f"Row {row_num}: Missing Work Order Number"
    
    # Check Total (column I, index 5)  
    total = row_data[5] if len(row_data) > 5 and row_data[5] else None
    if not total or str(total).strip() == '':
        return False, f"Row {row_num}: Missing Total"
    
    # Check URL (column H, index 4)
    url = row_data[4] if len(row_data) > 4 and row_data[4] else None
    if not url or str(url).strip() == '':
        return False, f"Row {row_num}: Missing URL"
    
    return True, None

def sanitize_filename(work_order):
    """Create safe filename from work order number."""
    # Remove any non-alphanumeric characters except periods and hyphens
    import re
    safe_name = re.sub(r'[^\w\-.]', '', str(work_order))
    return f"{safe_name}.jpg"

def download_image_from_drive(file_id, filename, drive_service):
    """Download image from Google Drive."""
    try:
        # Get file metadata first
        file_metadata = drive_service.files().get(fileId=file_id).execute()
        print(f"📁 Downloading: {file_metadata.get('name', filename)} → {filename}")
        
        # Download file content
        request = drive_service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        
        downloader = MediaIoBaseDownload(file_content, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # Save to file
        output_path = RAW_DOWNLOAD_DIR / filename
        with open(output_path, 'wb') as f:
            f.write(file_content.getvalue())
        
        return True, None
    except Exception as e:
        return False, str(e)

def download_estimate_images():
    """Download images filtered for 'Estimate' entries with ground truth creation."""
    global metadata_df, ground_truth_df
    
    if not drive_service or not sheets_service:
        print("❌ Google Drive services not authenticated. Please run authentication cell first.")
        return
    
    # Use configuration from gdrive_oauth.json
    spreadsheet_id = sheets_config.get('spreadsheet_id')
    sheet_name = sheets_config.get('sheet_name')
    
    if not spreadsheet_id or not sheet_name:
        print("❌ Missing configuration in gdrive_oauth.json")
        print("Required: spreadsheet_id, sheet_name")
        return
    
    # Read multiple columns: D, E, F, G, H, I
    # D=Type, E=Date, F=Name, G=Work Order, H=URL, I=Total
    try:
        range_name = f"{sheet_name}!D:I"
        print(f"📖 Reading from: {range_name}")
        print("📋 Column mapping: D=Type, E=Date, F=Name, G=Work Order, H=URL, I=Total")
        
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=range_name).execute()
        
        all_data = result.get('values', [])
        print(f"🔍 Found {len(all_data)} total rows")
        
        # Filter for "Estimate" entries (column D)
        estimate_rows = []
        for i, row in enumerate(all_data[1:], start=2):  # Skip header, start row numbering at 2
            if len(row) > 0 and row[0] and str(row[0]).strip().lower() == 'estimate':
                estimate_rows.append((i, row))
        
        print(f"📊 Found {len(estimate_rows)} 'Estimate' entries")
        
        if len(estimate_rows) == 0:
            print("⚠️  No 'Estimate' entries found. Check column D values.")
            return
        
        downloaded_count = 0
        skipped_count = 0
        validation_failures = 0
        
        for row_num, row_data in tqdm(estimate_rows, desc="Processing estimates"):
            # Validate required fields
            is_valid, validation_error = validate_required_fields(row_data, row_num)
            if not is_valid:
                print(f"⚠️  {validation_error}")
                validation_failures += 1
                continue
            
            # Extract data
            doc_type = row_data[0]  # D: Type
            date = row_data[1] if len(row_data) > 1 else ''  # E: Date
            name = row_data[2] if len(row_data) > 2 else ''  # F: Name
            work_order = row_data[3]  # G: Work Order Number
            url = row_data[4]  # H: URL
            total = row_data[5]  # I: Total
            
            # Generate filename from work order number
            filename = sanitize_filename(work_order)
            file_id = extract_file_id_from_url(url)
            
            if not file_id:
                print(f"⚠️  Row {row_num}: Invalid Google Drive URL")
                skipped_count += 1
                continue
            
            # Check if already downloaded
            existing_ground_truth = ground_truth_df[ground_truth_df['work_order_number'] == str(work_order)]
            if not existing_ground_truth.empty and existing_ground_truth.iloc[0]['download_status'] == 'downloaded':
                print(f"⏭️  Skipping {filename} (already downloaded)")
                continue
            
            # Download image
            success, error = download_image_from_drive(file_id, filename, drive_service)
            
            # Update ground truth CSV
            ground_truth_record = {
                'filename': filename,
                'work_order_number': str(work_order),
                'total': str(total),
                'date': str(date),
                'name': str(name),
                'drive_url': url,
                'download_status': 'downloaded' if success else 'failed',
                'file_id': file_id
            }
            
            if str(work_order) not in ground_truth_df['work_order_number'].values:
                new_ground_truth = pd.DataFrame([ground_truth_record])
                ground_truth_df = pd.concat([ground_truth_df, new_ground_truth], ignore_index=True)
            else:
                # Update existing record
                mask = ground_truth_df['work_order_number'] == str(work_order)
                for key, value in ground_truth_record.items():
                    ground_truth_df.loc[mask, key] = value
            
            # Update processing metadata
            metadata_record = {
                'filename': filename,
                'work_order_number': str(work_order),
                'downloaded': success,
                'rotated': False,
                'rotation_angle': 0,
                'hand_selected': False,
                'brightness_needed': False,
                'brightness_direction': '',
                'brightness_amount': 0,
                'contrast_needed': False,
                'contrast_direction': '',
                'contrast_amount': 0,
                'color_needed': False,
                'color_direction': '',
                'color_amount': 0,
                'notes': error if error else '',
                'moved_to_curated': False,
                'processed_version_created': False,
                'final_choice': '',
                'processing_complete': False
            }
            
            if filename not in metadata_df['filename'].values:
                new_metadata = pd.DataFrame([metadata_record])
                metadata_df = pd.concat([metadata_df, new_metadata], ignore_index=True)
            else:
                # Update existing record
                mask = metadata_df['filename'] == filename
                for key, value in metadata_record.items():
                    metadata_df.loc[mask, key] = value
            
            if success:
                downloaded_count += 1
            else:
                skipped_count += 1
            
            # Save progress periodically
            if (downloaded_count + skipped_count) % 10 == 0:
                ground_truth_df.to_csv(ground_truth_path, index=False)
                metadata_df.to_csv(metadata_path, index=False)
                print(f"💾 Progress saved: {downloaded_count} downloaded, {skipped_count} failed")
        
        # Final save
        ground_truth_df.to_csv(ground_truth_path, index=False)
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"✅ Download complete!")
        print(f"   Downloaded: {downloaded_count} images")
        print(f"   Skipped/Failed: {skipped_count} images")
        print(f"   Validation failures: {validation_failures} rows")
        print(f"   Ground truth records: {len(ground_truth_df)}")
        print(f"📁 Ground truth saved to: {ground_truth_path}")
        
    except Exception as e:
        print(f"❌ Error downloading images: {e}")
        import traceback
        traceback.print_exc()

# Legacy function for compatibility (now points to new function)
def download_images_from_sheet(spreadsheet_id=None, sheet_name=None, url_column=None):
    """Legacy function - now redirects to enhanced estimate download."""
    print("ℹ️  Redirecting to enhanced estimate download function...")
    download_estimate_images()

# To run download, uncomment and execute:
print("📥 To download 'Estimate' images with ground truth, run:")
print("download_estimate_images()")

# %%
# Cell 7: Download execution (uncomment to run)
# download_estimate_images()

# %%
# Cell 8: Auto-orient all images to portrait
def detect_and_fix_orientation(image_path):
    """Detect image orientation and rotate to portrait if needed."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if already portrait (height > width)
            if height >= width:
                return 0, False  # No rotation needed
            
            # Rotate 90 degrees clockwise to make portrait
            rotated_img = img.rotate(-90, expand=True)
            rotated_img.save(image_path, quality=95, optimize=True)
            
            return 90, True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0, False

def orient_all_images():
    """Orient all downloaded images to portrait."""
    global metadata_df
    
    image_files = list(RAW_DOWNLOAD_DIR.glob("*.jpg")) + list(RAW_DOWNLOAD_DIR.glob("*.jpeg")) + list(RAW_DOWNLOAD_DIR.glob("*.png"))
    
    if not image_files:
        print("⚠️  No images found to orient. Please download images first.")
        return
    
    rotated_count = 0
    for image_path in tqdm(image_files, desc="Orienting images"):
        filename = image_path.name
        
        # Check if already processed
        existing_record = metadata_df[metadata_df['filename'] == filename]
        if not existing_record.empty and existing_record.iloc[0]['rotated']:
            continue
        
        # Process orientation
        rotation_angle, was_rotated = detect_and_fix_orientation(image_path)
        
        # Update metadata
        if filename in metadata_df['filename'].values:
            metadata_df.loc[metadata_df['filename'] == filename, 'rotated'] = True
            metadata_df.loc[metadata_df['filename'] == filename, 'rotation_angle'] = rotation_angle
        
        if was_rotated:
            rotated_count += 1
    
    # Save updated metadata
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✅ Oriented {len(image_files)} images ({rotated_count} rotated)")

# Run orientation correction
orient_all_images()

# %%
# Cell 9: Enhanced summary with ground truth information
def display_download_summary():
    """Display summary of download and orientation status."""
    global metadata_df, ground_truth_df
    
    total_estimates = len(ground_truth_df)
    downloaded = len(ground_truth_df[ground_truth_df['download_status'] == 'downloaded'])
    failed = len(ground_truth_df[ground_truth_df['download_status'] == 'failed'])
    
    # Processing metadata
    total_files = len(metadata_df)
    rotated = len(metadata_df[metadata_df['rotated'] == True])
    
    print(f"📊 Estimate Images Download Summary:")
    print(f"   Total 'Estimate' entries processed: {total_estimates}")
    print(f"   Successfully downloaded: {downloaded}")
    print(f"   Failed downloads: {failed}")
    print(f"   Oriented to portrait: {rotated}")
    print(f"   Ready for curation: {downloaded}")
    
    # Ground truth summary
    if total_estimates > 0:
        print(f"\n📋 Ground Truth Dataset:")
        print(f"   Records with work order numbers: {len(ground_truth_df[ground_truth_df['work_order_number'] != ''])}")
        print(f"   Records with totals: {len(ground_truth_df[ground_truth_df['total'] != ''])}")
        print(f"   Records with dates: {len(ground_truth_df[ground_truth_df['date'] != ''])}")
        print(f"   Records with names: {len(ground_truth_df[ground_truth_df['name'] != ''])}")
        
        # Sample data
        if len(ground_truth_df) > 0:
            print(f"\n📄 Sample ground truth record:")
            sample = ground_truth_df.iloc[0]
            print(f"   Filename: {sample['filename']}")
            print(f"   Work Order: {sample['work_order_number']}")
            print(f"   Total: {sample['total']}")
            print(f"   Name: {sample['name']}")
    
    if downloaded == 0:
        print("\n💡 No images downloaded yet. Run the download cell above.")
    elif downloaded < 50:
        print(f"\n📈 Progress: {downloaded}/~150 estimate images downloaded")
        print(f"   Target: ~50 curated images from these downloads")
    else:
        print(f"\n✅ Good progress! {downloaded} images ready for curation!")

display_download_summary()

# %% [markdown]
# ## Ready for Manual Testing
# 
# **Next Steps:**
# 1. ✅ Environment setup complete
# 2. ✅ Directory structure created
# 3. ✅ Authentication configured
# 4. ✅ Basic processing functions ready
# 
# **To Continue:**
# - Uncomment and run the download cell when ready
# - Proceed with image curation (GUI implementation coming next)
# 
# **For Runpod Deployment:**
# - This notebook will work in runpod after cloning from GitHub
# - Run the requirements installation cell first
# - Ensure OAuth credentials are available 