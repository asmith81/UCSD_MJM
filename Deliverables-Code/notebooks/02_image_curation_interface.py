# %% [markdown]
# # Image Curation Interface
# 
# This notebook provides an interactive interface for manually curating images from the downloaded dataset.
# 
# **Features:**
# - Grid-based image browser with metadata overlay
# - One-click selection/deselection with visual feedback
# - Filtering by contractor, value range, and date
# - Progress tracking and statistics
# - Export selected images to curated directory
#
# **Target:** Select 100-150 high-quality images from 549 available images
#
# **Environment Setup:**
# - Local: Create `.venv` and run this notebook
# - Runpod: Clone repo and run requirements install cell

# %% [markdown]
# ## Cell Block 1: Setup & Configuration

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
    from PIL import Image, ImageEnhance, ImageDraw, ImageFont
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

# Google API libraries (optional for curation interface)
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
    print(f"✗ (Google APIs: {e}) - Optional for curation interface")

# Utility libraries
try:
    print("   • Loading utilities...", end=" ")
    import io
    import requests
    import shutil
    import base64
    from datetime import datetime
    import math
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

# Google APIs are optional for curation interface
try:
    google_test = Request
except NameError:
    # Don't add to import_errors since it's optional
    pass

if import_errors:
    print(f"❌ Some imports failed: {', '.join(import_errors)}")
    print("🔧 Please run the requirements installation cell above.")
    print("💡 Or manually install missing packages:")
    for package in set(missing_packages):
        print(f"   pip install {package}")
else:
    print("🎉 All required libraries loaded successfully!")

# %%
# Cell 3: Configuration and constants
PROJECT_ROOT = find_project_root()
IMAGES_DIR = PROJECT_ROOT / "Deliverables-Code" / "data" / "images" / "0_raw_download"
METADATA_DIR = PROJECT_ROOT / "Deliverables-Code" / "data" / "images" / "metadata"
CURATED_DIR = PROJECT_ROOT / "Deliverables-Code" / "data" / "images" / "1_curated"
CACHE_DIR = PROJECT_ROOT / "Deliverables-Code" / "data" / "images" / "display_cache"

# UI Configuration
THUMBNAIL_SIZE = (300, 400)  # (width, height) for thumbnails
IMAGES_PER_PAGE = 12  # 3x4 grid
GRID_COLS = 3
GRID_ROWS = 4

# Create required directories
for directory in [IMAGES_DIR, METADATA_DIR, CURATED_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"📁 Project structure:")
print(f"   • Project root: {PROJECT_ROOT}")
print(f"   • Images: {IMAGES_DIR}")
print(f"   • Metadata: {METADATA_DIR}")
print(f"   • Curated: {CURATED_DIR}")
print(f"   • Cache: {CACHE_DIR}")

# %% [markdown]
# ## Cell Block 2: Metadata Loading & Validation

# %%
# Cell 4: Metadata loading functions
def load_metadata():
    """Load and validate image metadata from CSV files."""
    ground_truth_path = METADATA_DIR / "ground_truth.csv"
    processing_log_path = METADATA_DIR / "processing_log.csv"
    
    if not ground_truth_path.exists():
        print(f"❌ Ground truth file not found: {ground_truth_path}")
        return None
    
    if not processing_log_path.exists():
        print(f"❌ Processing log file not found: {processing_log_path}")
        return None
    
    try:
        # Load ground truth data
        print("📊 Loading metadata...")
        ground_truth_df = pd.read_csv(ground_truth_path)
        processing_log_df = pd.read_csv(processing_log_path)
        
        # Filter for successfully downloaded images using the correct column names
        # Option 1: Use ground_truth.csv download_status column
        successful_downloads = ground_truth_df[
            ground_truth_df['download_status'] == 'downloaded'
        ]['filename'].tolist()
        
        # Alternative: Use processing_log.csv downloaded column (boolean)
        # successful_downloads = processing_log_df[
        #     processing_log_df['downloaded'] == True
        # ]['filename'].tolist()
        
        # Filter ground truth for images that were actually downloaded
        available_images_df = ground_truth_df[
            ground_truth_df['filename'].isin(successful_downloads)
        ].copy()
        
        # Verify image files exist
        existing_files = []
        for filename in available_images_df['filename']:
            image_path = IMAGES_DIR / filename
            if image_path.exists():
                existing_files.append(filename)
        
        # Final filtered dataset
        final_df = available_images_df[
            available_images_df['filename'].isin(existing_files)
        ].copy()
        
        print(f"📈 Metadata summary:")
        print(f"   • Total in ground truth: {len(ground_truth_df)}")
        print(f"   • Successfully downloaded: {len(successful_downloads)}")
        print(f"   • Files found on disk: {len(existing_files)}")
        print(f"   • Final available images: {len(final_df)}")
        
        if len(final_df) == 0:
            print("❌ No images available for curation!")
            return None
        
        # Add helpful derived columns
        final_df['value_numeric'] = pd.to_numeric(
            final_df['total'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        )
        
        # Sort by value (highest first) for better browsing experience
        final_df = final_df.sort_values('value_numeric', ascending=False).reset_index(drop=True)
        
        return final_df
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None

def display_metadata_summary(df):
    """Display summary statistics about the available images."""
    if df is None or len(df) == 0:
        print("❌ No data to summarize")
        return
    
    print(f"\n📊 Available Images Summary:")
    print(f"   • Total images: {len(df)}")
    
    # Contractor distribution
    if 'name' in df.columns:
        contractor_counts = df['name'].value_counts()
        print(f"   • Contractors: {len(contractor_counts)}")
        for contractor, count in contractor_counts.head(5).items():
            print(f"     - {contractor}: {count} images")
        if len(contractor_counts) > 5:
            print(f"     - ... and {len(contractor_counts) - 5} more contractors")
    
    # Value distribution
    if 'value_numeric' in df.columns:
        values = df['value_numeric'].dropna()
        if len(values) > 0:
            print(f"   • Value range: ${values.min():,.0f} - ${values.max():,.0f}")
            print(f"   • Average value: ${values.mean():,.0f}")
            
            # Value brackets
            high_value = len(values[values > 2000])
            mid_value = len(values[(values >= 500) & (values <= 2000)])
            low_value = len(values[values < 500])
            
            print(f"   • Value distribution:")
            print(f"     - High value (>$2,000): {high_value} images")
            print(f"     - Mid value ($500-$2,000): {mid_value} images")
            print(f"     - Low value (<$500): {low_value} images")

# Load the metadata
images_df = load_metadata()
if images_df is not None:
    display_metadata_summary(images_df)
    print("\n✅ Ready to start image curation!")
else:
    print("❌ Cannot proceed without valid metadata. Please run the image download notebook first.")

# %% [markdown]
# ## Cell Block 3: Thumbnail Generation & Caching

# %%
# Cell 5: Thumbnail generator
class ThumbnailGenerator:
    """Generate and cache display-optimized thumbnails."""
    
    def __init__(self, cache_dir, thumbnail_size=THUMBNAIL_SIZE):
        self.cache_dir = Path(cache_dir)
        self.thumbnail_size = thumbnail_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_thumbnail_path(self, filename):
        """Get cached thumbnail path for a filename."""
        name_without_ext = Path(filename).stem
        return self.cache_dir / f"{name_without_ext}_thumb.jpg"
    
    def create_thumbnail(self, image_path, filename):
        """Create a thumbnail for an image."""
        try:
            thumbnail_path = self.get_thumbnail_path(filename)
            
            # Check if thumbnail already exists
            if thumbnail_path.exists():
                return thumbnail_path
            
            # Create thumbnail
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Create a centered thumbnail on white background
                thumb = Image.new('RGB', self.thumbnail_size, 'white')
                
                # Calculate position to center the image
                x = (self.thumbnail_size[0] - img.size[0]) // 2
                y = (self.thumbnail_size[1] - img.size[1]) // 2
                
                thumb.paste(img, (x, y))
                
                # Save thumbnail
                thumb.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                
                return thumbnail_path
                
        except Exception as e:
            print(f"❌ Error creating thumbnail for {filename}: {e}")
            return None
    
    def create_metadata_overlay(self, thumbnail_path, metadata, selected=False):
        """Add metadata overlay to thumbnail."""
        try:
            with Image.open(thumbnail_path) as img:
                # Create a copy for overlay
                overlay_img = img.copy()
                draw = ImageDraw.Draw(overlay_img)
                
                # Try to use a better font, fall back to default
                try:
                    font_large = ImageFont.truetype("arial.ttf", 16)
                    font_small = ImageFont.truetype("arial.ttf", 12)
                except:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()
                
                # Add semi-transparent overlay at bottom
                overlay_height = 80
                overlay = Image.new('RGBA', (self.thumbnail_size[0], overlay_height), (0, 0, 0, 180))
                overlay_img.paste(overlay, (0, self.thumbnail_size[1] - overlay_height), overlay)
                
                # Add metadata text
                y_pos = self.thumbnail_size[1] - overlay_height + 5
                
                # Work order
                work_order = str(metadata.get('work_order_number', 'N/A'))
                draw.text((5, y_pos), f"WO: {work_order}", fill='white', font=font_large)
                
                # Contractor and value
                contractor = str(metadata.get('name', 'Unknown'))[:12]
                total = str(metadata.get('total', 'N/A'))
                draw.text((5, y_pos + 20), f"{contractor}", fill='white', font=font_small)
                draw.text((5, y_pos + 35), f"{total}", fill='yellow', font=font_small)
                
                # Date
                date_str = str(metadata.get('date', ''))[:10]
                draw.text((5, y_pos + 50), f"{date_str}", fill='lightgray', font=font_small)
                
                # Selection indicator
                if selected:
                    # Green border
                    border_width = 6
                    for i in range(border_width):
                        draw.rectangle([i, i, self.thumbnail_size[0]-1-i, self.thumbnail_size[1]-1-i], 
                                     outline='green', width=1)
                    
                    # Checkmark
                    check_size = 30
                    check_x = self.thumbnail_size[0] - check_size - 10
                    check_y = 10
                    
                    # Draw checkmark background
                    draw.ellipse([check_x, check_y, check_x + check_size, check_y + check_size], 
                               fill='green', outline='darkgreen', width=2)
                    
                    # Draw checkmark
                    draw.line([check_x + 8, check_y + 15, check_x + 12, check_y + 19], fill='white', width=3)
                    draw.line([check_x + 12, check_y + 19, check_x + 22, check_y + 9], fill='white', width=3)
                
                return overlay_img
                
        except Exception as e:
            print(f"❌ Error creating overlay for {thumbnail_path}: {e}")
            return None

# Initialize thumbnail generator
thumbnail_gen = ThumbnailGenerator(CACHE_DIR)

# %%
# Cell 6: Generate thumbnails for all images
print("🖼️  Generating thumbnails...")
thumbnail_cache = {}

for idx, (_, row) in enumerate(images_df.iterrows()):
    filename = row['filename']
    image_path = IMAGES_DIR / filename
    
    thumbnail_path = thumbnail_gen.create_thumbnail(image_path, filename)
    if thumbnail_path:
        thumbnail_cache[filename] = thumbnail_path
    
    # Progress indicator
    if (idx + 1) % 50 == 0 or (idx + 1) == len(images_df):
        print(f"   Generated {idx + 1}/{len(images_df)} thumbnails")

print(f"✅ Thumbnail generation complete: {len(thumbnail_cache)} thumbnails ready")

# %% [markdown]
# ## Cell Block 4: Selection Management & State Persistence

# %%
# Cell 7: Selection manager
class SelectionManager:
    """Manage image selection state with persistence."""
    
    def __init__(self, metadata_dir):
        self.metadata_dir = Path(metadata_dir)
        self.selection_file = self.metadata_dir / "curation_selections.json"
        self.selected_images = set()
        self.load_selections()
    
    def load_selections(self):
        """Load previously saved selections."""
        if self.selection_file.exists():
            try:
                with open(self.selection_file, 'r') as f:
                    data = json.load(f)
                    self.selected_images = set(data.get('selected_images', []))
                print(f"📋 Loaded {len(self.selected_images)} previously selected images")
            except Exception as e:
                print(f"⚠️  Error loading selections: {e}")
                self.selected_images = set()
        else:
            print("📋 No previous selections found, starting fresh")
    
    def save_selections(self):
        """Save current selections to file."""
        try:
            data = {
                'selected_images': list(self.selected_images),
                'last_updated': datetime.now().isoformat(),
                'total_selected': len(self.selected_images)
            }
            with open(self.selection_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving selections: {e}")
    
    def toggle_selection(self, filename):
        """Toggle selection state for an image."""
        if filename in self.selected_images:
            self.selected_images.remove(filename)
            return False  # Deselected
        else:
            self.selected_images.add(filename)
            return True   # Selected
    
    def is_selected(self, filename):
        """Check if an image is selected."""
        return filename in self.selected_images
    
    def get_selection_stats(self):
        """Get selection statistics."""
        return {
            'total_selected': len(self.selected_images),
            'selected_images': list(self.selected_images)
        }
    
    def clear_selections(self):
        """Clear all selections."""
        self.selected_images.clear()
        self.save_selections()

# Initialize selection manager
selection_manager = SelectionManager(METADATA_DIR)

# %% [markdown]
# ## Cell Block 5: Grid Display & Navigation

# %%
# Cell 8: Image grid viewer
class ImageGridViewer:
    """Interactive grid-based image viewer with selection."""
    
    def __init__(self, images_df, thumbnail_gen, selection_manager, images_per_page=IMAGES_PER_PAGE):
        self.images_df = images_df.copy()
        self.thumbnail_gen = thumbnail_gen
        self.selection_manager = selection_manager
        self.images_per_page = images_per_page
        self.current_page = 0
        self.total_pages = math.ceil(len(self.images_df) / self.images_per_page)
        self.current_filter = "All"
        self.current_sort = "work_order"
        
        # Create UI components
        self.create_ui()
        
    def create_ui(self):
        """Create the user interface."""
        # Header with statistics
        self.stats_label = widgets.HTML()
        
        # Filter controls
        contractors = ["All"] + sorted(self.images_df['name'].unique().tolist())
        self.contractor_filter = widgets.Dropdown(
            options=contractors,
            value="All",
            description="Contractor:",
            style={'description_width': 'initial'}
        )
        
        value_ranges = ["All", "< $500", "$500 - $2000", "> $2000"]
        self.value_filter = widgets.Dropdown(
            options=value_ranges,
            value="All",
            description="Value Range:",
            style={'description_width': 'initial'}
        )
        
        sort_options = [("Work Order", "work_order_number"), ("Value", "total"), ("Date", "date"), ("Contractor", "name")]
        self.sort_dropdown = widgets.Dropdown(
            options=sort_options,
            value="work_order_number",
            description="Sort by:",
            style={'description_width': 'initial'}
        )
        
        # Navigation controls
        self.prev_button = widgets.Button(description="← Previous", button_style='info')
        self.next_button = widgets.Button(description="Next →", button_style='info')
        self.page_label = widgets.HTML()
        
        # Action buttons
        self.clear_button = widgets.Button(description="Clear All Selections", button_style='warning')
        self.export_button = widgets.Button(description="Export Selected Images", button_style='success')
        
        # Grid container
        self.grid_container = widgets.VBox()
        
        # Wire up events
        self.contractor_filter.observe(self.on_filter_change, names='value')
        self.value_filter.observe(self.on_filter_change, names='value')
        self.sort_dropdown.observe(self.on_sort_change, names='value')
        self.prev_button.on_click(self.prev_page)
        self.next_button.on_click(self.next_page)
        self.clear_button.on_click(self.clear_selections)
        self.export_button.on_click(self.export_selections)
        
        # Layout
        filter_controls = widgets.HBox([self.contractor_filter, self.value_filter, self.sort_dropdown])
        nav_controls = widgets.HBox([self.prev_button, self.page_label, self.next_button])
        action_controls = widgets.HBox([self.clear_button, self.export_button])
        
        self.ui = widgets.VBox([
            self.stats_label,
            filter_controls,
            nav_controls,
            self.grid_container,
            action_controls
        ])
        
    def apply_filters(self):
        """Apply current filters to the dataset."""
        filtered_df = self.images_df.copy()
        
        # Contractor filter
        if self.contractor_filter.value != "All":
            filtered_df = filtered_df[filtered_df['name'] == self.contractor_filter.value]
        
        # Value filter
        if self.value_filter.value != "All":
            # Convert total to numeric for filtering
            filtered_df['total_numeric'] = pd.to_numeric(
                filtered_df['total'].str.replace('$', '').str.replace(',', ''), 
                errors='coerce'
            )
            
            if self.value_filter.value == "< $500":
                filtered_df = filtered_df[filtered_df['total_numeric'] < 500]
            elif self.value_filter.value == "$500 - $2000":
                filtered_df = filtered_df[
                    (filtered_df['total_numeric'] >= 500) & 
                    (filtered_df['total_numeric'] <= 2000)
                ]
            elif self.value_filter.value == "> $2000":
                filtered_df = filtered_df[filtered_df['total_numeric'] > 2000]
        
        # Sort
        if self.sort_dropdown.value in filtered_df.columns:
            if self.sort_dropdown.value == 'total':
                # Sort by numeric value
                filtered_df['total_numeric'] = pd.to_numeric(
                    filtered_df['total'].str.replace('$', '').str.replace(',', ''), 
                    errors='coerce'
                )
                filtered_df = filtered_df.sort_values('total_numeric', ascending=False)
            else:
                filtered_df = filtered_df.sort_values(self.sort_dropdown.value)
        
        return filtered_df
    
    def on_filter_change(self, change):
        """Handle filter changes."""
        self.current_page = 0
        self.update_display()
    
    def on_sort_change(self, change):
        """Handle sort changes."""
        self.current_page = 0
        self.update_display()
    
    def prev_page(self, button):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()
    
    def next_page(self, button):
        """Go to next page."""
        filtered_df = self.apply_filters()
        total_pages = math.ceil(len(filtered_df) / self.images_per_page)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_display()
    
    def clear_selections(self, button):
        """Clear all selections."""
        self.selection_manager.clear_selections()
        self.update_display()
    
    def export_selections(self, button):
        """Export selected images."""
        exporter = CurationExporter(self.selection_manager, self.images_df)
        exporter.export_selected_images()
        self.update_display()
    
    def create_image_widget(self, row):
        """Create a clickable image widget."""
        filename = row['filename']
        thumbnail_path = self.thumbnail_gen.get_thumbnail_path(filename)
        
        if not thumbnail_path or not thumbnail_path.exists():
            return widgets.HTML("<div style='width:300px;height:400px;border:1px solid gray;'>No thumbnail</div>")
        
        # Create thumbnail with overlay
        is_selected = self.selection_manager.is_selected(filename)
        thumbnail_with_overlay = self.thumbnail_gen.create_metadata_overlay(
            thumbnail_path, row, selected=is_selected
        )
        
        if thumbnail_with_overlay:
            # Convert to base64 for display
            buffer = io.BytesIO()
            thumbnail_with_overlay.save(buffer, format='JPEG')
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Create clickable image
            img_html = f"""
            <div style="margin: 5px; cursor: pointer; text-align: center;" 
                 onclick="toggleSelection('{filename}')">
                <img src="data:image/jpeg;base64,{img_data}" 
                     style="max-width: 300px; max-height: 400px; border: 2px solid #ddd; border-radius: 5px;">
            </div>
            """
            
            return widgets.HTML(img_html)
        
        return widgets.HTML("<div style='width:300px;height:400px;'>Error loading image</div>")
    
    def update_display(self):
        """Update the grid display."""
        # Apply filters
        filtered_df = self.apply_filters()
        
        # Calculate pagination
        total_pages = math.ceil(len(filtered_df) / self.images_per_page) if len(filtered_df) > 0 else 1
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(filtered_df))
        page_images = filtered_df.iloc[start_idx:end_idx]
        
        # Update statistics
        selected_count = len(self.selection_manager.selected_images)
        self.stats_label.value = f"""
        <h3>📊 Curation Progress</h3>
        <p><strong>Selected:</strong> {selected_count} images | 
           <strong>Showing:</strong> {len(page_images)} of {len(filtered_df)} filtered images | 
           <strong>Total Available:</strong> {len(self.images_df)} images</p>
        """
        
        # Update page label
        self.page_label.value = f"<strong>Page {self.current_page + 1} of {total_pages}</strong>"
        
        # Update navigation buttons
        self.prev_button.disabled = (self.current_page == 0)
        self.next_button.disabled = (self.current_page >= total_pages - 1)
        
        # Create grid
        rows = []
        for i in range(0, len(page_images), GRID_COLS):
            row_widgets = []
            for j in range(GRID_COLS):
                if i + j < len(page_images):
                    row_data = page_images.iloc[i + j]
                    img_widget = self.create_image_widget(row_data)
                    row_widgets.append(img_widget)
                else:
                    row_widgets.append(widgets.HTML("<div style='width:300px;'></div>"))
            
            rows.append(widgets.HBox(row_widgets))
        
        self.grid_container.children = rows
        
        # Add JavaScript for click handling
        js_code = f"""
        <script>
        function toggleSelection(filename) {{
            // This would normally communicate back to Python
            // For now, we'll use a simpler approach through button clicks
            console.log('Toggle selection for:', filename);
        }}
        </script>
        """
        
        # Update export button state
        self.export_button.disabled = (selected_count == 0)
    
    def display(self):
        """Display the grid viewer."""
        self.update_display()
        return self.ui

# %% [markdown]
# ## Cell Block 6: Click Handler & Selection Interface

# %%
# Cell 9: Selection interface with click handling
class InteractiveSelector:
    """Handle image selection through interactive buttons."""
    
    def __init__(self, grid_viewer):
        self.grid_viewer = grid_viewer
    
    def create_selection_interface(self, images_on_page):
        """Create selection buttons for current page."""
        selection_buttons = []
        
        for _, row in images_on_page.iterrows():
            filename = row['filename']
            work_order = row['work_order_number']
            is_selected = self.grid_viewer.selection_manager.is_selected(filename)
            
            # Create toggle button
            button_text = f"✓ {work_order}" if is_selected else f"○ {work_order}"
            button_style = 'success' if is_selected else 'info'
            
            button = widgets.Button(
                description=button_text,
                button_style=button_style,
                layout=widgets.Layout(width='300px', margin='2px')
            )
            
            # Create click handler
            def make_handler(fname):
                def handler(b):
                    selected = self.grid_viewer.selection_manager.toggle_selection(fname)
                    self.grid_viewer.selection_manager.save_selections()
                    
                    # Update button appearance
                    if selected:
                        b.description = f"✓ {work_order}"
                        b.button_style = 'success'
                    else:
                        b.description = f"○ {work_order}"
                        b.button_style = 'info'
                    
                    # Update grid display
                    self.grid_viewer.update_display()
                    
                return handler
            
            button.on_click(make_handler(filename))
            selection_buttons.append(button)
        
        # Arrange buttons in grid layout
        button_rows = []
        for i in range(0, len(selection_buttons), GRID_COLS):
            row_buttons = selection_buttons[i:i+GRID_COLS]
            # Pad row if needed
            while len(row_buttons) < GRID_COLS:
                row_buttons.append(widgets.HTML("<div style='width:300px;'></div>"))
            button_rows.append(widgets.HBox(row_buttons))
        
        return widgets.VBox(button_rows)

# %% [markdown]
# ## Cell Block 7: Export & Curation Management

# %%
# Cell 10: Curation exporter
class CurationExporter:
    """Handle exporting selected images to curated directory."""
    
    def __init__(self, selection_manager, images_df):
        self.selection_manager = selection_manager
        self.images_df = images_df
    
    def export_selected_images(self):
        """Export selected images to curated directory."""
        selected_files = self.selection_manager.selected_images
        
        if not selected_files:
            print("⚠️  No images selected for export")
            return False
        
        print(f"📦 Exporting {len(selected_files)} selected images...")
        
        # Ensure curated directory exists
        CURATED_DIR.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        skipped_count = 0
        
        # Create export metadata
        export_metadata = []
        
        for filename in selected_files:
            source_path = IMAGES_DIR / filename
            dest_path = CURATED_DIR / filename
            
            if not source_path.exists():
                print(f"⚠️  Source file not found: {filename}")
                skipped_count += 1
                continue
            
            try:
                # Copy image file
                shutil.copy2(source_path, dest_path)
                
                # Get metadata for this image
                image_metadata = self.images_df[self.images_df['filename'] == filename]
                if not image_metadata.empty:
                    export_metadata.append(image_metadata.iloc[0].to_dict())
                
                exported_count += 1
                
                if exported_count % 10 == 0:
                    print(f"   Exported {exported_count}/{len(selected_files)} images")
                    
            except Exception as e:
                print(f"❌ Error exporting {filename}: {e}")
                skipped_count += 1
        
        # Save export metadata
        if export_metadata:
            export_df = pd.DataFrame(export_metadata)
            export_metadata_path = CURATED_DIR / "curated_metadata.csv"
            export_df.to_csv(export_metadata_path, index=False)
            
            # Create summary report
            self.create_export_report(export_df, exported_count, skipped_count)
        
        print(f"✅ Export complete!")
        print(f"   Successfully exported: {exported_count} images")
        print(f"   Skipped: {skipped_count} images")
        print(f"   📁 Exported to: {CURATED_DIR}")
        
        return exported_count > 0
    
    def create_export_report(self, export_df, exported_count, skipped_count):
        """Create a summary report of the export."""
        report_path = CURATED_DIR / "curation_report.txt"
        
        # Calculate statistics
        contractor_stats = export_df['name'].value_counts()
        
        # Value statistics
        export_df['total_numeric'] = pd.to_numeric(
            export_df['total'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        )
        total_value = export_df['total_numeric'].sum()
        avg_value = export_df['total_numeric'].mean()
        
        # Create report
        report = f"""
# Image Curation Export Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Images Exported: {exported_count}
- Images Skipped: {skipped_count}
- Total Estimated Value: ${total_value:,.2f}
- Average Estimate Value: ${avg_value:,.2f}

## Contractor Distribution
"""
        
        for contractor, count in contractor_stats.items():
            percentage = (count / exported_count) * 100
            report += f"- {contractor}: {count} images ({percentage:.1f}%)\n"
        
        report += f"""
## Value Distribution
- Under $500: {len(export_df[export_df['total_numeric'] < 500])} images
- $500 - $2000: {len(export_df[(export_df['total_numeric'] >= 500) & (export_df['total_numeric'] <= 2000)])} images
- Over $2000: {len(export_df[export_df['total_numeric'] > 2000])} images

## File Locations
- Curated Images: {CURATED_DIR}
- Metadata: {CURATED_DIR / "curated_metadata.csv"}
- Original Images: {IMAGES_DIR}
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Export report saved: {report_path}")

# %% [markdown]
# ## Cell Block 8: Main Interface Launch

# %%
# Cell 11: Initialize and display the curation interface
print("🚀 Initializing Image Curation Interface...")

# Create the main grid viewer
grid_viewer = ImageGridViewer(images_df, thumbnail_gen, selection_manager)

# Create selection interface
selector = InteractiveSelector(grid_viewer)

print("✅ Curation interface ready!")
print(f"📊 Dataset: {len(images_df)} images available for curation")
print(f"🎯 Target: Select 100-150 high-quality images")

# Display interface
display(HTML("<h2>🖼️ Image Curation Interface</h2>"))
display(HTML("<p><strong>Instructions:</strong> Use the grid below to browse images. Click the selection buttons below each image to select/deselect. Use filters to narrow down images by contractor or value range.</p>"))

grid_viewer.display()

# %%
# Cell 12: Quick selection helpers
class QuickSelectionHelper:
    """Helper functions for quick bulk selections."""
    
    def __init__(self, grid_viewer):
        self.grid_viewer = grid_viewer
    
    def create_quick_selection_ui(self):
        """Create quick selection interface."""
        # Quick selection buttons
        select_high_value_btn = widgets.Button(
            description="Select High Value (>$2000)",
            button_style='warning'
        )
        select_by_contractor_btn = widgets.Button(
            description="Select All Current Contractor",
            button_style='info'
        )
        deselect_page_btn = widgets.Button(
            description="Deselect Current Page",
            button_style='warning'
        )
        
        # Wire up events
        select_high_value_btn.on_click(self.select_high_value)
        select_by_contractor_btn.on_click(self.select_current_contractor)
        deselect_page_btn.on_click(self.deselect_current_page)
        
        return widgets.HBox([
            select_high_value_btn,
            select_by_contractor_btn,
            deselect_page_btn
        ])
    
    def select_high_value(self, button):
        """Select all high-value estimates (>$2000)."""
        high_value_images = self.grid_viewer.images_df.copy()
        high_value_images['total_numeric'] = pd.to_numeric(
            high_value_images['total'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        )
        high_value_images = high_value_images[high_value_images['total_numeric'] > 2000]
        
        for _, row in high_value_images.iterrows():
            self.grid_viewer.selection_manager.selected_images.add(row['filename'])
        
        self.grid_viewer.selection_manager.save_selections()
        self.grid_viewer.update_display()
        print(f"✅ Selected {len(high_value_images)} high-value images")
    
    def select_current_contractor(self, button):
        """Select all images from current contractor filter."""
        if self.grid_viewer.contractor_filter.value != "All":
            contractor_images = self.grid_viewer.images_df[
                self.grid_viewer.images_df['name'] == self.grid_viewer.contractor_filter.value
            ]
            
            for _, row in contractor_images.iterrows():
                self.grid_viewer.selection_manager.selected_images.add(row['filename'])
            
            self.grid_viewer.selection_manager.save_selections()
            self.grid_viewer.update_display()
            print(f"✅ Selected all {len(contractor_images)} images from {self.grid_viewer.contractor_filter.value}")
    
    def deselect_current_page(self, button):
        """Deselect all images on current page."""
        filtered_df = self.grid_viewer.apply_filters()
        start_idx = self.grid_viewer.current_page * self.grid_viewer.images_per_page
        end_idx = min(start_idx + self.grid_viewer.images_per_page, len(filtered_df))
        page_images = filtered_df.iloc[start_idx:end_idx]
        
        for _, row in page_images.iterrows():
            self.grid_viewer.selection_manager.selected_images.discard(row['filename'])
        
        self.grid_viewer.selection_manager.save_selections()
        self.grid_viewer.update_display()
        print(f"✅ Deselected {len(page_images)} images from current page")

# Create and display quick selection helpers
quick_helper = QuickSelectionHelper(grid_viewer)
display(HTML("<h3>⚡ Quick Selection Tools</h3>"))
display(quick_helper.create_quick_selection_ui())

# %%
# Cell 13: Final statistics and export summary
def display_final_summary():
    """Display final curation statistics."""
    stats = selection_manager.get_selection_stats()
    selected_count = stats['total_selected']
    
    if selected_count > 0:
        # Get metadata for selected images
        selected_df = images_df[images_df['filename'].isin(stats['selected_images'])]
        
        # Calculate statistics
        contractor_dist = selected_df['name'].value_counts()
        
        selected_df['total_numeric'] = pd.to_numeric(
            selected_df['total'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        )
        total_value = selected_df['total_numeric'].sum()
        avg_value = selected_df['total_numeric'].mean()
        
        print(f"\n📊 Current Selection Summary:")
        print(f"   Total Selected: {selected_count} images")
        print(f"   Target Range: 100-150 images")
        print(f"   Progress: {min(100, (selected_count/125)*100):.1f}% toward target")
        print(f"\n💰 Value Statistics:")
        print(f"   Total Estimated Value: ${total_value:,.2f}")
        print(f"   Average Value: ${avg_value:,.2f}")
        print(f"\n👥 Contractor Distribution:")
        for contractor, count in contractor_dist.head().items():
            print(f"   {contractor}: {count} images")
        
        if selected_count >= 100:
            print(f"\n🎯 Ready for export! You have selected {selected_count} images.")
            print(f"   Use the 'Export Selected Images' button to finalize curation.")
    else:
        print(f"\n📊 No images selected yet.")
        print(f"   Browse through images and select 100-150 high-quality images for curation.")

# Display current summary
display_final_summary()

print(f"\n🎯 Curation Goals:")
print(f"   • Select 100-150 high-quality images")
print(f"   • Ensure balanced contractor representation")
print(f"   • Include mix of estimate values")
print(f"   • Focus on clear, well-lit images")
print(f"\n💡 Tips:")
print(f"   • Use filters to focus on specific contractors or value ranges")
print(f"   • Use quick selection tools for bulk operations")
print(f"   • Selection state is automatically saved between sessions")
print(f"   • Export creates a complete curated dataset with metadata") 