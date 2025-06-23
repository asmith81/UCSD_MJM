"""
Data loading functions for the Final Analysis Framework.

This module contains all data loading utilities for processing results from
the construction invoice processing study.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

def find_project_root():
    """
    Find project root by locating directory containing .gitignore and .gitattributes.
    Similar to implementation in 03_pixtral_model.py
    """
    try:
        # When running as a script, start from script location
        start_path = Path(__file__).parent
    except NameError:
        # When running in a notebook, start from current working directory
        start_path = Path.cwd()
    
    # Walk up the directory tree to find git markers
    current_path = start_path
    while current_path != current_path.parent:  # Stop at filesystem root
        if (current_path / ".gitignore").exists() and (current_path / ".gitattributes").exists():
            return current_path
        current_path = current_path.parent
    
    raise RuntimeError("Could not find project root (directory containing .gitignore and .gitattributes)")

def setup_project_paths():
    """Set up all project directory paths and verify they exist."""
    # Find and set root directory
    ROOT_DIR = find_project_root()
    
    # Set up key directories
    DELIVERABLES_DIR = ROOT_DIR / "Deliverables-Code"
    DATA_DIR = DELIVERABLES_DIR / "data"
    RESULTS_DIR = DELIVERABLES_DIR / "results"
    ANALYSIS_DIR = DELIVERABLES_DIR / "analysis"
    CONFIG_DIR = DELIVERABLES_DIR / "config"
    
    # Create analysis directory if it doesn't exist
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    return {
        'ROOT_DIR': ROOT_DIR,
        'DELIVERABLES_DIR': DELIVERABLES_DIR,
        'DATA_DIR': DATA_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'ANALYSIS_DIR': ANALYSIS_DIR,
        'CONFIG_DIR': CONFIG_DIR
    }

# Initialize paths
PROJECT_PATHS = setup_project_paths()
ROOT_DIR = PROJECT_PATHS['ROOT_DIR']
DELIVERABLES_DIR = PROJECT_PATHS['DELIVERABLES_DIR']
DATA_DIR = PROJECT_PATHS['DATA_DIR']
RESULTS_DIR = PROJECT_PATHS['RESULTS_DIR']
ANALYSIS_DIR = PROJECT_PATHS['ANALYSIS_DIR']
CONFIG_DIR = PROJECT_PATHS['CONFIG_DIR']

def load_ground_truth_data(ground_truth_file: str = None) -> pd.DataFrame:
    """Load and validate ground truth CSV data."""
    # Set default ground truth file path using ROOT_DIR
    if ground_truth_file is None:
        ground_truth_file = DATA_DIR / "images" / "metadata" / "ground_truth.csv"
    else:
        ground_truth_file = Path(ground_truth_file)
    
    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
    
    try:
        # Load with explicit string type for filename column to ensure consistent matching
        ground_truth = pd.read_csv(ground_truth_file, dtype={'filename': str})
        
        # Validate required columns
        required_columns = {'filename', 'work_order_number', 'total'}
        missing_columns = required_columns - set(ground_truth.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in ground truth: {missing_columns}")
        
        # Clean and validate data
        ground_truth['filename'] = ground_truth['filename'].str.strip()
        ground_truth['work_order_number'] = ground_truth['work_order_number'].astype(str).str.strip()
        
        print(f"INFO: Loaded ground truth data: {len(ground_truth)} records")
        return ground_truth
        
    except Exception as e:
        print(f"ERROR: Error loading ground truth data: {e}")
        raise

def discover_results_files() -> Dict[str, List[Path]]:
    """Discover all results files organized by model type."""
    print("INFO: Discovering results files")
    
    results_files = {
        'pixtral': [],
        'llama': [],
        'doctr': [],
        'all': []
    }
    
    # Get all results JSON files
    all_files = list(RESULTS_DIR.glob("results-*.json"))
    
    for file in all_files:
        results_files['all'].append(file)
        
        # Categorize by model type based on filename pattern
        if 'pixtral' in file.name:
            results_files['pixtral'].append(file)
        elif 'llama' in file.name:
            results_files['llama'].append(file)
        elif 'doctr' in file.name:
            results_files['doctr'].append(file)
    
    # Sort files by modification time (newest first)
    for model_type in results_files:
        results_files[model_type].sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"INFO: Found {len(results_files['all'])} total results files")
    for model_type, files in results_files.items():
        if model_type != 'all' and files:
            print(f"INFO:   {model_type}: {len(files)} files")
    
    return results_files

def discover_analysis_files() -> Dict[str, List[Path]]:
    """Discover all analysis files organized by model type."""
    print("INFO: Discovering analysis files")
    
    analysis_files = {
        'pixtral': [],
        'llama': [],
        'doctr': [],
        'all': []
    }
    
    # Get all analysis JSON files
    all_files = list(ANALYSIS_DIR.glob("analysis-*.json"))
    
    for file in all_files:
        analysis_files['all'].append(file)
        
        # Categorize by model type based on filename pattern
        if 'pixtral' in file.name:
            analysis_files['pixtral'].append(file)
        elif 'llama' in file.name:
            analysis_files['llama'].append(file)
        elif 'doctr' in file.name:
            analysis_files['doctr'].append(file)
    
    # Sort files by modification time (newest first)
    for model_type in analysis_files:
        analysis_files[model_type].sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"INFO: Found {len(analysis_files['all'])} total analysis files")
    for model_type, files in analysis_files.items():
        if model_type != 'all' and files:
            print(f"INFO:   {model_type}: {len(files)} files")
    
    return analysis_files

def load_results_file(file_path: Path) -> Dict[str, Any]:
    """Load and validate a results JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        required_keys = {'metadata', 'results'}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in results file: {missing_keys}")
        
        # Add file metadata
        data['file_info'] = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size_mb': round(file_path.stat().st_size / (1024*1024), 2),
            'modification_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return data
        
    except Exception as e:
        print(f"ERROR: Error loading results file {file_path}: {e}")
        raise

def load_analysis_file(file_path: Path) -> Dict[str, Any]:
    """Load and validate an analysis JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        required_keys = {'metadata', 'summary', 'extracted_data'}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in analysis file: {missing_keys}")
        
        # Add file metadata
        data['file_info'] = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size_mb': round(file_path.stat().st_size / (1024*1024), 2),
            'modification_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return data
        
    except Exception as e:
        print(f"ERROR: Error loading analysis file {file_path}: {e}")
        raise

def load_all_results(model_types: List[str] = None) -> Dict[str, List[Dict]]:
    """Load all results files for specified model types."""
    print("INFO: Loading all results files")
    
    if model_types is None:
        model_types = ['pixtral', 'llama', 'doctr']
    
    results_files = discover_results_files()
    all_results = {}
    
    for model_type in model_types:
        if model_type in results_files:
            all_results[model_type] = []
            for file_path in results_files[model_type]:
                try:
                    result_data = load_results_file(file_path)
                    all_results[model_type].append(result_data)
                except Exception as e:
                    print(f"WARNING: Skipping corrupted results file {file_path}: {e}")
    
    total_loaded = sum(len(results) for results in all_results.values())
    print(f"INFO: Loaded {total_loaded} results files across {len(all_results)} model types")
    
    return all_results

def load_all_analysis(model_types: List[str] = None) -> Dict[str, List[Dict]]:
    """Load all analysis files for specified model types."""
    print("INFO: Loading all analysis files")
    
    if model_types is None:
        model_types = ['pixtral', 'llama', 'doctr']
    
    analysis_files = discover_analysis_files()
    all_analysis = {}
    
    for model_type in model_types:
        if model_type in analysis_files:
            all_analysis[model_type] = []
            for file_path in analysis_files[model_type]:
                try:
                    analysis_data = load_analysis_file(file_path)
                    all_analysis[model_type].append(analysis_data)
                except Exception as e:
                    print(f"WARNING: Skipping corrupted analysis file {file_path}: {e}")
    
    total_loaded = sum(len(analyses) for analyses in all_analysis.values())
    print(f"INFO: Loaded {total_loaded} analysis files across {len(all_analysis)} model types")
    
    return all_analysis

def select_files_interactive(file_type: str = "results") -> List[Path]:
    """Interactive file selection for analysis."""
    if file_type == "results":
        files_dict = discover_results_files()
        title = "Available Results Files"
    elif file_type == "analysis":
        files_dict = discover_analysis_files()
        title = "Available Analysis Files"
    else:
        raise ValueError("file_type must be 'results' or 'analysis'")
    
    all_files = files_dict['all']
    if not all_files:
        print(f"No {file_type} files found.")
        return []
    
    print(f"\n{title}:")
    print("-" * 50)
    for i, file_path in enumerate(all_files, 1):
        # Extract model info from filename
        model_info = ""
        if 'pixtral' in file_path.name:
            model_info = " [Pixtral]"
        elif 'llama' in file_path.name:
            model_info = " [Llama]"
        elif 'doctr' in file_path.name:
            model_info = " [DocTR]"
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        print(f"{i:2d}. {file_path.name}{model_info}")
        print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{len(all_files) + 1}. Load all files")
    
    while True:
        try:
            choice = input(f"\nSelect files (comma-separated numbers, or {len(all_files) + 1} for all): ")
            
            if choice.strip() == str(len(all_files) + 1):
                return all_files
            
            # Parse comma-separated choices
            choices = [int(x.strip()) for x in choice.split(',')]
            selected_files = []
            
            for choice_num in choices:
                if 1 <= choice_num <= len(all_files):
                    selected_files.append(all_files[choice_num - 1])
                else:
                    print(f"Invalid choice: {choice_num}")
                    continue
            
            if selected_files:
                print(f"\nSelected {len(selected_files)} file(s):")
                for file_path in selected_files:
                    print(f"  - {file_path.name}")
                return selected_files
            else:
                print("No valid files selected.")
                
        except ValueError:
            print("Please enter valid numbers separated by commas.")

def create_comprehensive_dataset() -> Dict[str, Any]:
    """Create a comprehensive dataset combining all available data."""
    print("INFO: Creating comprehensive dataset")
    
    # Load ground truth
    ground_truth = load_ground_truth_data()
    
    # Load all analysis files (which contain the processed results)
    all_analysis = load_all_analysis()
    
    # Create comprehensive dataset structure
    dataset = {
        'ground_truth': ground_truth,
        'model_data': {},
        'metadata': {
            'created_timestamp': datetime.now().isoformat(),
            'total_models': 0,
            'total_experiments': 0,
            'data_sources': {
                'ground_truth_file': str(DATA_DIR / "images" / "metadata" / "ground_truth.csv"),
                'results_directory': str(RESULTS_DIR),
                'analysis_directory': str(ANALYSIS_DIR)
            }
        }
    }
    
    total_experiments = 0
    for model_type, analyses in all_analysis.items():
        if analyses:
            dataset['model_data'][model_type] = analyses
            total_experiments += len(analyses)
            print(f"INFO: Added {len(analyses)} experiments for {model_type}")
    
    dataset['metadata']['total_models'] = len(dataset['model_data'])
    dataset['metadata']['total_experiments'] = total_experiments
    
    print(f"INFO: Comprehensive dataset created with {dataset['metadata']['total_models']} models and {total_experiments} experiments")
    
    return dataset

def get_project_info():
    """Get project path information for reference."""
    return {
        'ROOT_DIR': ROOT_DIR,
        'DELIVERABLES_DIR': DELIVERABLES_DIR,
        'DATA_DIR': DATA_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'ANALYSIS_DIR': ANALYSIS_DIR,
        'CONFIG_DIR': CONFIG_DIR
    }

def initialize_data_loader():
    """Initialize data loader with verification of directories and available data."""
    print("ℹ️  Initializing data loader")
    
    # Verify data directories exist
    required_dirs = [RESULTS_DIR, ANALYSIS_DIR, DATA_DIR / "images" / "metadata"]
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"WARNING: Creating missing directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Discover available data files
    available_results = discover_results_files()
    available_analysis = discover_analysis_files()
    
    # Load ground truth data
    try:
        ground_truth_data = load_ground_truth_data()
        print(f"✅ Ground truth loaded: {len(ground_truth_data)} records")
    except Exception as e:
        print(f"WARNING: Could not load ground truth data: {e}")
        ground_truth_data = None

    # Create comprehensive dataset for analysis
    try:
        comprehensive_dataset = create_comprehensive_dataset()
        print("✅ Comprehensive dataset created successfully")
    except Exception as e:
        print(f"WARNING: Could not create comprehensive dataset: {e}")
        comprehensive_dataset = None

    # Display summary of available data
    print("\n📊 Data Loading Summary:")
    print(f"   • Ground truth records: {len(ground_truth_data) if ground_truth_data is not None else 'Not available'}")
    print(f"   • Results files found: {len(available_results['all'])}")
    print(f"   • Analysis files found: {len(available_analysis['all'])}")

    if available_results['all']:
        print("\n   Results by model type:")
        for model_type, files in available_results.items():
            if model_type != 'all' and files:
                print(f"     - {model_type.title()}: {len(files)} files")

    if available_analysis['all']:
        print("\n   Analysis by model type:")
        for model_type, files in available_analysis.items():
            if model_type != 'all' and files:
                print(f"     - {model_type.title()}: {len(files)} files")

    print("\n✅ Data loader initialized successfully")
    
    return {
        'ground_truth_data': ground_truth_data,
        'comprehensive_dataset': comprehensive_dataset,
        'available_results': available_results,
        'available_analysis': available_analysis
    }

# Suppress warnings by default
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 