"""
Source code package for Final Analysis Framework.

This package contains modular utilities for the construction invoice processing analysis.
"""

from .data_loader import (
    # Core data loading functions
    load_ground_truth_data,
    discover_results_files,
    discover_analysis_files,
    load_results_file,
    load_analysis_file,
    load_all_results,
    load_all_analysis,
    select_files_interactive,
    create_comprehensive_dataset,
    
    # Utility functions
    find_project_root,
    setup_project_paths,
    get_project_info,
    initialize_data_loader,
    
    # Path constants (available after import)
    ROOT_DIR,
    DELIVERABLES_DIR,
    DATA_DIR,
    RESULTS_DIR,
    ANALYSIS_DIR,
    CONFIG_DIR
)

__version__ = "2.0.0"
__author__ = "Final Analysis Framework"
__description__ = "Data loading utilities for construction invoice processing analysis" 