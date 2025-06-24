"""
Styling Configuration Module

This module provides consistent styling settings, color palettes, and 
configuration for all visualizations in the construction invoice processing study.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from typing import Dict, Any

# Custom color palette for consistent visualization
ANALYSIS_COLORS = {
    'LMM': '#4682B4',        # Blue for LMM models (when Pixtral+Llama combined)
    'OCR': '#D87093',        # Pink/magenta for OCR models  
    'Pixtral': '#20B2AA',    # Teal for Pixtral (distinct from blue)
    'Llama': '#4682B4',      # Steel blue for Llama (matches LMM when combined)
    'DocTR': '#D87093',      # Pink/magenta for DocTR (consistent with OCR)
    'accuracy': '#7C9885',    # Muted sage green for accuracy metrics
    'cer': '#B85C5C',        # Muted dusty red for error metrics
    'work_order': '#D4A574',  # Muted warm orange for work order
    'total_cost': '#8B7CAE',  # Muted lavender purple for total cost
    'baseline': '#6C757D',    # Gray for baseline/reference
    'improvement': '#20C997'   # Teal for improvements
}

# Industry standard settings
INDUSTRY_STANDARDS = {
    'automation_threshold': 85,  # Industry automation accuracy threshold (%)
    'reference_line_color': 'red',
    'reference_line_style': '--',
    'reference_line_width': 2,
    'reference_line_alpha': 0.8
}

# Matplotlib style configuration - white background, no gridlines
MATPLOTLIB_CONFIG = {
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'axes.grid': False,           # No gridlines by default
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',  # White background
    'axes.facecolor': 'white'     # White background
}

# Seaborn style settings
SEABORN_STYLE = "white"           # White background style
SEABORN_PALETTE = "husl"

# Pandas display configuration
PANDAS_CONFIG = {
    'display.max_columns': None,
    'display.width': None,
    'display.max_colwidth': 50
}

# Chart-specific styling - updated for white background, no gridlines
CHART_STYLES = {
    'bar_chart': {
        'alpha': 0.8,
        'edgecolor': 'black',
        'linewidth': 1.2
    },
    'annotation': {
        'fontsize': 12,
        'fontweight': 'bold',
        'ha': 'center',
        'va': 'bottom',
        'xytext': (0, 3),
        'textcoords': 'offset points'
    },
    'title': {
        'fontsize': 14,
        'fontweight': 'bold',
        'pad': 15
    },
    'axis_label': {
        'fontsize': 11,
        'fontweight': 'bold'
    },
    'grid': {
        'alpha': 0,              # No grid by default
        'axis': 'none'           # No grid axis
    }
}


def configure_styling() -> None:
    """
    Configure all styling settings for matplotlib, seaborn, and pandas.
    This should be called once at the beginning of analysis notebooks.
    """
    # Configure matplotlib
    plt.style.use('default')  # Start with clean default style
    plt.rcParams.update(MATPLOTLIB_CONFIG)
    
    # Configure seaborn
    sns.set_style(SEABORN_STYLE)
    sns.set_palette(SEABORN_PALETTE)
    
    # Configure pandas display
    for option, value in PANDAS_CONFIG.items():
        pd.set_option(option, value)
    
    # Configure warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)


def get_model_color(model_name: str) -> str:
    """
    Get the standard color for a model or metric type.
    
    Args:
        model_name: Name of the model or metric type
        
    Returns:
        Hex color code for the model/metric
    """
    model_key = model_name.lower()
    
    # Direct matches with proper key mapping
    direct_mapping = {
        'lmm': 'LMM',
        'pixtral': 'Pixtral', 
        'llama': 'Llama',
        'ocr': 'OCR',
        'doctr': 'DocTR'
    }
    
    if model_key in direct_mapping:
        color_key = direct_mapping[model_key]
        color = ANALYSIS_COLORS.get(color_key, ANALYSIS_COLORS['baseline'])
        return color
    
    # Partial matches
    if 'pixtral' in model_key:
        return ANALYSIS_COLORS['Pixtral']
    elif 'llama' in model_key:
        return ANALYSIS_COLORS['Llama']
    elif 'doctr' in model_key or 'ocr' in model_key:
        return ANALYSIS_COLORS['OCR']
    elif 'lmm' in model_key:
        return ANALYSIS_COLORS['LMM']
    
    # Metric types
    if 'accuracy' in model_key:
        return ANALYSIS_COLORS['accuracy']
    elif 'cer' in model_key or 'error' in model_key:
        return ANALYSIS_COLORS['cer']
    elif 'work' in model_key:
        return ANALYSIS_COLORS['work_order']
    elif 'cost' in model_key or 'total' in model_key:
        return ANALYSIS_COLORS['total_cost']
    
    # Default
    return ANALYSIS_COLORS['baseline']


def get_model_colors(model_names: list) -> list:
    """
    Get a list of colors for multiple models.
    
    Args:
        model_names: List of model names
        
    Returns:
        List of hex color codes
    """
    return [get_model_color(name) for name in model_names]


def apply_chart_styling(ax: plt.Axes, chart_type: str = 'bar_chart') -> plt.Axes:
    """
    Apply standard styling to a matplotlib axes object.
    
    Args:
        ax: Matplotlib axes object
        chart_type: Type of chart (bar_chart, line_chart, etc.)
        
    Returns:
        Styled axes object
    """
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # White background, no gridlines (benchmark lines added separately)
    ax.set_facecolor('white')
    ax.grid(False)  # Explicitly disable all gridlines
    ax.set_axisbelow(True)
    
    return ax


def add_industry_standard_line(ax: plt.Axes, value: float = None, label: str = None) -> None:
    """
    Add industry standard reference line to a chart.
    
    Args:
        ax: Matplotlib axes object
        value: Value for the reference line (default: industry automation threshold)
        label: Label for the reference line
    """
    if value is None:
        value = INDUSTRY_STANDARDS['automation_threshold']
    
    if label is None:
        label = f'Industry Automation Standard ({value}%)'
    
    ax.axhline(y=value, 
               color=INDUSTRY_STANDARDS['reference_line_color'],
               linestyle=INDUSTRY_STANDARDS['reference_line_style'],
               linewidth=INDUSTRY_STANDARDS['reference_line_width'],
               alpha=INDUSTRY_STANDARDS['reference_line_alpha'])
    
    # Add label
    ax.text(0.98, value + 1, label, 
            transform=ax.get_yaxis_transform(), 
            ha='right', va='bottom',
            fontsize=10, 
            color=INDUSTRY_STANDARDS['reference_line_color'], 
            fontweight='bold')


def print_styling_info() -> None:
    """Print information about available styling options."""
    print("✓ All libraries imported successfully")
    print("✓ Plotting parameters configured")
    print("✓ Custom color palette defined")
    print("✓ Analysis environment ready")
    print()
    print(f"📊 Available analysis colors: {list(ANALYSIS_COLORS.keys())}")
    print("🎨 Visualization settings optimized for analysis reports")
    print(f"📏 Industry automation threshold: {INDUSTRY_STANDARDS['automation_threshold']}%")


# Export key constants for easy access
__all__ = [
    'ANALYSIS_COLORS',
    'INDUSTRY_STANDARDS',
    'MATPLOTLIB_CONFIG',
    'SEABORN_STYLE',
    'SEABORN_PALETTE',
    'PANDAS_CONFIG',
    'CHART_STYLES',
    'configure_styling',
    'get_model_color',
    'get_model_colors',
    'apply_chart_styling',
    'add_industry_standard_line',
    'print_styling_info'
] 