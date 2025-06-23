"""
Primary Performance Comparison Visualization Module

This module provides functions to create the primary performance comparison
between Large Multimodal Models (LMM) and Optical Character Recognition (OCR)
based on analysis results from the construction invoice processing study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns

# Import styling configuration
from .styling import (
    ANALYSIS_COLORS, 
    INDUSTRY_STANDARDS, 
    CHART_STYLES,
    get_model_colors,
    apply_chart_styling,
    add_industry_standard_line
)


def load_analysis_data(analysis_dir: Path) -> Dict[str, Any]:
    """
    Load all analysis files and organize by model type.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Dictionary with 'lmm' and 'ocr' keys containing analysis data
    """
    analysis_files = list(analysis_dir.glob("analysis-*.json"))
    
    lmm_data = []
    ocr_data = []
    
    for file in analysis_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Determine model type based on filename
        if 'pixtral' in file.name or 'llama' in file.name:
            lmm_data.append(data)
        elif 'doctr' in file.name:
            ocr_data.append(data)
    
    return {
        'lmm': lmm_data,
        'ocr': ocr_data
    }


def load_analysis_data_by_individual_models(analysis_dir: Path) -> Dict[str, Any]:
    """
    Load all analysis files and organize by individual model type.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Dictionary with 'pixtral', 'llama', and 'ocr' keys containing analysis data
    """
    analysis_files = list(analysis_dir.glob("analysis-*.json"))
    
    pixtral_data = []
    llama_data = []
    ocr_data = []
    
    for file in analysis_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Determine model type based on filename
        if 'pixtral' in file.name:
            pixtral_data.append(data)
        elif 'llama' in file.name:
            llama_data.append(data)
        elif 'doctr' in file.name:
            ocr_data.append(data)
    
    return {
        'pixtral': pixtral_data,
        'llama': llama_data,
        'ocr': ocr_data
    }


def calculate_aggregate_performance(model_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate performance metrics across all trials for a model type.
    
    Args:
        model_data: List of analysis dictionaries for a model type
        
    Returns:
        Dictionary with aggregate performance metrics
    """
    if not model_data:
        return {'work_order_accuracy': 0.0, 'total_cost_accuracy': 0.0, 'overall_accuracy': 0.0}
    
    # Collect all results across trials
    all_work_order_correct = []
    all_total_cost_correct = []
    
    for analysis in model_data:
        if 'extracted_data' in analysis:
            for item in analysis['extracted_data']:
                if 'performance' in item:
                    all_work_order_correct.append(item['performance'].get('work_order_correct', False))
                    all_total_cost_correct.append(item['performance'].get('total_cost_correct', False))
    
    if not all_work_order_correct:
        return {'work_order_accuracy': 0.0, 'total_cost_accuracy': 0.0, 'overall_accuracy': 0.0}
    
    work_order_accuracy = sum(all_work_order_correct) / len(all_work_order_correct)
    total_cost_accuracy = sum(all_total_cost_correct) / len(all_total_cost_correct)
    
    # Calculate overall accuracy as the average of both fields
    overall_accuracy = (work_order_accuracy + total_cost_accuracy) / 2
    
    return {
        'work_order_accuracy': work_order_accuracy,
        'total_cost_accuracy': total_cost_accuracy, 
        'overall_accuracy': overall_accuracy
    }


def calculate_aggregate_cer(model_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate CER (Character Error Rate) metrics across all trials for a model type.
    
    Args:
        model_data: List of analysis dictionaries for a model type
        
    Returns:
        Dictionary with aggregate CER metrics
    """
    if not model_data:
        return {'work_order_cer': 0.0, 'total_cost_cer': 0.0, 'overall_cer': 0.0}
    
    # Collect all CER values across trials
    all_work_order_cer = []
    all_total_cost_cer = []
    
    for analysis in model_data:
        if 'extracted_data' in analysis:
            for item in analysis['extracted_data']:
                if 'performance' in item:
                    # Get work order CER (default to 1.0 if not found, indicating complete error)
                    work_order_cer = item['performance'].get('work_order_cer', 1.0)
                    all_work_order_cer.append(work_order_cer)
                    
                    # For total cost CER, we need to calculate it from the extracted data
                    # If the total cost is correct, CER is 0, otherwise it's 1.0 (simplified)
                    if item['performance'].get('total_cost_correct', False):
                        total_cost_cer = 0.0
                    else:
                        total_cost_cer = 1.0
                    all_total_cost_cer.append(total_cost_cer)
    
    if not all_work_order_cer:
        return {'work_order_cer': 0.0, 'total_cost_cer': 0.0, 'overall_cer': 0.0}
    
    work_order_cer = sum(all_work_order_cer) / len(all_work_order_cer)
    total_cost_cer = sum(all_total_cost_cer) / len(all_total_cost_cer)
    
    # Calculate overall CER as the average of both fields
    overall_cer = (work_order_cer + total_cost_cer) / 2
    
    return {
        'work_order_cer': work_order_cer,
        'total_cost_cer': total_cost_cer, 
        'overall_cer': overall_cer
    }


def create_primary_performance_comparison_chart(analysis_dir: Path) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create the primary performance comparison bar chart.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Tuple of (matplotlib figure, model accuracies dictionary)
    """
    # Load and analyze data
    data = load_analysis_data(analysis_dir)
    
    lmm_performance = calculate_aggregate_performance(data['lmm'])
    ocr_performance = calculate_aggregate_performance(data['ocr'])
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data for the bar chart
    categories = ['Large Multimodal Models\n(LMM)', 'Optical Character Recognition\n(OCR)']
    accuracies = [lmm_performance['overall_accuracy'] * 100, ocr_performance['overall_accuracy'] * 100]
    colors = get_model_colors(['LMM', 'OCR'])
    
    # Create bars
    bars = ax.bar(categories, accuracies, color=colors, **CHART_STYLES['bar_chart'])
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{accuracy:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   **CHART_STYLES['annotation'])
    
    # Add industry standard reference line
    add_industry_standard_line(ax)
    
    # Customize the chart
    ax.set_title('Primary Performance Comparison:\nLarge Multimodal Models vs. Optical Character Recognition', 
                **CHART_STYLES['title'])
    ax.set_ylabel('Overall Accuracy (%)', **CHART_STYLES['axis_label'])
    ax.set_ylim(0, 105)
    
    # Apply standard chart styling
    apply_chart_styling(ax)
    
    # Tight layout
    plt.tight_layout()
    
    # Return both figure and the accuracy data
    model_accuracies = {
        'LMM': lmm_performance,
        'OCR': ocr_performance
    }
    
    return fig, model_accuracies


def create_individual_model_accuracy_chart(analysis_dir: Path, ax: plt.Axes) -> Dict[str, Dict[str, float]]:
    """
    Create accuracy comparison chart with individual models (Pixtral, Llama, OCR).
    
    Args:
        analysis_dir: Path to the analysis directory
        ax: Matplotlib axes to plot on
        
    Returns:
        Dictionary with individual model performance metrics
    """
    # Load and analyze data by individual models
    data = load_analysis_data_by_individual_models(analysis_dir)
    
    pixtral_performance = calculate_aggregate_performance(data['pixtral'])
    llama_performance = calculate_aggregate_performance(data['llama'])
    ocr_performance = calculate_aggregate_performance(data['ocr'])
    
    # Data for the bar chart
    categories = ['Pixtral-12B', 'Llama-3.2-11B', 'DocTR (OCR)']
    accuracies = [
        pixtral_performance['overall_accuracy'] * 100, 
        llama_performance['overall_accuracy'] * 100,
        ocr_performance['overall_accuracy'] * 100
    ]
    colors = get_model_colors(['Pixtral', 'Llama', 'OCR'])
    
    # Create bars
    bars = ax.bar(categories, accuracies, color=colors, **CHART_STYLES['bar_chart'])
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{accuracy:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   **CHART_STYLES['annotation'])
    
    # Add industry standard reference line
    add_industry_standard_line(ax)
    
    # Customize the chart
    ax.set_title('Individual Model Performance:\nPixtral vs. Llama vs. OCR', 
                **CHART_STYLES['title'])
    ax.set_ylabel('Overall Accuracy (%)', **CHART_STYLES['axis_label'])
    ax.set_ylim(0, 105)
    
    # Apply standard chart styling
    apply_chart_styling(ax)
    
    # Rotate x-axis labels for better fit
    ax.tick_params(axis='x', rotation=15)
    
    # Return model performance data
    model_performance = {
        'Pixtral': pixtral_performance,
        'Llama': llama_performance,
        'OCR': ocr_performance
    }
    
    return model_performance


def create_lmm_vs_ocr_cer_chart(analysis_dir: Path, ax: plt.Axes) -> Dict[str, Dict[str, float]]:
    """
    Create CER comparison chart between LMM and OCR.
    
    Args:
        analysis_dir: Path to the analysis directory
        ax: Matplotlib axes to plot on
        
    Returns:
        Dictionary with CER performance metrics
    """
    # Load and analyze data
    data = load_analysis_data(analysis_dir)
    
    lmm_cer = calculate_aggregate_cer(data['lmm'])
    ocr_cer = calculate_aggregate_cer(data['ocr'])
    
    # Data for the bar chart (CER - lower is better)
    categories = ['Large Multimodal Models\n(LMM)', 'Optical Character Recognition\n(OCR)']
    cer_values = [lmm_cer['overall_cer'] * 100, ocr_cer['overall_cer'] * 100]
    colors = get_model_colors(['LMM', 'OCR'])
    
    # Create bars
    bars = ax.bar(categories, cer_values, color=colors, **CHART_STYLES['bar_chart'])
    
    # Add value labels on bars
    for bar, cer in zip(bars, cer_values):
        height = bar.get_height()
        ax.annotate(f'{cer:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   **CHART_STYLES['annotation'])
    
    # Customize the chart
    ax.set_title('Character Error Rate Comparison:\nLMM vs. OCR (Lower is Better)', 
                **CHART_STYLES['title'])
    ax.set_ylabel('Character Error Rate (%)', **CHART_STYLES['axis_label'])
    ax.set_ylim(0, max(cer_values) * 1.2)
    
    # Apply standard chart styling
    apply_chart_styling(ax)
    
    # Return CER data
    model_cer = {
        'LMM': lmm_cer,
        'OCR': ocr_cer
    }
    
    return model_cer


def create_individual_model_cer_chart(analysis_dir: Path, ax: plt.Axes) -> Dict[str, Dict[str, float]]:
    """
    Create CER comparison chart with individual models (Pixtral, Llama, OCR).
    
    Args:
        analysis_dir: Path to the analysis directory
        ax: Matplotlib axes to plot on
        
    Returns:
        Dictionary with individual model CER metrics
    """
    # Load and analyze data by individual models
    data = load_analysis_data_by_individual_models(analysis_dir)
    
    pixtral_cer = calculate_aggregate_cer(data['pixtral'])
    llama_cer = calculate_aggregate_cer(data['llama'])
    ocr_cer = calculate_aggregate_cer(data['ocr'])
    
    # Data for the bar chart (CER - lower is better)
    categories = ['Pixtral-12B', 'Llama-3.2-11B', 'DocTR (OCR)']
    cer_values = [
        pixtral_cer['overall_cer'] * 100,
        llama_cer['overall_cer'] * 100,
        ocr_cer['overall_cer'] * 100
    ]
    colors = get_model_colors(['Pixtral', 'Llama', 'OCR'])
    
    # Create bars
    bars = ax.bar(categories, cer_values, color=colors, **CHART_STYLES['bar_chart'])
    
    # Add value labels on bars
    for bar, cer in zip(bars, cer_values):
        height = bar.get_height()
        ax.annotate(f'{cer:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   **CHART_STYLES['annotation'])
    
    # Customize the chart
    ax.set_title('Individual Model CER:\nPixtral vs. Llama vs. OCR (Lower is Better)', 
                **CHART_STYLES['title'])
    ax.set_ylabel('Character Error Rate (%)', **CHART_STYLES['axis_label'])
    ax.set_ylim(0, max(cer_values) * 1.2)
    
    # Apply standard chart styling
    apply_chart_styling(ax)
    
    # Rotate x-axis labels for better fit
    ax.tick_params(axis='x', rotation=15)
    
    # Return model CER data
    model_cer = {
        'Pixtral': pixtral_cer,
        'Llama': llama_cer,
        'OCR': ocr_cer
    }
    
    return model_cer


def create_performance_comparison_grid(analysis_dir: Path) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create a 2x2 grid of performance comparison charts.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Tuple of (matplotlib figure, all performance metrics dictionary)
    """
    # Create 2x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Upper left: LMM vs OCR accuracy (recreate original chart)
    data = load_analysis_data(analysis_dir)
    lmm_performance = calculate_aggregate_performance(data['lmm'])
    ocr_performance = calculate_aggregate_performance(data['ocr'])
    
    categories = ['LMM', 'OCR']
    accuracies = [lmm_performance['overall_accuracy'] * 100, ocr_performance['overall_accuracy'] * 100]
    colors = get_model_colors(['LMM', 'OCR'])
    
    bars = ax1.bar(categories, accuracies, color=colors, **CHART_STYLES['bar_chart'])
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax1.annotate(f'{accuracy:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    **CHART_STYLES['annotation'])
    
    add_industry_standard_line(ax1)
    ax1.set_title('LMM vs. OCR\n(Overall Accuracy)', **CHART_STYLES['title'])
    ax1.set_ylabel('Overall Accuracy (%)', **CHART_STYLES['axis_label'])
    ax1.set_ylim(0, 105)
    apply_chart_styling(ax1)
    
    # Upper right: Individual models accuracy
    individual_accuracy = create_individual_model_accuracy_chart(analysis_dir, ax2)
    
    # Lower left: LMM vs OCR CER
    lmm_ocr_cer = create_lmm_vs_ocr_cer_chart(analysis_dir, ax3)
    
    # Lower right: Individual models CER
    individual_cer = create_individual_model_cer_chart(analysis_dir, ax4)
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Compile all performance data
    all_performance_data = {
        'lmm_vs_ocr_accuracy': {'LMM': lmm_performance, 'OCR': ocr_performance},
        'individual_accuracy': individual_accuracy,
        'lmm_vs_ocr_cer': lmm_ocr_cer,
        'individual_cer': individual_cer
    }
    
    return fig, all_performance_data


def print_primary_performance_summary(model_accuracies: Dict[str, Dict[str, float]]) -> None:
    """
    Print the primary performance comparison summary.
    
    Args:
        model_accuracies: Dictionary containing LMM and OCR performance metrics
    """
    lmm_perf = model_accuracies['LMM']
    ocr_perf = model_accuracies['OCR']
    
    print("🎯 PRIMARY PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    print()
    
    print("📊 AGGREGATE PERFORMANCE RESULTS")
    print("-" * 40)
    print(f"Large Multimodal Models (LMM):")
    print(f"  • Overall Accuracy: {lmm_perf['overall_accuracy']:.1%}")
    print(f"  • Work Order Accuracy: {lmm_perf['work_order_accuracy']:.1%}")
    print(f"  • Total Cost Accuracy: {lmm_perf['total_cost_accuracy']:.1%}")
    print()
    
    print(f"Optical Character Recognition (OCR):")
    print(f"  • Overall Accuracy: {ocr_perf['overall_accuracy']:.1%}")
    print(f"  • Work Order Accuracy: {ocr_perf['work_order_accuracy']:.1%}")
    print(f"  • Total Cost Accuracy: {ocr_perf['total_cost_accuracy']:.1%}")
    print()
    
    # Performance gap analysis
    overall_gap = (lmm_perf['overall_accuracy'] - ocr_perf['overall_accuracy']) * 100
    work_order_gap = (lmm_perf['work_order_accuracy'] - ocr_perf['work_order_accuracy']) * 100
    total_cost_gap = (lmm_perf['total_cost_accuracy'] - ocr_perf['total_cost_accuracy']) * 100
    
    print("📈 PERFORMANCE GAP ANALYSIS")
    print("-" * 40)
    print(f"LMM Advantage over OCR:")
    print(f"  • Overall Performance: +{overall_gap:.1f} percentage points")
    print(f"  • Work Order Extraction: +{work_order_gap:.1f} percentage points")
    print(f"  • Total Cost Extraction: +{total_cost_gap:.1f} percentage points")
    print()
    
    # Industry standard comparison
    industry_standard = 0.85
    print("🎯 INDUSTRY AUTOMATION THRESHOLD ANALYSIS")
    print("-" * 40)
    print(f"Industry Standard: {industry_standard:.0%}")
    
    lmm_meets_standard = lmm_perf['overall_accuracy'] >= industry_standard
    ocr_meets_standard = ocr_perf['overall_accuracy'] >= industry_standard
    
    print(f"LMM Performance vs Standard: {'✅ MEETS' if lmm_meets_standard else '❌ BELOW'} "
          f"({lmm_perf['overall_accuracy']:.1%} vs {industry_standard:.0%})")
    print(f"OCR Performance vs Standard: {'✅ MEETS' if ocr_meets_standard else '❌ BELOW'} "
          f"({ocr_perf['overall_accuracy']:.1%} vs {industry_standard:.0%})")
    print()
    
    if lmm_meets_standard and not ocr_meets_standard:
        print("🏆 CONCLUSION: LMM approach achieves production-ready accuracy for automated processing")
    elif not lmm_meets_standard and not ocr_meets_standard:
        print("⚠️  CONCLUSION: Neither approach meets industry automation standards")
    elif lmm_meets_standard and ocr_meets_standard:
        print("✅ CONCLUSION: Both approaches meet industry standards, LMM shows superior performance")
    
    print("=" * 60)


def generate_primary_performance_analysis(analysis_dir: Path) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """
    Generate complete primary performance analysis with chart and summary.
    
    Args:
        analysis_dir: Path to the analysis directory
        
    Returns:
        Tuple of (matplotlib figure, model accuracies dictionary)
    """
    # Create the chart
    fig, model_accuracies = create_primary_performance_comparison_chart(analysis_dir)
    
    # Print the summary  
    print_primary_performance_summary(model_accuracies)
    
    return fig, model_accuracies 


def extract_prompt_performance_data(comprehensive_dataset):
    """Extract aggregate accuracy and CER data by prompt type across LMM models."""
    prompt_accuracy_data = {}
    prompt_cer_data = {}
    
    # Process LMM models (pixtral and llama)
    for model_type in ['pixtral', 'llama']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                # Extract prompt type from metadata
                prompt_type = None
                if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                    if 'prompt_type' in experiment['metadata']['prompt_info']:
                        prompt_type = experiment['metadata']['prompt_info']['prompt_type']
                
                # Look for performance data directly in the experiment structure
                if prompt_type and 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Extract accuracy data
                    if 'work_order_accuracy' in summary and 'total_cost_accuracy' in summary:
                        avg_accuracy = (summary['work_order_accuracy'] + summary['total_cost_accuracy']) / 2
                        
                        if prompt_type not in prompt_accuracy_data:
                            prompt_accuracy_data[prompt_type] = []
                        prompt_accuracy_data[prompt_type].append(avg_accuracy)
                    
                    # Extract CER data (if available)
                    if 'average_cer' in summary:
                        if prompt_type not in prompt_cer_data:
                            prompt_cer_data[prompt_type] = []
                        prompt_cer_data[prompt_type].append(summary['average_cer'])
    
    # Calculate aggregate means for each prompt type
    prompt_accuracy_means = {}
    prompt_cer_means = {}
    
    for prompt_type, accuracies in prompt_accuracy_data.items():
        prompt_accuracy_means[prompt_type] = np.mean(accuracies)
    
    for prompt_type, cers in prompt_cer_data.items():
        prompt_cer_means[prompt_type] = np.mean(cers)
    
    return prompt_accuracy_means, prompt_cer_means


def create_prompt_performance_sidebyside(comprehensive_dataset):
    """Create side-by-side bar charts for accuracy and CER by prompt type."""
    
    # Extract data
    prompt_accuracy, prompt_cer = extract_prompt_performance_data(comprehensive_dataset)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Accuracy by prompt type (highest to lowest)
    if prompt_accuracy:
        # Sort by accuracy (highest to lowest)
        sorted_accuracy = dict(sorted(prompt_accuracy.items(), key=lambda x: x[1], reverse=True))
        
        prompt_names = list(sorted_accuracy.keys())
        accuracy_values = [sorted_accuracy[p] * 100 for p in prompt_names]  # Convert to percentage
        
        # Create bar chart
        bars1 = ax1.bar(range(len(prompt_names)), accuracy_values, 
                       color=ANALYSIS_COLORS['accuracy'], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, accuracy_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Customize left plot
        ax1.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Aggregate Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('LMM Aggregate Accuracy by Prompt Type\n(Sorted: Highest to Lowest)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(range(len(prompt_names)))
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in prompt_names], 
                           rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(accuracy_values) * 1.1)
        
        # Add industry standard line
        ax1.axhline(y=85, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(len(prompt_names)*0.02, 86, 'Industry Standard (85%)', 
                fontsize=10, color='red', fontweight='bold')
    
    # Right plot: CER by prompt type (lowest to highest)
    if prompt_cer:
        # Sort by CER (lowest to highest, since lower is better)
        sorted_cer = dict(sorted(prompt_cer.items(), key=lambda x: x[1]))
        
        prompt_names_cer = list(sorted_cer.keys())
        cer_values = [sorted_cer[p] * 100 for p in prompt_names_cer]  # Convert to percentage
        
        # Create bar chart
        bars2 = ax2.bar(range(len(prompt_names_cer)), cer_values, 
                       color=ANALYSIS_COLORS['cer'], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars2, cer_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Customize right plot
        ax2.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Character Error Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('LMM Character Error Rate by Prompt Type\n(Sorted: Lowest to Highest)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(range(len(prompt_names_cer)))
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in prompt_names_cer], 
                           rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(cer_values) * 1.1)
        
        # Add excellent performance line (CER < 10%)
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(len(prompt_names_cer)*0.02, 10.5, 'Excellent Performance (<10%)', 
                fontsize=10, color='green', fontweight='bold')
    
    # Overall styling
    plt.tight_layout()
    plt.suptitle('LMM Performance Analysis: Aggregate Accuracy vs Character Error Rate by Prompt Type', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Add analysis summary below the plots
    if prompt_accuracy and prompt_cer:
        # Calculate data points safely by counting the original data
        try:
            original_accuracy_data, _ = extract_prompt_performance_data(comprehensive_dataset)
            total_data_points = sum(len(v) if hasattr(v, '__len__') else 1 
                                   for v in original_accuracy_data.values() 
                                   if hasattr(original_accuracy_data, 'values'))
        except:
            total_data_points = len(prompt_accuracy)
        
        fig.text(0.02, -0.05, 
                 f"📊 Analysis Summary:\n"
                 f"   • Best Accuracy Prompt: {max(prompt_accuracy.items(), key=lambda x: x[1])[0].replace('_', ' ').title()} ({max(prompt_accuracy.values())*100:.1f}%)\n"
                 f"   • Lowest CER Prompt: {min(prompt_cer.items(), key=lambda x: x[1])[0].replace('_', ' ').title()} ({min(prompt_cer.values())*100:.1f}%)\n"
                 f"   • Prompt Types Analyzed: {len(prompt_accuracy)}\n"
                 f"   • Data Points: {total_data_points} accuracy measurements",
                 fontsize=10, ha='left', va='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    return fig, prompt_accuracy, prompt_cer


def generate_prompt_performance_analysis(comprehensive_dataset):
    """
    Generate the complete prompt type performance analysis with detailed results.
    
    Args:
        comprehensive_dataset: The comprehensive dataset from data_loader
        
    Returns:
        Tuple of (matplotlib figure, accuracy data dict, CER data dict)
    """
    print("🚀 Generating Prompt Type Performance Analysis...")
    print("="*70)
    
    # Generate the visualization
    fig, accuracy_data, cer_data = create_prompt_performance_sidebyside(comprehensive_dataset)
    
    # Display detailed results
    print("\n" + "="*80)
    print("📈 DETAILED RESULTS BY PROMPT TYPE")  
    print("="*80)
    
    if accuracy_data:
        print("\n🎯 ACCURACY PERFORMANCE (sorted highest to lowest):")
        for prompt_type, accuracy in sorted(accuracy_data.items(), key=lambda x: x[1], reverse=True):
            status = "✅ Above 85%" if accuracy >= 0.85 else "❌ Below 85%"
            print(f"   • {prompt_type.replace('_', ' ').title()}: {accuracy*100:.1f}% {status}")
    
    if cer_data:
        print("\n🎯 CHARACTER ERROR RATE PERFORMANCE (sorted lowest to highest):")
        for prompt_type, cer in sorted(cer_data.items(), key=lambda x: x[1]):
            status = "🌟 Excellent (<10%)" if cer < 0.10 else "⚠️ Needs Improvement (≥10%)"
            print(f"   • {prompt_type.replace('_', ' ').title()}: {cer*100:.1f}% {status}")
    
    print("\n" + "="*80)
    print("✅ Prompt Type Performance Analysis Complete")
    print(f"🎯 Accuracy data points analyzed: {len(accuracy_data)} prompt types")
    print(f"🎯 CER data points analyzed: {len(cer_data)} prompt types")
    print("="*70)
    
    return fig, accuracy_data, cer_data


def extract_field_performance_data(comprehensive_dataset):
    """Extract accuracy and CER data by field type (work order vs total cost) across LMM models."""
    field_accuracy_data = {'work_order': [], 'total_cost': []}
    field_cer_data = {'work_order': [], 'total_cost': []}
    
    # Process LMM models (pixtral and llama)
    for model_type in ['pixtral', 'llama']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                if 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Extract field-specific accuracy data
                    if 'work_order_accuracy' in summary:
                        field_accuracy_data['work_order'].append(summary['work_order_accuracy'])
                    
                    if 'total_cost_accuracy' in summary:
                        field_accuracy_data['total_cost'].append(summary['total_cost_accuracy'])
                    
                    # For CER, try to calculate from individual results if available
                    if 'extracted_data' in experiment:
                        work_order_cers = []
                        total_cost_cers = []
                        
                        for result in experiment['extracted_data']:
                            if 'performance' in result:
                                perf = result['performance']
                                
                                # Extract work order CER if available
                                if 'work_order_cer' in perf:
                                    work_order_cers.append(perf['work_order_cer'])
                                
                                # For total cost CER, it might not be directly available
                                # So we'll use the overall CER or calculate it differently
                                if 'total_cost_cer' in perf:
                                    total_cost_cers.append(perf['total_cost_cer'])
                                elif 'work_order_cer' in perf:
                                    # If only work_order_cer is available, 
                                    # we'll need to use the average_cer for total_cost
                                    # This will be handled below
                                    pass
                        
                        # Add average CER for work order if we have individual CERs
                        if work_order_cers:
                            field_cer_data['work_order'].append(np.mean(work_order_cers))
                        
                        # For total cost CER, if individual CERs aren't available,
                        # use the overall average CER as an approximation
                        if total_cost_cers:
                            field_cer_data['total_cost'].append(np.mean(total_cost_cers))
                        elif 'average_cer' in summary:
                            # Use overall CER as approximation for total cost
                            field_cer_data['total_cost'].append(summary['average_cer'])
                    
                    # Fallback: use overall CER for both fields if individual data not available
                    elif 'average_cer' in summary:
                        # Only add if we haven't already collected data for this experiment
                        if not any('extracted_data' in exp for exp in [experiment]):
                            field_cer_data['work_order'].append(summary['average_cer'])
                            field_cer_data['total_cost'].append(summary['average_cer'])
    
    # Calculate aggregate means for each field type
    field_accuracy_means = {}
    field_cer_means = {}
    
    for field, accuracies in field_accuracy_data.items():
        if accuracies:  # Only calculate if we have data
            field_accuracy_means[field] = np.mean(accuracies)
    
    for field, cers in field_cer_data.items():
        if cers:  # Only calculate if we have data
            field_cer_means[field] = np.mean(cers)
    
    return field_accuracy_means, field_cer_means


def create_field_performance_sidebyside(comprehensive_dataset):
    """Create bar chart for accuracy by field type (work order vs total cost)."""
    
    # Extract data
    field_accuracy, field_cer = extract_field_performance_data(comprehensive_dataset)
    
    # Create figure with single subplot for accuracy only
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Accuracy by field type
    if field_accuracy:
        # Sort by accuracy (highest to lowest)
        sorted_accuracy = dict(sorted(field_accuracy.items(), key=lambda x: x[1], reverse=True))
        
        field_names = list(sorted_accuracy.keys())
        accuracy_values = [sorted_accuracy[f] * 100 for f in field_names]  # Convert to percentage
        
        # Use different colors for each field
        colors = ['#2E86AB', '#A23B72']  # Blue for work order, Purple for total cost
        field_colors = [colors[0] if 'work' in field else colors[1] for field in field_names]
        
        # Create bar chart
        bars = ax.bar(range(len(field_names)), accuracy_values, 
                     color=field_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, accuracy_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Customize plot
        ax.set_xlabel('Field Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Aggregate Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('LMM Accuracy by Field Type\n(Work Order vs Total Cost)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(field_names)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in field_names], fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(accuracy_values) * 1.15)
        
        # Add industry standard line
        ax.axhline(y=85, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(len(field_names)*0.02, 86, 'Industry Standard (85%)', 
               fontsize=12, color='red', fontweight='bold')
    
    # Overall styling
    plt.tight_layout()
    
    # Add analysis summary below the plot
    if field_accuracy:
        # Create color legend and performance summary
        legend_text = "🎨 Color Legend: 🔵 Work Order  🟣 Total Cost"
        
        # Calculate performance gap
        wo_acc = field_accuracy.get('work_order', 0) * 100
        tc_acc = field_accuracy.get('total_cost', 0) * 100
        acc_gap = abs(wo_acc - tc_acc)
        better_field = 'Work Order' if wo_acc > tc_acc else 'Total Cost'
        
        fig.text(0.02, -0.08, 
                 f"📊 Field Performance Summary:\n"
                 f"   • Work Order Accuracy: {wo_acc:.1f}%\n"
                 f"   • Total Cost Accuracy: {tc_acc:.1f}%\n"
                 f"   • Performance Gap: {acc_gap:.1f} percentage points\n"
                 f"   • Better Performing Field: {better_field}\n"
                 f"   {legend_text}",
                 fontsize=11, ha='left', va='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    return fig, field_accuracy, field_cer


def generate_field_performance_analysis(comprehensive_dataset):
    """
    Generate the complete field type performance analysis with detailed results.
    
    Args:
        comprehensive_dataset: The comprehensive dataset from data_loader
        
    Returns:
        Tuple of (matplotlib figure, accuracy data dict, CER data dict)
    """
    print("🚀 Generating Field Type Performance Analysis...")
    print("="*70)
    
    # Generate the visualization
    fig, accuracy_data, cer_data = create_field_performance_sidebyside(comprehensive_dataset)
    
    # Display detailed results
    print("\n" + "="*80)
    print("📈 DETAILED RESULTS BY FIELD TYPE")  
    print("="*80)
    
    if accuracy_data:
        print("\n🎯 ACCURACY PERFORMANCE:")
        for field_type, accuracy in sorted(accuracy_data.items(), key=lambda x: x[1], reverse=True):
            status = "✅ Above 85%" if accuracy >= 0.85 else "❌ Below 85%"
            print(f"   • {field_type.replace('_', ' ').title()}: {accuracy*100:.1f}% {status}")
    
    # Comparative analysis
    if len(accuracy_data) == 2:
        wo_acc = accuracy_data.get('work_order', 0)
        tc_acc = accuracy_data.get('total_cost', 0)
        acc_diff = abs(wo_acc - tc_acc) * 100
        better_field = 'Work Order' if wo_acc > tc_acc else 'Total Cost'
        
        print(f"\n🔍 COMPARATIVE ANALYSIS:")
        print(f"   • Better Accuracy Field: {better_field} (+{acc_diff:.1f}pp advantage)")
        
        # Industry standard analysis
        industry_standard = 0.85
        wo_meets = wo_acc >= industry_standard
        tc_meets = tc_acc >= industry_standard
        
        print(f"   • Work Order meets 85% standard: {'✅ Yes' if wo_meets else '❌ No'}")
        print(f"   • Total Cost meets 85% standard: {'✅ Yes' if tc_meets else '❌ No'}")
        
        if acc_diff > 5:
            print(f"   • ⚠️  Notable performance gap: {acc_diff:.1f} percentage points")
            print(f"   • 💡 Consider optimizing {better_field.lower()} extraction strategies")
        else:
            print(f"   • ✅ Balanced performance across both fields")
    
    print("\n" + "="*80)
    print("✅ Field Type Performance Analysis Complete")
    print(f"🎯 Field types analyzed: {len(accuracy_data)}")
    print("📊 Focus: Accuracy comparison (CER analysis removed for clarity)")
    print("="*70)
    
    return fig, accuracy_data, cer_data


# ====================
# HEATMAP FUNCTIONS
# ====================

def create_lmm_prompt_heatmap_data(comprehensive_dataset):
    """Create data structure for LMM Models vs Prompts heatmap."""
    heatmap_data = {}
    
    # Process LMM models (pixtral and llama)
    for model_type in ['pixtral', 'llama']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                # Extract prompt type and model info
                prompt_type = None
                model_name = model_type.title()
                
                if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                    if 'prompt_type' in experiment['metadata']['prompt_info']:
                        prompt_type = experiment['metadata']['prompt_info']['prompt_type']
                
                if prompt_type and 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Calculate average accuracy
                    if 'work_order_accuracy' in summary and 'total_cost_accuracy' in summary:
                        avg_accuracy = (summary['work_order_accuracy'] + summary['total_cost_accuracy']) / 2
                        
                        # Store in heatmap structure
                        if model_name not in heatmap_data:
                            heatmap_data[model_name] = {}
                        
                        if prompt_type not in heatmap_data[model_name]:
                            heatmap_data[model_name][prompt_type] = avg_accuracy
    
    return heatmap_data


def plot_lmm_prompt_accuracy_heatmap(heatmap_data):
    """Create heatmap visualization for LMM Models vs Prompts accuracy."""
    
    # Prepare data for heatmap
    data_rows = []
    for model, prompts in heatmap_data.items():
        for prompt, accuracy in prompts.items():
            data_rows.append({'Model': model, 'Prompt': prompt, 'Accuracy': accuracy})
    
    if not data_rows:
        print("No data available for LMM Models vs Prompts heatmap")
        return plt.figure()
    
    df = pd.DataFrame(data_rows)
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Model', columns='Prompt', values='Accuracy')
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    
    # Create heatmap with custom colormap
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn', 
                center=0.85,
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Accuracy'},
                linewidths=0.5)
    
    plt.title('LMM Models vs Prompts - Accuracy Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prompt Type', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add industry standard note
    plt.figtext(0.02, 0.02, 'Industry Standard: 85% accuracy', fontsize=10, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def create_lmm_prompt_cer_heatmap_data(comprehensive_dataset):
    """Create data structure for LMM Models vs Prompts CER heatmap."""
    heatmap_data = {}
    
    # Process LMM models (pixtral and llama)
    for model_type in ['pixtral', 'llama']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                # Extract prompt type and model info
                prompt_type = None
                model_name = model_type.title()
                
                if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                    if 'prompt_type' in experiment['metadata']['prompt_info']:
                        prompt_type = experiment['metadata']['prompt_info']['prompt_type']
                
                if prompt_type and 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Use average CER if available
                    if 'average_cer' in summary:
                        cer = summary['average_cer']
                        
                        # Store in heatmap structure
                        if model_name not in heatmap_data:
                            heatmap_data[model_name] = {}
                        
                        if prompt_type not in heatmap_data[model_name]:
                            heatmap_data[model_name][prompt_type] = cer
    
    return heatmap_data


def plot_lmm_prompt_cer_heatmap(heatmap_data):
    """Create heatmap visualization for LMM Models vs Prompts CER."""
    
    # Prepare data for heatmap
    data_rows = []
    for model, prompts in heatmap_data.items():
        for prompt, cer in prompts.items():
            data_rows.append({'Model': model, 'Prompt': prompt, 'CER': cer})
    
    if not data_rows:
        print("No data available for LMM Models vs Prompts CER heatmap")
        return plt.figure()
    
    df = pd.DataFrame(data_rows)
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Model', columns='Prompt', values='CER')
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    
    # Create heatmap with reversed colormap (lower CER is better, so should be green)
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn_r', 
                center=0.1,
                vmin=0, 
                vmax=0.5,
                cbar_kws={'label': 'Character Error Rate (CER)'},
                linewidths=0.5)
    
    plt.title('LMM Models vs Prompts - Character Error Rate (CER) Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prompt Type', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add performance note
    plt.figtext(0.02, 0.02, 'Lower CER values indicate better performance', fontsize=10, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def create_lmm_prompt_query_heatmap_data(comprehensive_dataset):
    """Create data structure for LMM Prompts vs Query types heatmap."""
    heatmap_data = {}
    
    # Process LMM models (pixtral and llama)
    for model_type in ['pixtral', 'llama']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                # Extract prompt type
                prompt_type = None
                
                if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                    if 'prompt_type' in experiment['metadata']['prompt_info']:
                        prompt_type = experiment['metadata']['prompt_info']['prompt_type']
                
                if prompt_type and 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Extract field-specific accuracies
                    work_order_acc = summary.get('work_order_accuracy', 0)
                    total_cost_acc = summary.get('total_cost_accuracy', 0)
                    
                    if prompt_type not in heatmap_data:
                        heatmap_data[prompt_type] = {
                            'Work Order': [],
                            'Total Cost': []
                        }
                    
                    heatmap_data[prompt_type]['Work Order'].append(work_order_acc)
                    heatmap_data[prompt_type]['Total Cost'].append(total_cost_acc)
    
    # Average the results for each combination
    for prompt_type, queries in heatmap_data.items():
        for query_type, values in queries.items():
            if values:
                heatmap_data[prompt_type][query_type] = np.mean(values)
            else:
                heatmap_data[prompt_type][query_type] = 0
    
    return heatmap_data


def plot_lmm_prompt_query_heatmap(heatmap_data):
    """Create heatmap visualization for LMM Prompts vs Query types."""
    
    # Prepare data for heatmap
    data_rows = []
    for prompt, queries in heatmap_data.items():
        for query_type, accuracy in queries.items():
            data_rows.append({'Query Type': query_type, 'Prompt': prompt, 'Accuracy': accuracy})
    
    if not data_rows:
        print("No data available for LMM Prompts vs Query types heatmap")
        return plt.figure()
    
    df = pd.DataFrame(data_rows)
    
    # Pivot for heatmap - flipped orientation: Query Type as rows, Prompt as columns
    pivot_df = df.pivot(index='Query Type', columns='Prompt', values='Accuracy')
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    
    # Create heatmap with custom colormap
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn', 
                center=0.85,
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Accuracy'},
                linewidths=0.5)
    
    plt.title('LMM Query Types vs Prompts - Accuracy Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prompt Type', fontsize=12, fontweight='bold')
    plt.ylabel('Query Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add industry standard note
    plt.figtext(0.02, 0.02, 'Industry Standard: 85% accuracy', fontsize=10, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def create_lmm_prompt_query_cer_heatmap_data(comprehensive_dataset):
    """Create data structure for LMM Prompts vs Query types CER heatmap."""
    heatmap_data = {}
    
    # Process LMM models (pixtral and llama)
    for model_type in ['pixtral', 'llama']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                # Extract prompt type
                prompt_type = None
                
                if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                    if 'prompt_type' in experiment['metadata']['prompt_info']:
                        prompt_type = experiment['metadata']['prompt_info']['prompt_type']
                
                if prompt_type and 'extracted_data' in experiment:
                    # Try to extract individual CERs from performance data
                    work_order_cers = []
                    total_cost_cers = []
                    
                    for result in experiment['extracted_data']:
                        if 'performance' in result:
                            perf = result['performance']
                            if 'work_order_cer' in perf:
                                work_order_cers.append(perf['work_order_cer'])
                                # Use work_order_cer for total_cost if total_cost_cer not available
                                total_cost_cers.append(perf['work_order_cer'])
                    
                    if work_order_cers:
                        if prompt_type not in heatmap_data:
                            heatmap_data[prompt_type] = {
                                'Work Order': [],
                                'Total Cost': []
                            }
                        
                        heatmap_data[prompt_type]['Work Order'].extend(work_order_cers)
                        heatmap_data[prompt_type]['Total Cost'].extend(total_cost_cers)
    
    # Average the results for each combination
    for prompt_type, queries in heatmap_data.items():
        for query_type, values in queries.items():
            if values:
                heatmap_data[prompt_type][query_type] = np.mean(values)
            else:
                heatmap_data[prompt_type][query_type] = 0
    
    return heatmap_data


def plot_lmm_prompt_query_cer_heatmap(heatmap_data):
    """Create heatmap visualization for LMM Prompts vs Query types CER."""
    
    # Prepare data for heatmap
    data_rows = []
    for prompt, queries in heatmap_data.items():
        for query_type, cer in queries.items():
            data_rows.append({'Query Type': query_type, 'Prompt': prompt, 'CER': cer})
    
    if not data_rows:
        print("No data available for LMM Prompts vs Query types CER heatmap")
        return plt.figure()
    
    df = pd.DataFrame(data_rows)
    
    # Pivot for heatmap - Query Type as rows, Prompt as columns
    pivot_df = df.pivot(index='Query Type', columns='Prompt', values='CER')
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    
    # Create heatmap with reversed colormap (lower CER is better)
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn_r', 
                center=0.1,
                vmin=0, 
                vmax=0.5,
                cbar_kws={'label': 'Character Error Rate (CER)'},
                linewidths=0.5)
    
    plt.title('LMM Query Types vs Prompts - Character Error Rate (CER) Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prompt Type', fontsize=12, fontweight='bold')
    plt.ylabel('Query Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add performance note
    plt.figtext(0.02, 0.02, 'Lower CER values indicate better performance', fontsize=10, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def create_all_models_query_heatmap_data(comprehensive_dataset):
    """Create data structure for All Models vs Query types heatmap."""
    heatmap_data = {}
    
    # Process all model types including OCR
    for model_type in ['pixtral', 'llama', 'doctr']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                # Get model name and create unique identifier
                model_name = model_type.title()
                
                # For LMM models, add prompt type to model name
                if model_type in ['pixtral', 'llama']:
                    if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                        prompt_type = experiment['metadata']['prompt_info'].get('prompt_type', 'unknown')
                        model_name = f"{model_name}-{prompt_type}"
                else:
                    # For OCR models, use different naming
                    model_name = f"docTR-{experiment.get('metadata', {}).get('test_id', 'unknown')}"
                
                if 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Extract field-specific accuracies
                    work_order_acc = summary.get('work_order_accuracy', 0)
                    total_cost_acc = summary.get('total_cost_accuracy', 0)
                    
                    if model_name not in heatmap_data:
                        heatmap_data[model_name] = {}
                    
                    heatmap_data[model_name]['Work Order'] = work_order_acc
                    heatmap_data[model_name]['Total Cost'] = total_cost_acc
    
    return heatmap_data


def plot_all_models_query_heatmap(heatmap_data):
    """Create heatmap visualization for All Models vs Query types."""
    
    # Prepare data for heatmap
    data_rows = []
    for model, queries in heatmap_data.items():
        for query_type, accuracy in queries.items():
            data_rows.append({'Query Type': query_type, 'Model': model, 'Accuracy': accuracy})
    
    if not data_rows:
        print("No data available for All Models vs Query types heatmap")
        return plt.figure()
    
    df = pd.DataFrame(data_rows)
    
    # Sort models by average performance for better visualization
    model_avg = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    sorted_models = model_avg.index.tolist()
    
    # Pivot for heatmap - Query Type as rows, Model as columns
    pivot_df = df.pivot(index='Query Type', columns='Model', values='Accuracy')
    pivot_df = pivot_df[sorted_models]  # Sort columns by performance
    
    # Create the heatmap
    plt.figure(figsize=(16, 6))
    
    # Create heatmap with custom colormap
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn', 
                center=0.85,
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Accuracy'},
                linewidths=0.5)
    
    plt.title('All Models vs Query Types - Accuracy Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Query Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add industry standard note
    plt.figtext(0.02, 0.02, 'Industry Standard: 85% accuracy | Sorted by average performance', fontsize=10, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


# ====================
# HEATMAP GENERATOR FUNCTIONS
# ====================

def generate_lmm_models_prompts_accuracy_heatmap(comprehensive_dataset):
    """Generate LMM Models vs Prompts accuracy heatmap analysis."""
    print("🔍 Creating LMM Models vs Prompts Accuracy Heatmap...")
    
    heatmap_data = create_lmm_prompt_heatmap_data(comprehensive_dataset)
    fig = plot_lmm_prompt_accuracy_heatmap(heatmap_data)
    
    # Analysis summary
    if heatmap_data:
        print(f"\n📊 HEATMAP ANALYSIS SUMMARY:")
        print(f"   • Models analyzed: {len(heatmap_data)}")
        print(f"   • Prompt types: {len(set().union(*[prompts.keys() for prompts in heatmap_data.values()]))}")
        
        # Find best performing combinations
        best_combo = None
        best_score = 0
        for model, prompts in heatmap_data.items():
            for prompt, score in prompts.items():
                if score > best_score:
                    best_score = score
                    best_combo = (model, prompt)
        
        if best_combo:
            print(f"   • Best combination: {best_combo[0]} + {best_combo[1]} ({best_score:.1%})")
    
    return fig, heatmap_data


def generate_lmm_models_prompts_cer_heatmap(comprehensive_dataset):
    """Generate LMM Models vs Prompts CER heatmap analysis."""
    print("🔍 Creating LMM Models vs Prompts CER Heatmap...")
    
    heatmap_data = create_lmm_prompt_cer_heatmap_data(comprehensive_dataset)
    fig = plot_lmm_prompt_cer_heatmap(heatmap_data)
    
    # Analysis summary
    if heatmap_data:
        print(f"\n📊 CER HEATMAP ANALYSIS SUMMARY:")
        print(f"   • Models analyzed: {len(heatmap_data)}")
        print(f"   • Prompt types: {len(set().union(*[prompts.keys() for prompts in heatmap_data.values()]))}")
        
        # Find best performing combinations (lowest CER)
        best_combo = None
        best_score = float('inf')
        for model, prompts in heatmap_data.items():
            for prompt, score in prompts.items():
                if score < best_score:
                    best_score = score
                    best_combo = (model, prompt)
        
        if best_combo:
            print(f"   • Best combination (lowest CER): {best_combo[0]} + {best_combo[1]} ({best_score:.1%})")
    
    return fig, heatmap_data


def generate_lmm_query_prompts_accuracy_heatmap(comprehensive_dataset):
    """Generate LMM Query Types vs Prompts accuracy heatmap analysis."""
    print("🔍 Creating LMM Query Types vs Prompts Accuracy Heatmap...")
    
    heatmap_data = create_lmm_prompt_query_heatmap_data(comprehensive_dataset)
    fig = plot_lmm_prompt_query_heatmap(heatmap_data)
    
    # Analysis summary
    if heatmap_data:
        print(f"\n📊 QUERY-PROMPT HEATMAP ANALYSIS:")
        print(f"   • Prompt types analyzed: {len(heatmap_data)}")
        print(f"   • Query types: Work Order, Total Cost")
        
        # Compare performance across query types
        wo_scores = [queries.get('Work Order', 0) for queries in heatmap_data.values()]
        tc_scores = [queries.get('Total Cost', 0) for queries in heatmap_data.values()]
        
        if wo_scores and tc_scores:
            print(f"   • Average Work Order accuracy: {np.mean(wo_scores):.1%}")
            print(f"   • Average Total Cost accuracy: {np.mean(tc_scores):.1%}")
    
    return fig, heatmap_data


def generate_lmm_query_prompts_cer_heatmap(comprehensive_dataset):
    """Generate LMM Query Types vs Prompts CER heatmap analysis."""
    print("🔍 Creating LMM Query Types vs Prompts CER Heatmap...")
    
    heatmap_data = create_lmm_prompt_query_cer_heatmap_data(comprehensive_dataset)
    fig = plot_lmm_prompt_query_cer_heatmap(heatmap_data)
    
    # Analysis summary
    if heatmap_data:
        print(f"\n📊 QUERY-PROMPT CER HEATMAP ANALYSIS:")
        print(f"   • Prompt types analyzed: {len(heatmap_data)}")
        print(f"   • Query types: Work Order, Total Cost")
        
        # Compare CER across query types
        wo_cers = [queries.get('Work Order', 0) for queries in heatmap_data.values()]
        tc_cers = [queries.get('Total Cost', 0) for queries in heatmap_data.values()]
        
        if wo_cers and tc_cers:
            print(f"   • Average Work Order CER: {np.mean(wo_cers):.1%}")
            print(f"   • Average Total Cost CER: {np.mean(tc_cers):.1%}")
    
    return fig, heatmap_data


def generate_all_models_query_heatmap(comprehensive_dataset):
    """Generate All Models vs Query Types accuracy heatmap analysis."""
    print("🔍 Creating All Models vs Query Types Accuracy Heatmap...")
    
    heatmap_data = create_all_models_query_heatmap_data(comprehensive_dataset)
    fig = plot_all_models_query_heatmap(heatmap_data)
    
    # Analysis summary
    if heatmap_data:
        print(f"\n📊 ALL MODELS HEATMAP ANALYSIS:")
        print(f"   • Total model configurations: {len(heatmap_data)}")
        print(f"   • Query types: Work Order, Total Cost")
        
        # Calculate overall performance statistics
        all_scores = []
        for model, queries in heatmap_data.items():
            all_scores.extend(queries.values())
        
        if all_scores:
            print(f"   • Overall average accuracy: {np.mean(all_scores):.1%}")
            print(f"   • Best performing model: {max(heatmap_data.items(), key=lambda x: np.mean(list(x[1].values())))[0]}")
    
    return fig, heatmap_data


# ====================
# PERFORMANCE RANGE FUNCTIONS
# ====================

def calculate_model_performance_ranges(comprehensive_dataset):
    """Calculate performance ranges (max-min) and statistics for each model type."""
    model_stats = {}
    
    # Process each model type
    for model_type in ['pixtral', 'llama', 'doctr']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            # Collect all accuracy values for this model type
            accuracies = []
            for experiment in experiments:
                if 'summary' in experiment:
                    summary = experiment['summary']
                    # Calculate average accuracy across work order and total cost
                    if 'work_order_accuracy' in summary and 'total_cost_accuracy' in summary:
                        avg_acc = (summary['work_order_accuracy'] + summary['total_cost_accuracy']) / 2
                        accuracies.append(avg_acc)
            
            if accuracies:
                model_name = model_type.title()
                model_stats[model_name] = {
                    'accuracies': accuracies,
                    'min': min(accuracies),
                    'max': max(accuracies),
                    'mean': np.mean(accuracies),
                    'range': max(accuracies) - min(accuracies),
                    'std': np.std(accuracies)
                }
    
    return model_stats


def calculate_model_cer_ranges(comprehensive_dataset):
    """Calculate CER ranges (max-min) and statistics for each model type."""
    model_stats = {}
    
    # Process each model type
    for model_type in ['pixtral', 'llama', 'doctr']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            # Collect all CER values for this model type
            cers = []
            for experiment in experiments:
                if 'summary' in experiment:
                    summary = experiment['summary']
                    # Use average CER if available
                    if 'average_cer' in summary:
                        cers.append(summary['average_cer'])
            
            if cers:
                model_name = model_type.title()
                model_stats[model_name] = {
                    'cers': cers,
                    'min': min(cers),
                    'max': max(cers),
                    'mean': np.mean(cers),
                    'range': max(cers) - min(cers),
                    'std': np.std(cers)
                }
    
    return model_stats


def plot_model_performance_ranges(model_stats):
    """Create side-by-side performance range visualization."""
    
    if not model_stats:
        print("No model statistics available for performance range analysis")
        return plt.figure()
    
    # Prepare data
    models = list(model_stats.keys())
    ranges = [stats['range'] for stats in model_stats.values()]
    means = [stats['mean'] for stats in model_stats.values()]
    mins = [stats['min'] for stats in model_stats.values()]
    maxs = [stats['max'] for stats in model_stats.values()]
    
    # Sort by range (consistency - lower is better)
    sorted_indices = np.argsort(ranges)
    models_sorted = [models[i] for i in sorted_indices]
    ranges_sorted = [ranges[i] for i in sorted_indices]
    means_sorted = [means[i] for i in sorted_indices]
    mins_sorted = [mins[i] for i in sorted_indices]
    maxs_sorted = [maxs[i] for i in sorted_indices]
    
    # Color mapping
    colors = [ANALYSIS_COLORS['accuracy'] if r < 0.1 else 
              ANALYSIS_COLORS['work_order'] if r < 0.15 else 
              ANALYSIS_COLORS['cer'] for r in ranges_sorted]
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Performance Range (Bar Chart)
    bars1 = ax1.bar(models_sorted, ranges_sorted, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, ranges_sorted)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_title('Model Performance Range\n(Max - Min Accuracy)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Range', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(ranges_sorted) * 1.2)
    
    # Add consistency note
    ax1.text(0.02, 0.98, 'Lower values = More consistent', 
             transform=ax1.transAxes, fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
             verticalalignment='top')
    
    # Right plot: Performance Range with Mean (Error Bar Chart)
    x_pos = range(len(models_sorted))
    
    # Create error bars (from min to max)
    yerr_lower = [means_sorted[i] - mins_sorted[i] for i in range(len(means_sorted))]
    yerr_upper = [maxs_sorted[i] - means_sorted[i] for i in range(len(means_sorted))]
    
    # Plot error bars
    ax2.errorbar(x_pos, means_sorted, 
                yerr=[yerr_lower, yerr_upper],
                fmt='none', capsize=8, capthick=2, elinewidth=3,
                color=ANALYSIS_COLORS['LMM'], alpha=0.7)
    
    # Plot mean points
    scatter = ax2.scatter(x_pos, means_sorted, s=100, c=colors, alpha=0.8, 
                         edgecolors='black', linewidth=1, zorder=3)
    
    # Add mean value labels
    for i, (x, mean_val) in enumerate(zip(x_pos, means_sorted)):
        ax2.text(x, mean_val + 0.02, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_title('Performance Range: Min-Max with Mean\n(Dots = Mean, Lines = Range)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_sorted)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    # Add industry standard line
    ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Industry Standard (85%)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_model_cer_ranges(model_stats):
    """Create side-by-side CER range visualization."""
    
    if not model_stats:
        print("No model statistics available for CER range analysis")
        return plt.figure()
    
    # Prepare data
    models = list(model_stats.keys())
    ranges = [stats['range'] for stats in model_stats.values()]
    means = [stats['mean'] for stats in model_stats.values()]
    mins = [stats['min'] for stats in model_stats.values()]
    maxs = [stats['max'] for stats in model_stats.values()]
    
    # Sort by range (consistency - lower is better)
    sorted_indices = np.argsort(ranges)
    models_sorted = [models[i] for i in sorted_indices]
    ranges_sorted = [ranges[i] for i in sorted_indices]
    means_sorted = [means[i] for i in sorted_indices]
    mins_sorted = [mins[i] for i in sorted_indices]
    maxs_sorted = [maxs[i] for i in sorted_indices]
    
    # Color mapping (for CER, lower is better)
    colors = [ANALYSIS_COLORS['accuracy'] if r < 0.05 else 
              ANALYSIS_COLORS['work_order'] if r < 0.1 else 
              ANALYSIS_COLORS['cer'] for r in ranges_sorted]
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: CER Range (Bar Chart)
    bars1 = ax1.bar(models_sorted, ranges_sorted, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, ranges_sorted)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(ranges_sorted) * 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_title('Model CER Range\n(Max - Min CER)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CER Range', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(ranges_sorted) * 1.2)
    
    # Add consistency note
    ax1.text(0.02, 0.98, 'Lower values = More consistent', 
             transform=ax1.transAxes, fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
             verticalalignment='top')
    
    # Right plot: CER Range with Mean (Error Bar Chart)
    x_pos = range(len(models_sorted))
    
    # Create error bars (from min to max)
    yerr_lower = [means_sorted[i] - mins_sorted[i] for i in range(len(means_sorted))]
    yerr_upper = [maxs_sorted[i] - means_sorted[i] for i in range(len(means_sorted))]
    
    # Plot error bars
    ax2.errorbar(x_pos, means_sorted, 
                yerr=[yerr_lower, yerr_upper],
                fmt='none', capsize=8, capthick=2, elinewidth=3,
                color=ANALYSIS_COLORS['LMM'], alpha=0.7)
    
    # Plot mean points
    scatter = ax2.scatter(x_pos, means_sorted, s=100, c=colors, alpha=0.8, 
                         edgecolors='black', linewidth=1, zorder=3)
    
    # Add mean value labels
    for i, (x, mean_val) in enumerate(zip(x_pos, means_sorted)):
        ax2.text(x, mean_val + max(means_sorted) * 0.02, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_title('CER Range: Min-Max with Mean\n(Dots = Mean, Lines = Range)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Character Error Rate (CER)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_sorted)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(maxs_sorted) * 1.1)
    
    plt.tight_layout()
    return fig


# ====================
# PERFORMANCE RANGE GENERATOR FUNCTIONS
# ====================

def generate_model_performance_range_analysis(comprehensive_dataset):
    """Generate complete model performance range analysis."""
    print("🔍 Calculating Model Performance Ranges...")
    
    model_stats = calculate_model_performance_ranges(comprehensive_dataset)
    fig = plot_model_performance_ranges(model_stats)
    
    # Analysis summary
    if model_stats:
        print(f"\n📊 PERFORMANCE RANGE ANALYSIS:")
        print(f"   • Models analyzed: {len(model_stats)}")
        
        # Find most and least consistent models
        ranges = [(model, stats['range']) for model, stats in model_stats.items()]
        ranges.sort(key=lambda x: x[1])
        
        if ranges:
            most_consistent = ranges[0]
            least_consistent = ranges[-1]
            print(f"   • Most consistent: {most_consistent[0]} (range: {most_consistent[1]:.3f})")
            print(f"   • Least consistent: {least_consistent[0]} (range: {least_consistent[1]:.3f})")
            
        # Performance statistics
        all_means = [stats['mean'] for stats in model_stats.values()]
        print(f"   • Average performance across models: {np.mean(all_means):.1%}")
        
        # Industry standard analysis
        above_standard = sum(1 for mean in all_means if mean >= 0.85)
        print(f"   • Models meeting industry standard (≥85%): {above_standard}/{len(all_means)}")
    
    return fig, model_stats


def generate_model_cer_range_analysis(comprehensive_dataset):
    """Generate complete model CER range analysis."""
    print("🔍 Calculating Model CER Ranges...")
    
    model_stats = calculate_model_cer_ranges(comprehensive_dataset)
    fig = plot_model_cer_ranges(model_stats)
    
    # Analysis summary
    if model_stats:
        print(f"\n📊 CER RANGE ANALYSIS:")
        print(f"   • Models analyzed: {len(model_stats)}")
        
        # Find most and least consistent models
        ranges = [(model, stats['range']) for model, stats in model_stats.items()]
        ranges.sort(key=lambda x: x[1])
        
        if ranges:
            most_consistent = ranges[0]
            least_consistent = ranges[-1]
            print(f"   • Most consistent: {most_consistent[0]} (CER range: {most_consistent[1]:.3f})")
            print(f"   • Least consistent: {least_consistent[0]} (CER range: {least_consistent[1]:.3f})")
            
        # CER statistics
        all_means = [stats['mean'] for stats in model_stats.values()]
        print(f"   • Average CER across models: {np.mean(all_means):.1%}")
        
        # Best performing model (lowest CER)
        best_model = min(model_stats.items(), key=lambda x: x[1]['mean'])
        print(f"   • Best performing model (lowest CER): {best_model[0]} ({best_model[1]['mean']:.1%})")
    
    return fig, model_stats


# ====================
# PERFORMANCE GAP FUNCTIONS
# ====================

def create_performance_gap_data(comprehensive_dataset):
    """Create performance gap data for analysis."""
    gap_data = []
    
    # Process all model types
    for model_type in ['pixtral', 'llama', 'doctr']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                if 'summary' in experiment:
                    summary = experiment['summary']
                    
                    # Calculate gap (Total Cost - Work Order accuracy)
                    if 'work_order_accuracy' in summary and 'total_cost_accuracy' in summary:
                        work_order_acc = summary['work_order_accuracy']
                        total_cost_acc = summary['total_cost_accuracy']
                        gap = total_cost_acc - work_order_acc
                        overall_acc = (work_order_acc + total_cost_acc) / 2
                        
                        # Get model details
                        model_name = model_type.title()
                        if model_type in ['pixtral', 'llama']:
                            # Add prompt type for LMM models
                            if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                                prompt_type = experiment['metadata']['prompt_info'].get('prompt_type', 'unknown')
                                model_label = f"{model_name}-{prompt_type}"
                            else:
                                model_label = model_name
                        else:
                            # OCR model
                            test_id = experiment.get('metadata', {}).get('test_id', 'unknown')
                            model_label = f"docTR-{test_id}"
                        
                        # Get CER if available
                        cer = summary.get('average_cer', 0)
                        
                        gap_data.append({
                            'model_type': model_type,
                            'model_label': model_label,
                            'work_order_accuracy': work_order_acc,
                            'total_cost_accuracy': total_cost_acc,
                            'gap': gap,
                            'overall_accuracy': overall_acc,
                            'cer': cer
                        })
    
    return gap_data


def plot_performance_gap_scatter_sidebyside(gap_data):
    """Create side-by-side scatter plots for performance gap analysis."""
    
    if not gap_data:
        print("No gap data available for analysis")
        return plt.figure()
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color mapping by model type
    color_map = {
        'pixtral': ANALYSIS_COLORS['Pixtral'],
        'llama': ANALYSIS_COLORS['Llama'], 
        'doctr': ANALYSIS_COLORS['DocTR']
    }
    
    # Left plot: Overall Accuracy vs Gap (both axes flipped for worst to best flow)
    for data_point in gap_data:
        model_type = data_point['model_type']
        color = color_map.get(model_type, ANALYSIS_COLORS['baseline'])
        
        # Invert gap (negative gap for plotting so lower gap appears higher)
        ax1.scatter(data_point['overall_accuracy'], -data_point['gap'], 
                   s=100, c=color, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add model label
        ax1.annotate(data_point['model_label'], 
                    (data_point['overall_accuracy'], -data_point['gap']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # Styling for left plot
    ax1.set_xlabel('Overall Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gap Quality (Inverted Gap)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy vs Gap Quality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero Gap')
    ax1.axvline(x=0.85, color='red', linestyle='--', alpha=0.7, linewidth=2, label='85% Standard')
    
    # Custom y-axis labels for inverted gap
    gap_values = [d['gap'] for d in gap_data]
    if gap_values:
        min_gap, max_gap = min(gap_values), max(gap_values)
        # Create tick positions (negative values) and labels (positive gap values)
        tick_positions = [-max_gap, -max_gap*0.5, 0, -min_gap*0.5, -min_gap]
        tick_labels = [f'{max_gap:.2f}', f'{max_gap*0.5:.2f}', '0.00', f'{min_gap*0.5:.2f}', f'{min_gap:.2f}']
        ax1.set_yticks(tick_positions)
        ax1.set_yticklabels(tick_labels)
        ax1.text(-0.15, 0.5, 'Lower Gap (Better) ↑↓ Higher Gap (Worse)', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=9, style='italic', rotation=90)
    
    # Add quadrant labels for left plot (worst to best flow)
    ax1.text(0.05, 0.05, 'WORST\nLow Acc,\nHigh Gap', transform=ax1.transAxes, 
             fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    ax1.text(0.95, 0.95, 'BEST\nHigh Acc,\nLow Gap', transform=ax1.transAxes, 
             fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax1.text(0.95, 0.05, 'High Acc,\nHigh Gap', transform=ax1.transAxes, 
             fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    ax1.text(0.05, 0.95, 'Low Acc,\nLow Gap', transform=ax1.transAxes, 
             fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax1.legend()
    
    # Right plot: CER vs Gap (both axes flipped for worst to best flow)
    for data_point in gap_data:
        model_type = data_point['model_type']
        color = color_map.get(model_type, ANALYSIS_COLORS['baseline'])
        
        # Flip both CER and gap axes for plotting
        ax2.scatter(-data_point['cer'], -data_point['gap'], 
                   s=100, c=color, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add model label
        ax2.annotate(data_point['model_label'], 
                    (-data_point['cer'], -data_point['gap']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # Styling for right plot
    ax2.set_xlabel('Performance Quality (Inverted CER)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gap Quality (Inverted Gap)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Quality vs Gap Quality', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero Gap')
    
    # Custom x-axis labels for inverted CER
    cer_values = [d['cer'] for d in gap_data]
    if cer_values:
        min_cer, max_cer = min(cer_values), max(cer_values)
        # Create tick positions (negative values) and labels (positive CER values)
        tick_positions = [-max_cer, -max_cer*0.75, -max_cer*0.5, -max_cer*0.25, -min_cer]
        tick_labels = [f'{max_cer:.2f}', f'{max_cer*0.75:.2f}', f'{max_cer*0.5:.2f}', f'{max_cer*0.25:.2f}', f'{min_cer:.2f}']
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)
        ax2.text(0.5, -0.1, 'Lower CER (Better) ←→ Higher CER (Worse)', 
                transform=ax2.transAxes, ha='center', va='top', fontsize=9, style='italic')
    
    # Custom y-axis labels for inverted gap
    gap_values = [d['gap'] for d in gap_data]
    if gap_values:
        min_gap, max_gap = min(gap_values), max(gap_values)
        # Create tick positions (negative values) and labels (positive gap values)
        tick_positions = [-max_gap, -max_gap*0.5, 0, -min_gap*0.5, -min_gap]
        tick_labels = [f'{max_gap:.2f}', f'{max_gap*0.5:.2f}', '0.00', f'{min_gap*0.5:.2f}', f'{min_gap:.2f}']
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels(tick_labels)
        ax2.text(-0.15, 0.5, 'Lower Gap (Better) ↑↓ Higher Gap (Worse)', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=9, style='italic', rotation=90)
    
    # Add quadrant labels for right plot (worst to best flow)
    ax2.text(0.05, 0.05, 'WORST\nHigh CER,\nHigh Gap', transform=ax2.transAxes, 
             fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    ax2.text(0.95, 0.95, 'BEST\nLow CER,\nLow Gap', transform=ax2.transAxes, 
             fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax2.text(0.95, 0.05, 'Low CER,\nHigh Gap', transform=ax2.transAxes, 
             fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    ax2.text(0.05, 0.95, 'High CER,\nLow Gap', transform=ax2.transAxes, 
             fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig


# ====================
# PERFORMANCE GAP GENERATOR FUNCTION
# ====================

def generate_performance_gap_scatter_analysis(comprehensive_dataset):
    """Generate complete performance gap scatter analysis."""
    print("🔍 Creating Performance Gap Scatter Analysis...")
    
    gap_data = create_performance_gap_data(comprehensive_dataset)
    fig = plot_performance_gap_scatter_sidebyside(gap_data)
    
    # Analysis summary
    if gap_data:
        print(f"\n📊 PERFORMANCE GAP ANALYSIS:")
        print(f"   • Model configurations analyzed: {len(gap_data)}")
        
        # Calculate gap statistics
        gaps = [d['gap'] for d in gap_data]
        positive_gaps = [g for g in gaps if g > 0]
        negative_gaps = [g for g in gaps if g < 0]
        
        print(f"   • Average gap (TC - WO): {np.mean(gaps):.3f}")
        print(f"   • Models where TC is easier: {len(positive_gaps)}/{len(gaps)}")
        print(f"   • Models where WO is easier: {len(negative_gaps)}/{len(gaps)}")
        
        # Find extreme cases
        max_gap_model = max(gap_data, key=lambda x: x['gap'])
        min_gap_model = min(gap_data, key=lambda x: x['gap'])
        
        print(f"   • Largest TC advantage: {max_gap_model['model_label']} ({max_gap_model['gap']:.3f})")
        print(f"   • Largest WO advantage: {min_gap_model['model_label']} ({min_gap_model['gap']:.3f})")
        
        # Accuracy vs CER correlation analysis
        accuracies = [d['overall_accuracy'] for d in gap_data]
        cers = [d['cer'] for d in gap_data]
        
        if len(set(cers)) > 1:  # Check if we have CER variation
            correlation = np.corrcoef(accuracies, cers)[0, 1]
            print(f"   • Accuracy-CER correlation: {correlation:.3f}")
            
        # Note about axis orientation
        print(f"\n📝 NOTE: Both charts now flow from worst (lower-left) to best (upper-right):")
        print(f"         Left: Low accuracy + high gap → High accuracy + low gap")
        print(f"         Right: High CER + high gap → Low CER + low gap")
        print(f"         Lower gaps are better (smaller performance difference between TC and WO extraction).")
    
    return fig, gap_data


# ====================
# CER BOX PLOT COMPARISON FUNCTIONS
# ====================

def prepare_model_cer_boxplot_data(comprehensive_dataset):
    """Prepare CER data for box plot comparison."""
    model_cer_data = {}
    model_details = {}
    
    # Process all model types
    for model_type in ['pixtral', 'llama', 'doctr']:
        if model_type in comprehensive_dataset['model_data']:
            experiments = comprehensive_dataset['model_data'][model_type]
            
            for experiment in experiments:
                if 'extracted_data' in experiment:
                    # Get model identifier
                    model_name = model_type.title()
                    if model_type in ['pixtral', 'llama']:
                        # Add prompt type for LMM models
                        if 'metadata' in experiment and 'prompt_info' in experiment['metadata']:
                            prompt_type = experiment['metadata']['prompt_info'].get('prompt_type', 'unknown')
                            model_label = f"{model_name}-{prompt_type}"
                        else:
                            model_label = model_name
                    else:
                        # OCR model
                        test_id = experiment.get('metadata', {}).get('test_id', 'unknown')
                        model_label = f"docTR-{test_id}"
                    
                    # Extract CER values from individual results
                    cer_values = []
                    for result in experiment['extracted_data']:
                        if 'performance' in result:
                            perf = result['performance']
                            # Collect work order CER (use as combined CER)
                            if 'work_order_cer' in perf:
                                cer_values.append(perf['work_order_cer'])
                    
                    if cer_values:
                        if model_label not in model_cer_data:
                            model_cer_data[model_label] = []
                            model_details[model_label] = {
                                'model_type': model_type,
                                'base_model': model_name
                            }
                        
                        model_cer_data[model_label].extend(cer_values)
    
    return model_cer_data, model_details


def plot_model_cer_comparison_boxplot(model_cer_data, model_details):
    """Create box plot comparison of CER across all models."""
    
    if not model_cer_data:
        print("No CER data available for box plot comparison")
        return plt.figure()
    
    # Calculate median CER for each model for sorting
    model_medians = {}
    for model, cer_values in model_cer_data.items():
        if cer_values:
            model_medians[model] = np.median(cer_values)
    
    # Sort models by median CER (ascending - lower is better)
    sorted_models = sorted(model_medians.keys(), key=lambda x: model_medians[x])
    
    # Prepare data for box plot
    sorted_cer_data = [model_cer_data[model] for model in sorted_models]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Create box plot
    box_plot = ax.boxplot(sorted_cer_data, labels=sorted_models, patch_artist=True)
    
    # Color boxes based on model type
    for i, model in enumerate(sorted_models):
        model_type = model_details[model]['model_type']
        if model_type == 'pixtral':
            color = ANALYSIS_COLORS['Pixtral']
        elif model_type == 'llama':
            color = ANALYSIS_COLORS['Llama']
        elif model_type == 'doctr':
            color = ANALYSIS_COLORS['DocTR']
        else:
            color = ANALYSIS_COLORS['baseline']
        
        box_plot['boxes'][i].set_facecolor(color)
        box_plot['boxes'][i].set_alpha(0.7)
    
    # Add performance threshold lines
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (CER < 0.1)')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Good (CER < 0.2)')
    ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fair (CER < 0.4)')
    
    # Styling
    ax.set_title('Model CER Comparison: All Fields Combined\n(Sorted by Median CER - Lower is Better)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Character Error Rate (CER)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits
    all_values = [val for values in sorted_cer_data for val in values]
    if all_values:
        y_max = max(all_values) * 1.1
        ax.set_ylim(0, min(y_max, 1.0))  # Cap at 1.0 for reasonable scale
    
    plt.tight_layout()
    return fig


# ====================
# CER BOX PLOT GENERATOR FUNCTION
# ====================

def generate_model_cer_boxplot_analysis(comprehensive_dataset):
    """Generate complete model CER box plot analysis."""
    print("🔍 Creating Model CER Box Plot Comparison...")
    
    model_cer_data, model_details = prepare_model_cer_boxplot_data(comprehensive_dataset)
    fig = plot_model_cer_comparison_boxplot(model_cer_data, model_details)
    
    # Analysis summary
    if model_cer_data:
        print(f"\n📊 CER BOX PLOT ANALYSIS:")
        print(f"   • Model configurations analyzed: {len(model_cer_data)}")
        
        # Calculate statistics for each model
        model_stats = {}
        for model, cer_values in model_cer_data.items():
            if cer_values:
                model_stats[model] = {
                    'median': np.median(cer_values),
                    'mean': np.mean(cer_values),
                    'std': np.std(cer_values),
                    'min': min(cer_values),
                    'max': max(cer_values),
                    'count': len(cer_values)
                }
        
        # Find best and worst performers
        if model_stats:
            best_model = min(model_stats.items(), key=lambda x: x[1]['median'])
            worst_model = max(model_stats.items(), key=lambda x: x[1]['median'])
            
            print(f"   • Best performer (lowest median CER): {best_model[0]} ({best_model[1]['median']:.3f})")
            print(f"   • Worst performer (highest median CER): {worst_model[0]} ({worst_model[1]['median']:.3f})")
            
            # Performance threshold analysis
            excellent_models = sum(1 for stats in model_stats.values() if stats['median'] < 0.1)
            good_models = sum(1 for stats in model_stats.values() if 0.1 <= stats['median'] < 0.2)
            fair_models = sum(1 for stats in model_stats.values() if 0.2 <= stats['median'] < 0.4)
            poor_models = sum(1 for stats in model_stats.values() if stats['median'] >= 0.4)
            
            print(f"   • Performance distribution:")
            print(f"     - Excellent (CER < 0.1): {excellent_models} models")
            print(f"     - Good (0.1 ≤ CER < 0.2): {good_models} models")
            print(f"     - Fair (0.2 ≤ CER < 0.4): {fair_models} models")
            print(f"     - Poor (CER ≥ 0.4): {poor_models} models")
            
            # Overall statistics
            all_medians = [stats['median'] for stats in model_stats.values()]
            print(f"   • Overall median CER across all models: {np.median(all_medians):.3f}")
            print(f"   • CER range: {min(all_medians):.3f} - {max(all_medians):.3f}")
    
    return fig, model_cer_data