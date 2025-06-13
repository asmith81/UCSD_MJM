# Initial Large Multimodal Model (LMM) Comparison Analysis

**Date:** December 2024  
**Models Tested:** Pixtral-12B vs Llama-3.2-11B-Vision  
**Task:** Invoice Work Order Number & Total Cost Extraction  
**Dataset:** 50 invoice images  

## Executive Summary

This analysis compares the performance of two leading vision-language models on invoice data extraction across four different prompting strategies. **Llama + Step-by-Step prompting emerged as the clear winner** with 94% total cost accuracy.

## Model & Prompt Performance Rankings

### 🥇 **TOP PERFORMER: Llama + Prompt 4 (Step-by-Step)**
- **Work Order Accuracy:** 74% (highest for Llama)  
- **Total Cost Accuracy:** 94% (highest overall)
- **Average CER:** 0.14 (Character Error Rate)
- **JSON Extraction Success:** 100%

### 🥈 **SECOND PLACE: Llama + Prompt 1 (Basic Extraction)**
- **Work Order Accuracy:** 66%
- **Total Cost Accuracy:** 92% (second highest)
- **Average CER:** 0.24
- **JSON Extraction Success:** 100%

### 🥉 **THIRD PLACE: Pixtral + Multiple Prompts (tied)**
- **Work Order Accuracy:** 70% (highest for Pixtral)
- **Total Cost Accuracy:** 82-84%
- **Average CER:** 0.08-0.10
- **JSON Extraction Success:** 98-100%

## Detailed Performance Breakdown

### Llama-3.2-11B-Vision Results

| Prompt Type | Work Order Acc | Total Cost Acc | Avg CER | JSON Success |
|-------------|----------------|----------------|---------|--------------|
| **Step-by-Step** | **74%** | **94%** | 0.14 | 100% |
| Basic | 66% | 92% | 0.24 | 100% |
| Locational | 70% | 86% | 0.19 | 100% |
| Detailed | 54% | 86% | 0.37 | 100% |

### Pixtral-12B Results

| Prompt Type | Work Order Acc | Total Cost Acc | Avg CER | JSON Success |
|-------------|----------------|----------------|---------|--------------|
| Locational | **70%** | 84% | 0.09 | 100% |
| Detailed | **70%** | 82% | 0.08 | 100% |
| Basic | **70%** | 84% | 0.10 | 100% |
| Step-by-Step | 66% | 80% | 0.07 | 98% |

## Key Findings

### Model Strengths & Weaknesses

**Llama-3.2-11B-Vision:**
- ✅ **Excellent at total cost extraction** (86-94% accuracy across all prompts)
- ✅ **Responds well to step-by-step guidance** (+20% improvement vs detailed prompting)
- ✅ **Perfect JSON formatting** across all tests
- ❌ **Sensitive to prompt complexity** (detailed prompting hurts performance significantly)
- ❌ **Lower work order accuracy ceiling** (74% max)

**Pixtral-12B:**
- ✅ **Consistent performance** across different prompt types
- ✅ **Lower character error rates** (0.07-0.10 vs 0.14-0.37)
- ✅ **Better work order extraction** (70% consistent)
- ❌ **Lower peak total cost accuracy** (84% max)
- ❌ **Doesn't benefit from step-by-step prompting** (unlike Llama)

### Prompt Strategy Analysis

1. **Step-by-Step Prompting:**
   - Works exceptionally well for Llama (+8-20% improvement)
   - No significant benefit for Pixtral
   - Best overall performance when matched with right model

2. **Basic Extraction Prompting:**
   - Solid baseline performance for both models
   - Good fallback option
   - Consistent JSON formatting

3. **Locational Prompting:**
   - Works well for Pixtral (70% work order accuracy)
   - Moderate performance for Llama
   - Useful when invoice layouts are consistent

4. **Detailed Prompting:**
   - **Harmful for Llama** (drops to 54% work order accuracy)
   - Neutral to slightly positive for Pixtral
   - Avoid for Llama-based systems

## Horizontal Error Trends Across Images

### Consistently Problematic Images

**Image 21084:** Both models misread as "210.84" (decimal confusion)
**Images 21081/21082:** Date format confusion ("2108/" extracted instead of work order)
**Image 21060:** Persistent $1,206 vs $1,200 total cost error
**Image 21054:** Address extraction ("2210 Adams place") instead of work order number

### Error Pattern Categories

#### Work Order Number Errors:
- **Date Confusion:** 15% of errors (mixing work orders with invoice dates)
- **Decimal Insertion:** 10% of errors (adding spurious decimal points)
- **Field Misidentification:** 20% of errors (extracting addresses, company names)

#### Total Cost Errors:
- **Minor Amount Discrepancies:** $6 differences on specific invoices
- **Currency Formatting:** Inconsistent dollar sign handling
- **Overall:** Much more reliable than work order extraction

### Quality Indicators:
- Lower-numbered invoice series (210xx) showed higher error rates
- Suggests potential template or image quality variations
- Invoice format standardization could improve accuracy

## Recommendations

### For Production Implementation:

1. **Primary Recommendation:** Use **Llama-3.2-11B-Vision with Step-by-Step prompting**
   - 94% total cost accuracy is production-ready
   - 74% work order accuracy acceptable with human review

2. **Fallback Option:** **Pixtral-12B with Locational prompting**
   - More consistent performance across prompt variations
   - Better for environments where prompt engineering is limited

3. **Hybrid Approach:** Consider ensemble methods
   - Use both models and compare results
   - Flag discrepancies for human review

### For Future Testing:

1. **Focus on problematic image patterns** identified in horizontal analysis
2. **Test prompt variations** specifically for work order number extraction
3. **Investigate image preprocessing** to address quality issues in 210xx series
4. **Consider fine-tuning** on invoice-specific datasets

## Technical Details

**Test Configuration:**
- 50 invoice images from construction/contracting domain
- 4 prompt strategies per model (8 total configurations)
- Metrics: Work Order Accuracy, Total Cost Accuracy, Character Error Rate
- JSON extraction success rate tracked separately

**Models:**
- Pixtral-12B (bfloat16 quantization)
- Llama-3.2-11B-Vision (torch_dtype optimization)

**Evaluation Method:**
- Ground truth comparison from manually verified labels
- Character-level error analysis for partial matches
- Categorical error classification for pattern analysis

---

*This analysis provides the foundation for model selection in the invoice processing pipeline. The clear winner (Llama + Step-by-Step) offers production-ready accuracy for the critical total cost extraction task.* 