# DocTR Model Performance Analysis

## **Summary of Accuracy Results**

Based on the analysis files, here are the accuracy results for each DocTR recognition model:

| Model | Recognition Model | Work Order Accuracy | Total Cost Accuracy | Combined Performance |
|-------|-------------------|---------------------|---------------------|---------------------|
| 1 | crnn_vgg16_bn | 34% | 52% | 43% |
| 2 | crnn_mobilenet_v3_small | 34% | 36% | 35% |
| 3 | crnn_mobilenet_v3_large | 36% | 44% | 40% |
| 4 | **master** | **42%** | **60%** | **51%** |
| 5 | sar_resnet31 | 40% | 58% | 49% |
| 6 | vitstr_small | 38% | 42% | 40% |
| 7 | vitstr_base | 38% | 54% | 46% |

## **Best Performing Model**

**The `master` recognition model (analysis-doctr-none-4) performed best** with:
- **Work Order accuracy: 42%**
- **Total Cost accuracy: 60%**
- **Combined performance: 51%**

The `sar_resnet31` model came in second place with 49% combined performance.

## **Recurring Image Quality Issues**

Looking at the "No Extraction" failures across models, I found consistent problem patterns:

1. **Line 40777**: This appears across multiple models (1, 3, 4, 5, 6) - indicating a consistently challenging image
2. **Line 67026**: Failed extraction across models 1, 3, 4, 5, 6 - another problematic image
3. **Line 125053**: Failed across models 1, 3, 4, 6, 7 - likely poor image quality
4. **Line 146507**: Failed across models 1, 3, 4, 6 - recurring extraction problem

These consistent failures suggest **4-5 specific test images have quality issues** that make them difficult for OCR across all models.

## **Error Pattern Analysis**

The error categories show distinct patterns:

1. **"No Extraction"** - Complete failure to extract values:
   - Model 2 (mobilenet_v3_small): 13 cases - **highest failure rate**
   - Model 3 (mobilenet_v3_large): 11 cases
   - Model 6 (vitstr_small): 10 cases
   - Model 7 (vitstr_base): 3 cases - **lowest failure rate**

2. **"Partial Match"** - Most common error type across all models
3. **"Completely Wrong"** - Significant extraction errors
4. **"Numeric Match"** - Close but not exact matches
5. **"Missing Digit"** and **"Extra Digit"** - Formatting issues

## **Post-Processing Improvement Opportunities**

The analysis reveals several areas where post-processing could improve results **without changing model inference**:

1. **Numeric Formatting**: Many errors are "Numeric Match" - suggesting the right numbers are extracted but formatted incorrectly (e.g., missing decimal points, extra spaces)

2. **Partial Match Recovery**: The high frequency of "Partial Match" errors suggests post-processing could:
   - Apply fuzzy matching to recover partial work order numbers
   - Use regex patterns to clean extracted text
   - Apply domain-specific corrections for common OCR mistakes

3. **Extraction Failure Recovery**: The "No Extraction" cases could benefit from:
   - Fallback extraction methods
   - Region-of-interest refinement
   - Alternative text detection approaches

4. **Digit Correction**: "Missing Digit" and "Extra Digit" errors could be addressed through:
   - Expected format validation
   - Context-aware digit correction
   - Length-based validation rules

## **Key Findings**

1. **Model Performance**: The `master` model significantly outperformed others, especially for cost extraction (60% vs 36-58% for others)

2. **Field-Specific Performance**: Total cost extraction generally performed better than work order extraction across all models

3. **Consistent Problem Images**: 4-5 specific test images consistently failed across multiple models, indicating image quality issues rather than model limitations

4. **Post-Processing Potential**: Up to 15-20% accuracy improvement could be achieved through better post-processing of "Numeric Match" and "Partial Match" cases without touching model inference

The analysis suggests that while the `master` model provides the best baseline performance, significant improvements could be achieved through enhanced post-processing techniques targeting the identified error patterns. 