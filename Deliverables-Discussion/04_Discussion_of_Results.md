# Applied AI/ML OCR Approaches for Construction Industry Invoice Data Extraction - Results Analysis

This analysis framework focuses on understanding the experimental results from the construction invoice processing study, incorporating controlled experimental design considerations and practical system improvement insights.

The primary variable tested was the performance of open source LMM models vs open source OCR software. Additional consideration is given to the performance of Self-Attention (Llama) vs cross-attention (Pixtral) models. Different prompting strategies were incorporated for consideration.

The test data comes from a real-world general contractor in the Washington, D.C. area. This selection of the 50 images in the test set was specifically curated for their consistency in format and photo quality. Two numeric data fields were chosen for extraction mostly because ground-truth data was readily available for these fields in the contractor's digital records. While the "total cost" data field was processed as a numeric data type the "Work Order Number" data type was processed as a string allowing as it could potentially include alpha-numeric characters.

Accuracy was used as the primary metric for analysis and evaluated as a binary outcome for each data field extracted from each image during the course of a single test-run. (Basically if the model returned the ground truth data 40 times in the 50 image test the accuracy would be 80%). Evaluating the work order number as a string also enabled character error rate to be calculated on the extraction of that data. Where applicable CER is used to amplify and further investigate the results of these trials. It should be noted that this measure was only performed on the work order number data field.

This analysis begins with a focus on the interactions of model type, prompt type, and field type on the accuracy of the results produced. It then looks at measures of spread in the results as indicative of model consistency and expectations of performance at scale. The analysis next focuses on the efficiency frontier presented by these models to better understand the trade-off between processing time and accuracy. Finally, a horizontal analysis of each individual image is presented to investigate how characteristics of the images themselves might be impacting model performance.

## Broad Conclusions Up Front

* **LMMs outperform OCR** - Large Multimodal Models significantly outperform traditional OCR approaches
* **The combination of the Llama model with step-by-step prompting yielded the best accuracy**, especially with the numeric field "Total Cost"
* **The Pixtral model both with basic, detailed, and step-by-step prompting provided the most consistent results** and the most computationally effective results
* **There were a few "sticky" images that seemed hard for all the models.** Looking at these images it likely was a handwriting issue.

## 1. Comprehensive Performance Overview

### 1.1 LMM vs OCR Performance Comparison

The comprehensive performance grid analysis reveals significant performance differences between Large Multimodal Models (LMMs) and traditional OCR approaches:

**Overall Accuracy Results:**
- **LMM Models: 76.9% average accuracy**
- **OCR (DocTR): 43.4% average accuracy**
- **Performance Gap: +33.5 percentage points in favor of LMMs**

**Character Error Rate Results:**
- **LMM Models: 15.0% average CER**
- **OCR (DocTR): 51.5% average CER**
- **LMM Advantage: -36.5 percentage points lower error rate**

### 1.2 Individual Model Performance

**Individual Model Accuracy Rankings:**
- **Llama: 77.8% accuracy** (highest overall)
- **Pixtral: 76.1% accuracy**
- **DocTR (OCR): 43.4% accuracy**

**Character Error Rate Rankings:**
- **Pixtral: 12.9% CER** (lowest error rate)
- **Llama: 17.1% CER**
- **DocTR (OCR): 51.5% CER**

**Key Insight:** While Llama achieves the highest accuracy, Pixtral demonstrates superior consistency with the lowest character error rate.

### 1.3 Industry Standard Assessment

**Industry Standard Threshold: 85% accuracy**
- ❌ **No models currently meet the 85% industry standard**
- **Closest performer:** Llama at 77.8% (7.2 percentage points below standard)
- **Improvement needed:** All models require significant enhancement for production deployment

## 2. Prompt Strategy Analysis

### 2.1 Prompt Type Performance Results

Four different prompt types were compared across both the Pixtral and Llama LMMs:

**Accuracy Performance (ranked highest to lowest):**
1. **Step By Step: 78.5%** - Best overall accuracy
2. **Basic Extraction: 78.0%** - Close second
3. **Locational: 77.5%** - Solid performance
4. **Detailed: 73.0%** - Lowest accuracy

**Character Error Rate Performance (ranked lowest to highest):**
1. **Step By Step: 10.6%** - Best CER performance
2. **Locational: 14.3%** - Good performance
3. **Basic Extraction: 17.2%** - Moderate performance
4. **Detailed: 22.7%** - Highest error rate

### 2.2 Prompt Strategy Insights

**Key Finding:** Most prompts performed equally well overall reaching close to 80%. The step-by-step prompt clearly outperformed the other prompt types in both accuracy and Character Error Rate (CER).

**Strategic Implications:**
- **Step-by-step prompting** provides the most reliable results across both accuracy metrics
- **Detailed prompts paradoxically performed worst**, suggesting that overly complex instructions may confuse the models
- **Simple, structured prompts** (Basic Extraction, Locational) performed competitively

## 3. Field-Specific Performance Analysis

### 3.1 Work Order vs Total Cost Extraction

The analysis reveals significant performance differences between the two extraction fields:

**Field-Specific Accuracy Results:**
- **Total Cost: 86.0%** ✅ **Exceeds 85% industry standard**
- **Work Order: 67.5%** ❌ **Below industry standard**
- **Performance Gap: 18.5 percentage points**

### 3.2 Field Performance Insights

**Total Cost Field:**
- Successfully meets industry standards
- More consistent extraction across models
- Numeric nature may contribute to better performance

**Work Order Field:**
- Significant performance challenge
- Alphanumeric complexity creates extraction difficulties
- Requires targeted improvement strategies
- Notable performance gap indicates field-specific optimization needed

**Strategic Implications:**
- **Total Cost extraction is production-ready** for models meeting accuracy thresholds
- **Work Order extraction requires significant improvement** before production deployment
- **Field-specific training** could address performance disparities

## 4. Model Consistency and Reliability Analysis

### 4.1 Accuracy Consistency Rankings

**Model Consistency Analysis (by performance range):**
- **Most Consistent: Pixtral** (range: 0.040) - Highly reliable results
- **Moderate Consistency: Llama** (moderate range variation)
- **Least Consistent: DocTR** (range: 0.160) - High variability

### 4.2 Character Error Rate Consistency

**CER Consistency Rankings:**
- **Most Consistent: Pixtral** (CER range: 0.030) - Highly predictable error rates
- **Least Consistent: Llama** (CER range: 0.228) - High CER variability

### 4.3 Consistency vs Performance Trade-offs

**Key Findings:**
- **Llama has a higher max accuracy than Pixtral but Pixtral is more consistent.** It has a smaller range of results well clustered around its mean accuracy.
- **Llama actually performs the worst in terms of CER spread** despite having good overall accuracy
- **Pixtral offers the best balance** of consistency and performance

**Implications for Production:**
- **Pixtral provides more predictable results** for production environments
- **Llama's high variability** may create operational challenges
- **Consistency is crucial** for scaling to production volumes

## 5. Model Performance Distributions

### 5.1 Statistical Performance Summary

**Pixtral Models:**
- **Mean Accuracy: 75.8%**
- **Standard Deviation: 1.6%** (highly consistent)
- **Range: 73.0% - 77.0%**
- **Meets 85% Standard: ❌ No**

**Llama Models:**
- **Mean Accuracy: 77.8%**
- **Standard Deviation: 5.0%** (more variable)
- **Range: 70.0% - 84.0%**
- **Meets 85% Standard: ❌ No** (though one configuration reached 84%)

**DocTR Models:**
- **Mean Accuracy: 43.4%**
- **Standard Deviation: 5.2%**
- **Range: 35.0% - 51.0%**
- **Meets 85% Standard: ❌ No**

## 6. Efficiency Frontier Analysis

### 6.1 Processing Time vs Accuracy Trade-offs

**Performance and Timing Results:**
- **Llama-step_by_step:** 84.0% accuracy, 5.82s processing time (closest to industry standard)
- **Pixtral models:** ~77% accuracy, ~1.5s processing time (most efficient)
- **DocTR models:** ~45% accuracy, ~0.6s processing time (fastest but least accurate)

### 6.2 Efficiency Insights

**Key Trade-off Analysis:** The tradeoff here is clear. Llama step-by-step comes the closest to meeting industry standards in accuracy however it takes about 5 seconds per trial - twice that of the Pixtral model. Improvements to the Pixtral accuracy with training could result in large compute-time savings.

**Strategic Implications:**
- **For maximum accuracy:** Use Llama-step_by_step (but accept longer processing times)
- **For production efficiency:** Pixtral provides the best accuracy/time ratio
- **For future optimization:** Focus on improving Pixtral accuracy while maintaining speed advantage

## 7. Image-Specific Performance Analysis

### 7.1 Horizontal Analysis Results

**Best Performing Images (Total Cost):**
- **10 images achieved 100% accuracy** across all 9 model combinations
- These images represent optimal characteristics for extraction

**Most Challenging Images:**
- **Several "sticky" images** consistently failed across multiple models
- Performance ranges from 55.6% to 66.7% for most difficult images

### 7.2 Error Pattern Insights

**Identified Challenges:** There are a few images that can't be solved by any model-prompt combination. Further study could examine these images for the characteristics that are common across them. This information can inform future pre-processing efforts.

**Root Cause Analysis:** A cursory examination shows **handwriting and strike-throughs** as the most likely underlying cause of the "sticky" images. A more thorough investigation could help with pre-processing and flagging these images or providing updated prompting to the model to improve performance.

## 8. Error Category Analysis

### 8.1 Work Order Error Patterns

**Overall Work Order Statistics:**
- **Exact Match: 53.5%** of all predictions
- **Partial Match: 39.7%** of predictions
- **Completely Wrong: 4.4%** of predictions
- **Date Confusion: 2.1%** of predictions

**Best Performing Models (Work Order):**
- **Llama-step_by_step: 74.0%** exact matches
- **Llama-locational: 70.0%** exact matches
- **Pixtral models: ~70%** exact matches

### 8.2 Total Cost Error Patterns

**Total Cost Performance:**
- **Numeric Match: 81.1%** of all predictions
- **Completely Wrong: 10.9%** of predictions
- **No Extraction: 4.2%** of predictions

**Best Performing Models (Total Cost):**
- **Llama-step_by_step: 94.0%** numeric matches
- **Llama-basic_extraction: 92.0%** numeric matches
- **Pixtral models: 80-84%** numeric matches

### 8.3 Error Pattern Strategic Insights

**Work Order Challenges:**
- **Partial matches dominate errors** - indicates models are close but not exact
- **Llama step-by-step gets the most exactly right** while pixtral-detailed gets the least totally wrong and limits their errors to partial matches
- **Could be solved with training** - error patterns suggest improvement potential

**Total Cost Advantages:**
- **Higher overall success rates** compared to work order extraction
- **Clear numeric format** contributes to better performance
- **LMM models significantly outperform OCR** on numeric extraction

## 9. Key Strategic Recommendations

### 9.1 Model Selection Recommendations

**For Production Deployment:**
- **Primary Choice: Llama with step-by-step prompting** for maximum accuracy
- **Alternative Choice: Pixtral models** for better consistency and efficiency
- **Avoid: DocTR/OCR approaches** due to significantly lower performance

### 9.2 Field-Specific Strategies

**Total Cost Extraction:**
- **Ready for production** with appropriate model selection
- Focus on **Llama-step_by_step** for best results
- Consider **Pixtral for high-volume processing** with acceptable accuracy trade-offs

**Work Order Extraction:**
- **Requires significant improvement** before production deployment
- **Implement field-specific training** to address performance gaps
- **Consider hybrid approaches** combining multiple model predictions

### 9.3 Preprocessing and Quality Improvements

**Image Quality Enhancement:**
- **Identify and preprocess handwritten elements** to improve accuracy
- **Flag images with strike-throughs** for special handling
- **Implement quality scoring** for incoming images

**Model Optimization:**
- **Focus training efforts on Pixtral** to improve accuracy while maintaining speed
- **Develop field-specific prompting strategies** for work order extraction
- **Implement ensemble approaches** for critical applications

### 9.4 Production Considerations

**Accuracy Thresholds:**
- **No current models meet 85% industry standard** for overall performance
- **Total Cost field meets standards** with appropriate model selection
- **Work Order field requires improvement** before production deployment

**Performance Monitoring:**
- **Implement consistency monitoring** for production systems
- **Track field-specific performance** separately
- **Monitor for "sticky" image patterns** in production data

This comprehensive analysis provides the foundation for strategic decision-making regarding the deployment and optimization of AI/ML approaches for construction industry invoice data extraction.
