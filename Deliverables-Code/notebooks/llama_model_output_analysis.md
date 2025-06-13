# Llama Vision Model Output Analysis

## Overview
This document analyzes the unexpected behavior observed in the Llama-3.2-11B-Vision model's raw outputs during invoice data extraction tasks and explains how these issues are being addressed through post-processing.

## Observed Issues

### 1. Prompt Echo Behavior
**Issue**: The model consistently echoes the input prompt in its response.

**Example**:
```
"Please extract the following information from this invoice:
1. Work Order Number
2. Total Cost

Return the information in JSON format with these exact keys:
{
  "work_order_number": "extracted value",
  "total_cost": "extracted value"
}

<|response|>
<|end_of_response|> {
  "work_order_number": "21089",
  "total_cost": "200"
} | ..."
```

**Possible Causes**:
- Llama Vision's chat template format may be causing input repetition
- The special tokens (`<|response|>`, `<|end_of_response|>`) may not be properly controlling output generation
- The model architecture may naturally include input context in its response generation

### 2. Repetitive Text Patterns
**Issue**: The model generates highly repetitive sequences after providing the correct answer.

**Pattern Observed**:
```
} | 1. Work Order Number: 21089
2. Total Cost: 200 | 1. Work Order Number: 21089
2. Total Cost: 200 | 1. Work Order Number: 21089
2. Total Cost: 200 | ...
```

**Possible Causes**:
- **Low Temperature Setting**: Temperature of 0.1 may be too restrictive, causing the model to fall into repetitive loops
- **Sampling Configuration**: `top_k=1` and `top_p=0.1` create very deterministic outputs that may trigger repetition
- **Max Token Limit**: The model may be filling the 256 max_new_tokens limit with repetitive content
- **Training Data Patterns**: The model may have learned repetitive patterns from training data

### 3. OCR-like Text Injection
**Issue**: Some responses include OCR-style text mixed with the structured output.

**Example**:
```
}. <OCR/> ESTIMATE Nombre detu Fecha: 11-1824 24 4809 49 MD 21086 86 Due Date: Payable to: del 3318 Estimate for: 500 20036 360 pies de de de...
```

**Possible Causes**:
- The model may be attempting to transcribe visible text from the image
- Vision-language model training may include OCR-style tasks that bleed into structured extraction
- The model may be confused between extraction and transcription tasks

### 4. Special Token Handling Issues
**Issue**: Special tokens appear in the raw output instead of being processed internally.

**Tokens Observed**:
- `<|response|>`
- `<|end_of_response|>`
- `<|image|>`
- `<|begin_of_text|>`
- `<|end_of_text|>`

**Possible Causes**:
- Incorrect tokenizer configuration for the Llama Vision model
- Special tokens not properly configured in the processor
- Model fine-tuning may not have properly learned special token behavior

## Mitigation Strategies

### Current Post-Processing Approach
The analysis notebook implements robust JSON extraction that handles these issues:

```python
def extract_json_from_response(raw_response: str) -> dict:
    """Extract JSON data from the raw response string."""
    # Find all JSON objects in the response
    json_objects = []
    # ... (implementation searches for valid JSON patterns)
    
    # The second JSON object should be the actual results
    # (first one is often the prompt template)
    if len(json_objects) >= 2:
        return json_objects[1]
    elif len(json_objects) == 1:
        return json_objects[0]
```

This approach:
1. **Ignores repetitive text** by focusing only on valid JSON structures
2. **Handles multiple JSON objects** by selecting the most relevant one
3. **Robust parsing** that continues to work despite formatting issues

### Recommended Configuration Changes

#### 1. Inference Parameters
Consider adjusting these settings to reduce repetition:
- **Temperature**: Increase from 0.1 to 0.3-0.5 for more diverse outputs
- **Top-K/Top-P**: Relax constraints (top_k=50, top_p=0.9) to prevent repetitive loops
- **Max New Tokens**: Reduce from 256 to 100-150 to prevent filler text

#### 2. Prompt Engineering
- **Simplify special tokens**: Remove complex token sequences that may confuse the model
- **Add stop sequences**: Include explicit stop tokens to prevent over-generation
- **Single-shot format**: Use simpler prompt format without complex chat templates

#### 3. Model Configuration
- **Flash Attention**: The current Flash Attention setup may contribute to these patterns
- **Quantization**: The bfloat16 quantization appears to be working well
- **Device Mapping**: Current auto mapping seems appropriate

## Performance Impact

Despite the output formatting issues, the model demonstrates:
- **High accuracy** in data extraction (work order numbers and costs are correctly identified)
- **Consistent JSON structure** in the core response
- **Reliable processing** across all test images

The post-processing approach successfully extracts the required information without requiring model reconfiguration, making it a pragmatic solution for production use.

## Recommendations for Future Improvements

1. **Experiment with prompt formats** to reduce echo behavior
2. **Test different sampling parameters** to minimize repetition
3. **Implement custom stopping criteria** in the generation process
4. **Consider fine-tuning** on invoice-specific data with proper formatting
5. **Evaluate alternative vision-language models** for comparison

## Conclusion

While the Llama Vision model exhibits some unexpected output formatting behaviors, the core extraction capability remains strong. The implemented post-processing approach provides a robust solution that maintains high accuracy while handling the model's quirks effectively. This demonstrates the importance of building resilient data processing pipelines that can adapt to model-specific behaviors. 