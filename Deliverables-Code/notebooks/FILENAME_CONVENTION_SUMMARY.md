# Filename Convention Implementation Summary

## Overview
All three model notebooks now use the new naming convention: `results-{model_name}-{quant_level}-{id_number}.json`

## Implementation Details

### 1. Pixtral Model (`03_pixtral_model.py`)
- **Model name**: `pixtral`
- **Quantization levels**: `bfloat16`, `int8`, `int4`
- **Example filenames**:
  - `results-pixtral-bfloat16-1.json`
  - `results-pixtral-int8-2.json`
  - `results-pixtral-int4-1.json`

### 2. Llama Model (`04_llama_model.py`)
- **Model name**: `llama`
- **Quantization levels**: `bfloat16`, `int8`, `int4`
- **Example filenames**:
  - `results-llama-bfloat16-1.json`
  - `results-llama-int8-3.json`
  - `results-llama-int4-2.json`

### 3. docTR Model (`05_doctr_model.py`)
- **Model name**: `doctr`
- **Quantization level**: `none` (OCR model doesn't use quantization)
- **Model configuration**: User selects detection and recognition architectures
- **Available detection models**: 
  - `db_resnet50` (default)
  - `db_mobilenet_v3_large`
  - `linknet_resnet18/34/50`
- **Available recognition models**:
  - `crnn_vgg16_bn` (default)
  - `crnn_mobilenet_v3_small/large`
  - `master`, `sar_resnet31`
  - `vitstr_small/base`
- **Example filenames**:
  - `results-doctr-none-1.json`
  - `results-doctr-none-2.json`
  - `results-doctr-none-3.json`
- **Model ID structure**:
  ```json
  "model_id": {
    "detection_model": "db_resnet50",
    "recognition_model": "crnn_vgg16_bn", 
    "combined": "db_resnet50+crnn_vgg16_bn"
  }
  ```

## Key Features

### Auto-Incrementing Counter
- Automatically finds the highest existing counter for each model+quantization combination
- Increments by 1 for the next test run
- Prevents filename collisions
- Handles gaps in numbering (e.g., if file 2 is deleted, next will be 4, not 2)

### Consistent Structure
Each results file contains:
- **Metadata**: Test ID, model info, prompt info, processing config
- **Results**: Raw model outputs with processing times and error handling
- **File naming**: Clear identification of model and configuration used

### Example Usage
```python
# The generate_results_filename function is called automatically
test_id, filepath = generate_results_filename("pixtral", "bfloat16", results_dir)
# Returns: ("results-pixtral-bfloat16-3", "/path/to/results-pixtral-bfloat16-3.json")
```

## Benefits
1. **Clear identification**: Easy to see what model and configuration generated each file
2. **Chronological ordering**: Counter shows the sequence of test runs
3. **No collisions**: Auto-incrementing prevents overwriting existing files
4. **Consistent format**: Same pattern across all three model types
5. **Analysis-friendly**: Easy to filter and group results by model or quantization

## Migration
- Old format: `test_results_20240101_123456.json`
- New format: `results-pixtral-bfloat16-1.json`
- All notebooks updated to use new convention
- Existing files with old naming will not be affected 