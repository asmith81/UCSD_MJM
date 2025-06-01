# Invoice Image Processing Pipeline

## Overview
This pipeline processes and tracks a collection of contractor invoice images, preparing them for future machine learning model training. The system handles invoice digitization from initial raw images through standardization and quality enhancement.

## Process Description
1. Raw images are stored in the initial directory
2. Metadata is generated for all images
3. Standard format invoices are manually identified and moved
4. Remaining images are processed for orientation and quality
5. All steps are tracked in metadata and logs

## Inputs & Outputs
### Inputs
- Raw invoice images (.jpg, .jpeg, .png)
- Files should be named with 4-digit invoice numbers (e.g., "1234.jpg")
- Mixed formats expected (handwritten forms, typed documents, multi-page files)

### Outputs
- Processed standard format images in standardized directory
- Metadata CSV tracking all files and their processing status
- Detailed processing logs
- Enhanced images corrected for orientation and quality

## Project Structure
```
project_root/
├── data/
│   ├── 0_raw/          # Original downloaded images
│   └── 1_standardized/ # Standard form images
├── logs/               # Processing logs
├── metadata/          # Metadata CSV files
├── notebooks/         # Processing notebooks
└── src/              # Source code
```

## Processing Workflow

### 1. Initial Metadata Generation
Run `1_metadata_generation.ipynb`:
- Scans all files in `data/0_raw`
- Creates initial metadata records
- Saves to `metadata/images_metadata.csv`
- Logs process in `logs/metadata_generator.log`

### 2. Manual Curation
Important manual step:
- Review images in `data/0_raw`
- Identify standard format handwritten forms
- Move selected files to `data/1_standardized`
- Leave non-standard formats in raw directory
- No metadata updates needed at this stage

### 3. Standard Form Processing
Run `2_image_processing.ipynb`:
- Processes images in `data/1_standardized`
- Analyzes dimensions and orientation
- Applies image corrections as needed
- Updates metadata CSV with processing status
- Logs process in `logs/standardized_processor.log`

## Understanding the Outputs

### Metadata CSV
The `images_metadata.csv` tracks:
- File locations and movement
- Processing status
- Image properties and corrections applied
- Standard vs non-standard classification

Fields include:
- filename_original: Original file name
- invoice_number: Extracted from filename
- is_standard_form: Manual classification status
- original_orientation: Current image orientation
- current_phase: Processing stage
- auto_processing_notes: Quality check results

Note: The `original_orientation` field is currently not accurately tracking image orientation. This will be corrected in future versions.

### Log Files
- `metadata_generator.log`: Records file discovery and initial metadata creation
- `standardized_processor.log`: Records image processing details including:
  - Dimension analysis
  - Quality checks
  - Applied corrections
  - Processing errors

## Known Issues
1. Orientation tracking not fully accurate
2. Some multi-page files may need manual handling
3. Image quality enhancements still being tuned

## Next Steps
Future versions will include:
- Accurate orientation tracking
- Enhanced image quality processing
- Validation steps for processed images
- Automated format detection