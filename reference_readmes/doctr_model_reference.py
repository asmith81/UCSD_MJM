# %% [markdown]
"""
# docTR Model Evaluation Notebook

This notebook demonstrates basic usage of the docTR model for document text recognition.
It focuses on direct model usage with clear logging of outputs.
"""

# %% [markdown]
"""
## Setup and Configuration
### Initial Imports
"""

# %%
# Install dependencies from requirements file
import subprocess
import sys
from pathlib import Path

# Determine root directory and requirements file path
try:
    # When running as a script
    current_file = Path(__file__)
    ROOT_DIR = current_file.parent
except NameError:
    # When running in a notebook
    ROOT_DIR = Path.cwd()

requirements_file = ROOT_DIR / "requirements_doctr.txt"

# Install requirements if file exists
if requirements_file.exists():
    print(f"Installing dependencies from {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)])
else:
    raise FileNotFoundError(f"Requirements file not found at {requirements_file}")

# %%
# Built-in Python modules
import os
import time
import json
from datetime import datetime

# External dependencies
import torch
from PIL import Image

# docTR specific imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# %% [markdown]
"""
## CUDA Availability Check
"""

# %%
def check_cuda_availability() -> bool:
    """
    Check if CUDA is available and log the GPU information.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        print(f"CUDA is available with {gpu_count} GPU(s)")
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("CUDA is not available. Running on CPU mode.")
        return False

# Check CUDA availability
check_cuda_availability()

# %% [markdown]
"""
## Model Initialization
"""

# %%
def initialize_model() -> tuple:
    """
    Initialize docTR OCR model with default configuration.
    
    Returns:
        tuple: (model, device)
    """
    try:
        print("\nInitializing docTR model...")
        
        # Initialize model with default configuration
        model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            resolve_blocks=True
        )
        
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            model.det_predictor.to(device)
            model.reco_predictor.to(device)
            print(f"Model moved to {device}")
        
        print("Model initialized successfully")
        return model, device
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

# Initialize the model
model, device = initialize_model()

# %% [markdown]
"""
## Process Single Image
"""

# %%
def process_single_image(image_path: str) -> None:
    """
    Process a single image using docTR model and display results.
    
    Args:
        image_path (str): Path to the image file
    """
    try:
        print(f"\nProcessing image: {image_path}")
        
        # Load and display the image
        print("Loading and displaying image...")
        image = Image.open(image_path)
        
        # Create a display version of the image with a max size of 800x800
        display_image = image.copy()
        max_display_size = (800, 800)
        display_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
        
        # Display the image
        print("\nInput Image (resized for display):")
        display(display_image)
        
        # Load image using DocumentFile
        print("\nLoading image with DocumentFile...")
        doc = DocumentFile.from_images(image_path)
        print(f"Document loaded successfully")
        
        # Run inference
        print("\nRunning inference...")
        start_time = time.time()
        with torch.no_grad():
            result = model(doc)
        processing_time = time.time() - start_time
        print(f"Inference completed in {processing_time:.2f} seconds")
        
        # Add rendered text output for easy comparison
        print("\n" + "="*50)
        print("RENDERED TEXT OUTPUT")
        print("="*50)
        rendered_text = result.render()
        print(rendered_text)
        print("="*50)
        
        # Store result globally for post-processing
        global last_result
        last_result = result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

# %% [markdown]
"""
## Test with Sample Image
"""

# %%
# Test with a sample image
test_image_path = ROOT_DIR / "data" / "images" / "1017.jpg"
if test_image_path.exists():
    process_single_image(str(test_image_path))
else:
    print(f"Test image not found at {test_image_path}")
    print("Please ensure the image file exists at the specified path.")
# %% [markdown]
"""
## Post-Process Single Image
"""

# %%
def extract_work_order_and_total(result) -> dict:
    """
    Extract work order number and total cost from docTR result using document type classification
    and targeted spatial analysis.
    
    Args:
        result: docTR result object
        
    Returns:
        dict: Extracted data with work_order_number and total_cost
    """
    try:
        # Convert result to JSON for easier processing
        json_result = result.export()
        
        extracted_data = {
            "work_order_number": None,
            "total_cost": None,
            "extraction_confidence": {
                "work_order_found": False,
                "total_cost_found": False,
                "spatial_match": False,
                "document_type": None
            }
        }
        
        def get_block_info(block):
            """Extract block information including text and spatial coordinates."""
            block_words = []
            for line in block['lines']:
                for word in line['words']:
                    block_words.append(word)
            
            if not block_words:
                return None, None, None, None
            
            # Calculate block center point for better spatial comparison
            all_coords = []
            for word in block_words:
                coords = word['geometry']
                all_coords.extend(coords)
            
            if not all_coords:
                return None, None, None, None
            
            # Calculate center point
            center_x = sum(coord[0] for coord in all_coords) / len(all_coords)
            center_y = sum(coord[1] for coord in all_coords) / len(all_coords)
            
            # Get block text
            block_text = ' '.join(word['value'] for word in block_words)
            
            return block_words, block_text, center_x, center_y
        
        def fuzzy_contains(text, target_words, threshold=0.7):
            """Check if text contains target words with OCR error tolerance."""
            text_lower = text.lower()
            
            # Direct substring check first (fastest)
            if all(word.lower() in text_lower for word in target_words):
                return True
            
            # Character substitution tolerance for common OCR errors
            ocr_substitutions = {
                'o': '0', '0': 'o', 'i': '1', '1': 'i', 'l': '1', '1': 'l',
                's': '5', '5': 's', 'g': '9', '9': 'g', 't': 'f', 'f': 't'
            }
            
            # Create variations of target words
            for target in target_words:
                variations = [target.lower()]
                for i, char in enumerate(target.lower()):
                    if char in ocr_substitutions:
                        new_word = target.lower()[:i] + ocr_substitutions[char] + target.lower()[i+1:]
                        variations.append(new_word)
                
                # Check if any variation is found
                found = False
                for var in variations:
                    if var in text_lower:
                        found = True
                        break
                
                if not found:
                    return False
            
            return True
        
        def classify_document_type(all_blocks):
            """Determine if document is Invoice or Estimate based on top content."""
            # Check blocks in the top third of the document
            top_blocks = []
            for block in all_blocks:
                _, block_text, _, center_y = get_block_info(block)
                if block_text and center_y < 0.33:  # Top third
                    top_blocks.append(block_text.lower())
            
            top_content = ' '.join(top_blocks)
            
            # Check for document type indicators
            if fuzzy_contains(top_content, ['invoice']):
                return 'invoice'
            elif fuzzy_contains(top_content, ['estimate']):
                return 'estimate'
            
            return None
        
        def find_primary_key_invoice(all_blocks):
            """Find work order number in invoice documents."""
            for block in all_blocks:
                block_words, block_text, center_x, center_y = get_block_info(block)
                if not block_words:
                    continue
                
                # Look for MJM Work Order Number pattern
                if fuzzy_contains(block_text, ['mjm', 'work', 'order', 'number']) or \
                   fuzzy_contains(block_text, ['mjm', 'order', 'number']):
                    
                    # First check within the same block for numbers
                    for word in block_words:
                        word_text = word['value'].strip()
                        if word_text.isdigit() and 4 <= len(word_text) <= 6:
                            return word_text
                    
                    # Then look for numbers to the right and nearby
                    candidates = []
                    for other_block in all_blocks:
                        other_words, other_text, other_x, other_y = get_block_info(other_block)
                        if not other_words:
                            continue
                        
                        # Calculate distance and position
                        distance = ((other_x - center_x) ** 2 + (other_y - center_y) ** 2) ** 0.5
                        if distance <= 0.2:  # Within reasonable distance
                            for word in other_words:
                                word_text = word['value'].strip()
                                if word_text.isdigit() and 4 <= len(word_text) <= 6:
                                    candidates.append({
                                        'value': word_text,
                                        'distance': distance,
                                        'same_line': abs(other_y - center_y) < 0.05,
                                        'to_right': other_x > center_x
                                    })
                    
                    # Sort by preference: same line and to the right, then by distance
                    candidates.sort(key=lambda x: (
                        not x['same_line'],
                        not x['to_right'],
                        x['distance']
                    ))
                    
                    if candidates:
                        return candidates[0]['value']
            
            return None
        
        def find_primary_key_estimate(all_blocks):
            """Find estimate number in estimate documents."""
            for block in all_blocks:
                block_words, block_text, center_x, center_y = get_block_info(block)
                if not block_words:
                    continue
                
                # Look for Estimate Number pattern
                if fuzzy_contains(block_text, ['estimate', 'number']):
                    
                    # First check within the same block
                    for word in block_words:
                        word_text = word['value'].strip()
                        if word_text.isdigit() and 4 <= len(word_text) <= 6:
                            return word_text
                    
                    # Look for numbers below and nearby
                    candidates = []
                    for other_block in all_blocks:
                        other_words, other_text, other_x, other_y = get_block_info(other_block)
                        if not other_words:
                            continue
                        
                        # Calculate distance and position
                        distance = ((other_x - center_x) ** 2 + (other_y - center_y) ** 2) ** 0.5
                        if distance <= 0.2:  # Within reasonable distance
                            for word in other_words:
                                word_text = word['value'].strip()
                                if word_text.isdigit() and 4 <= len(word_text) <= 6:
                                    candidates.append({
                                        'value': word_text,
                                        'distance': distance,
                                        'below': other_y > center_y,
                                        'nearby_x': abs(other_x - center_x) < 0.1
                                    })
                    
                    # Sort by preference: below and nearby horizontally, then by distance
                    candidates.sort(key=lambda x: (
                        not x['below'],
                        not x['nearby_x'],
                        x['distance']
                    ))
                    
                    if candidates:
                        return candidates[0]['value']
            
            return None
        
        def find_grand_total(all_blocks):
            """Find grand total amount with emphasis on lower portion of document."""
            # First, identify blocks in the lower portion (bottom half)
            lower_blocks = []
            for block in all_blocks:
                _, block_text, center_x, center_y = get_block_info(block)
                if center_y > 0.5:  # Lower half
                    lower_blocks.append((block, center_x, center_y))
            
            # If we have lower blocks, prioritize them
            target_blocks = lower_blocks if lower_blocks else [(block, *get_block_info(block)[2:4]) for block in all_blocks]
            
            for block, center_x, center_y in target_blocks:
                block_words, block_text, _, _ = get_block_info(block)
                if not block_words:
                    continue
                
                # Look for Grand Total pattern
                if fuzzy_contains(block_text, ['grand', 'total']) or \
                   fuzzy_contains(block_text, ['total']):
                    
                    # Check within the same block first
                    monetary_candidates = []
                    for word in block_words:
                        word_text = word['value'].strip()
                        clean_amount = extract_monetary_value(word_text)
                        if clean_amount:
                            monetary_candidates.append({
                                'value': clean_amount,
                                'distance': 0,
                                'same_block': True
                            })
                    
                    # Look for monetary values to the right and nearby
                    for other_block in all_blocks:
                        other_words, other_text, other_x, other_y = get_block_info(other_block)
                        if not other_words or other_block == block:
                            continue
                        
                        distance = ((other_x - center_x) ** 2 + (other_y - center_y) ** 2) ** 0.5
                        if distance <= 0.2:  # Within reasonable distance
                            for word in other_words:
                                word_text = word['value'].strip()
                                clean_amount = extract_monetary_value(word_text)
                                if clean_amount:
                                    monetary_candidates.append({
                                        'value': clean_amount,
                                        'distance': distance,
                                        'same_block': False,
                                        'to_right': other_x > center_x,
                                        'same_line': abs(other_y - center_y) < 0.05
                                    })
                    
                    # Sort by preference
                    monetary_candidates.sort(key=lambda x: (
                        not x.get('same_block', False),
                        not x.get('same_line', False),
                        not x.get('to_right', False),
                        x['distance']
                    ))
                    
                    if monetary_candidates:
                        return monetary_candidates[0]['value']
            
            return None
        
        def extract_monetary_value(text):
            """Extract clean monetary value from text."""
            if not text:
                return None
            
            # Remove common prefixes and clean up
            clean_text = text.replace('$', '').replace(',', '').strip()
            
            try:
                # Try to parse as float
                amount = float(clean_text)
                
                # Reasonable range check (between $10 and $10,000)
                if 10.0 <= amount <= 10000.0:
                    # Format consistently
                    if '.' in clean_text:
                        return f"{amount:.2f}"
                    else:
                        # If no decimal, assume whole dollars
                        return f"{amount:.2f}"
                        
            except ValueError:
                pass
            
            return None
        
        # Main processing logic
        for page in json_result['pages']:
            all_blocks = page['blocks']
            
            # Step 1: Classify document type
            doc_type = classify_document_type(all_blocks)
            extracted_data["extraction_confidence"]["document_type"] = doc_type
            
            if doc_type:
                extracted_data["extraction_confidence"]["spatial_match"] = True
            
            # Step 2: Extract primary key based on document type
            primary_key = None
            if doc_type == 'invoice':
                primary_key = find_primary_key_invoice(all_blocks)
            elif doc_type == 'estimate':
                primary_key = find_primary_key_estimate(all_blocks)
            else:
                # Fallback: try both methods
                primary_key = find_primary_key_invoice(all_blocks) or find_primary_key_estimate(all_blocks)
            
            if primary_key:
                extracted_data["work_order_number"] = primary_key
                extracted_data["extraction_confidence"]["work_order_found"] = True
            
            # Step 3: Extract total cost
            total_cost = find_grand_total(all_blocks)
            if total_cost:
                extracted_data["total_cost"] = total_cost
                extracted_data["extraction_confidence"]["total_cost_found"] = True
        
        return extracted_data
        
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return {
            "work_order_number": None,
            "total_cost": None,
            "extraction_confidence": {
                "work_order_found": False,
                "total_cost_found": False,
                "spatial_match": False,
                "document_type": None
            },
            "error": str(e)
        }

def post_process_single_image():
    """
    Post-process the last processed image result to extract structured data.
    """
    try:
        # Check if we have a result to process
        if 'last_result' not in globals():
            print("No image result available. Please run the single image test first.")
            return
        
        print("\n" + "="*50)
        print("POST-PROCESSING EXTRACTION")
        print("="*50)
        
        # Extract structured data
        extracted_data = extract_work_order_and_total(last_result)
        
        # Display results
        print("\nExtracted Data:")
        print(json.dumps(extracted_data, indent=2))
        
        # Provide user feedback
        if extracted_data["extraction_confidence"]["work_order_found"]:
            print(f"\n✅ Work Order Number found: {extracted_data['work_order_number']}")
        else:
            print("\n❌ Work Order Number not found")
            
        if extracted_data["extraction_confidence"]["total_cost_found"]:
            print(f"✅ Total Cost found: ${extracted_data['total_cost']}")
        else:
            print("❌ Total Cost not found")
            
        if extracted_data["extraction_confidence"]["spatial_match"]:
            print("✅ Spatial filtering successful")
        else:
            print("❌ No spatial matches found")
        
        return extracted_data
        
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return None

# Run post-processing on the last result
post_process_single_image()



# %% [markdown]
"""
## Batch Processing
"""

# %%
def generate_test_id() -> str:
    """Generate a unique test identifier using timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def collect_test_metadata() -> dict:
    """Collect metadata about the current test configuration."""
    return {
        "test_id": generate_test_id(),
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": "docTR",
            "version": "1.0",
            "model_id": "db_resnet50+crnn_vgg16_bn",
            "quantization": "N/A",
            "parameters": {
                "resolve_blocks": True,
                "device": str(device),
                "det_arch": "db_resnet50",
                "reco_arch": "crnn_vgg16_bn"
            }
        },
        "prompt_type": "N/A (OCR only)",
        "system_resources": check_cuda_availability()
    }

def save_incremental_results(results_file: Path, results: list, metadata: dict):
    """Save results incrementally to avoid losing progress."""
    complete_results = {
        "metadata": metadata,
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2)

def process_single_image_for_batch(image_path: str) -> dict:
    """
    Process a single image for batch processing, returning structured results.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Processing results including extracted data and metadata
    """
    result_entry = {
        "image_name": Path(image_path).name,
        "status": "processing",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        print(f"Processing image: {Path(image_path).name}")
        
        # Load image using DocumentFile
        doc = DocumentFile.from_images(image_path)
        
        # Run inference with timing
        start_time = time.time()
        with torch.no_grad():
            doctr_result = model(doc)
        processing_time = time.time() - start_time
        
        # Get rendered text output
        rendered_text = doctr_result.render()
        
        # Extract structured data using the same post-processing logic
        extracted_data = extract_work_order_and_total(doctr_result)
        
        # Update result entry with success data
        result_entry.update({
            "status": "completed",
            "processing_time": round(processing_time, 2),
            "extracted_data": {
                "work_order_number": extracted_data["work_order_number"],
                "total_cost": extracted_data["total_cost"]
            },
            "extraction_confidence": extracted_data["extraction_confidence"],
            "raw_ocr_output": rendered_text
        })
        
        print(f"✅ Successfully processed {Path(image_path).name}")
        return result_entry
        
    except Exception as e:
        print(f"❌ Error processing {Path(image_path).name}: {e}")
        result_entry.update({
            "status": "error",
            "error": {
                "message": str(e),
                "type": "processing_error"
            }
        })
        return result_entry

def process_batch(image_dir: str = None, output_dir: str = None) -> str:
    """
    Process a batch of images and save results incrementally.
    
    Args:
        image_dir (str): Directory containing images to process
        output_dir (str): Directory to save results
        
    Returns:
        str: Path to the results file
    """
    try:
        # Set up directories
        if image_dir is None:
            image_dir = ROOT_DIR / "data" / "images"
        if output_dir is None:
            output_dir = ROOT_DIR / "results"
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Generate test metadata
        metadata = collect_test_metadata()
        test_id = metadata["test_id"]
        results_file = output_dir / f"test_results_{test_id}.json"
        
        print(f"\nFound {len(image_files)} images to process")
        print(f"Results will be saved to: {results_file}")
        print("=" * 50)
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] ", end="")
            
            # Process single image
            result = process_single_image_for_batch(str(image_path))
            results.append(result)
            
            # Save incremental results after each image
            save_incremental_results(results_file, results, metadata)
            
        print("\n" + "=" * 50)
        print(f"Batch processing completed!")
        print(f"Processed: {len([r for r in results if r['status'] == 'completed'])}/{len(results)} images")
        print(f"Errors: {len([r for r in results if r['status'] == 'error'])}/{len(results)} images")
        print(f"Results saved to: {results_file}")
        
        return str(results_file)
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise

# %%
# Test batch processing
test_batch_results = process_batch()

# %% [markdown]
"""
## Analysis
Functions for analyzing model performance and generating analysis reports.
"""

# %%
def normalize_total_cost(cost_str: str) -> float:
    """Convert a cost string to a float by removing currency symbols and commas."""
    if not cost_str:
        return None
    # If already a float, return as is
    if isinstance(cost_str, (int, float)):
        return float(cost_str)
    # Remove $ and commas, then convert to float
    try:
        return float(cost_str.replace('$', '').replace(',', '').strip())
    except (ValueError, TypeError):
        return None

def categorize_work_order_error(predicted: str, ground_truth: str) -> str:
    """Categorize the type of error in work order number prediction."""
    if not predicted or not ground_truth:
        return "No Extraction"
    if predicted == ground_truth:
        return "Exact Match"
    # Check if prediction looks like a date (contains - or /)
    if '-' in predicted or '/' in predicted:
        return "Date Confusion"
    # Check for partial match (some digits match)
    if any(digit in ground_truth for digit in predicted):
        return "Partial Match"
    return "Completely Wrong"

def categorize_total_cost_error(predicted: float, ground_truth: float) -> str:
    """Categorize the type of error in total cost prediction."""
    if predicted is None or ground_truth is None:
        return "No Extraction"
    if predicted == ground_truth:
        return "Numeric Match"
    
    # Convert to strings for digit comparison
    pred_str = str(int(predicted))
    truth_str = str(int(ground_truth))
    
    # Check for digit reversal
    if pred_str[::-1] == truth_str:
        return "Digit Reversal"
    
    # Check for missing digit
    if len(pred_str) == len(truth_str) - 1 and all(d in truth_str for d in pred_str):
        return "Missing Digit"
    
    # Check for extra digit
    if len(pred_str) == len(truth_str) + 1 and all(d in pred_str for d in truth_str):
        return "Extra Digit"
    
    return "Completely Wrong"

def calculate_cer(str1: str, str2: str) -> float:
    """Calculate Character Error Rate between two strings."""
    if not str1 or not str2:
        return 1.0  # Return maximum error if either string is empty
    
    # Convert to strings and remove whitespace
    str1 = str(str1).strip()
    str2 = str(str2).strip()
    
    # Calculate Levenshtein distance
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    
    if len(str2) == 0:
        return 1.0
    
    previous_row = range(len(str2) + 1)
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        for j, c2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    # Return CER as distance divided by length of longer string
    return previous_row[-1] / len(str1)

def analyze_results(results_file: str, ground_truth_file: str = "data/ground_truth.csv") -> dict:
    """Analyze model performance and generate analysis report."""
    import pandas as pd
    
    # Load results and ground truth
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Read ground truth with explicit string type for Invoice column
    ground_truth = pd.read_csv(ground_truth_file, dtype={'Invoice': str})
    
    # Initialize analysis structure
    analysis = {
        "metadata": results["metadata"],
        "summary": {
            "total_images": len(results["results"]),
            "completed": 0,
            "errors": 0,
            "work_order_accuracy": 0,
            "total_cost_accuracy": 0,
            "average_cer": 0,
            "document_type_detection": {
                "invoice_detected": 0,
                "estimate_detected": 0,
                "type_unknown": 0
            },
            "confidence_analysis": {
                "spatial_match_rate": 0,
                "work_order_confidence_accuracy": 0,
                "total_cost_confidence_accuracy": 0
            }
        },
        "error_categories": {
            "work_order": {},
            "total_cost": {}
        },
        "document_type_performance": {
            "invoice": {"work_order_accuracy": 0, "total_cost_accuracy": 0, "count": 0},
            "estimate": {"work_order_accuracy": 0, "total_cost_accuracy": 0, "count": 0},
            "unknown": {"work_order_accuracy": 0, "total_cost_accuracy": 0, "count": 0}
        },
        "results": []
    }
    
    # Process each result
    total_cer = 0
    work_order_matches = 0
    total_cost_matches = 0
    spatial_match_count = 0
    confidence_work_order_correct = 0
    confidence_total_cost_correct = 0
    
    # Document type counters
    doc_type_counts = {"invoice": 0, "estimate": 0, "unknown": 0}
    doc_type_work_order_matches = {"invoice": 0, "estimate": 0, "unknown": 0}
    doc_type_total_cost_matches = {"invoice": 0, "estimate": 0, "unknown": 0}
    
    for result in results["results"]:
        # Get ground truth for this image - remove .jpg extension for matching
        image_id = result["image_name"].replace(".jpg", "")
        
        gt_row = ground_truth[ground_truth["Invoice"] == image_id]
        
        if gt_row.empty:
            print(f"Warning: No ground truth found for image {image_id}")
            continue
            
        gt_work_order = str(gt_row["Work Order Number/Numero de Orden"].iloc[0]).strip()
        gt_total_cost = normalize_total_cost(str(gt_row["Total"].iloc[0]))
        gt_doc_type = str(gt_row["Type"].iloc[0]).lower()
        
        # Initialize result analysis
        result_analysis = {
            "image_name": result["image_name"],
            "status": result["status"],
            "ground_truth_type": gt_doc_type
        }
        
        if result["status"] == "completed":
            analysis["summary"]["completed"] += 1
            
            # Get extracted data and confidence info
            extracted_data = result["extracted_data"]
            confidence_info = result["extraction_confidence"]
            
            # Document type analysis
            detected_type = confidence_info.get("document_type", "unknown")
            if detected_type == "invoice":
                analysis["summary"]["document_type_detection"]["invoice_detected"] += 1
            elif detected_type == "estimate":
                analysis["summary"]["document_type_detection"]["estimate_detected"] += 1
            else:
                analysis["summary"]["document_type_detection"]["type_unknown"] += 1
            
            # Count document types for performance analysis
            doc_type_key = detected_type if detected_type else "unknown"
            doc_type_counts[doc_type_key] += 1
            
            # Spatial match analysis
            if confidence_info.get("spatial_match", False):
                spatial_match_count += 1
            
            # Analyze work order
            pred_work_order = extracted_data["work_order_number"]
            work_order_error = categorize_work_order_error(pred_work_order, gt_work_order)
            work_order_cer = calculate_cer(pred_work_order, gt_work_order)
            
            work_order_correct = work_order_error == "Exact Match"
            if work_order_correct:
                work_order_matches += 1
                doc_type_work_order_matches[doc_type_key] += 1
            
            # Confidence vs accuracy analysis for work order
            if confidence_info.get("work_order_found", False) and work_order_correct:
                confidence_work_order_correct += 1
            
            # Analyze total cost
            pred_total_cost = normalize_total_cost(extracted_data["total_cost"])
            total_cost_error = categorize_total_cost_error(pred_total_cost, gt_total_cost)
            
            total_cost_correct = total_cost_error == "Numeric Match"
            if total_cost_correct:
                total_cost_matches += 1
                doc_type_total_cost_matches[doc_type_key] += 1
            
            # Confidence vs accuracy analysis for total cost
            if confidence_info.get("total_cost_found", False) and total_cost_correct:
                confidence_total_cost_correct += 1
            
            # Update result analysis
            result_analysis.update({
                "extracted_data": extracted_data,
                "extraction_confidence": confidence_info,
                "detected_type": detected_type,
                "type_match": detected_type == gt_doc_type,
                "work_order": {
                    "predicted": pred_work_order,
                    "ground_truth": gt_work_order,
                    "error_category": work_order_error,
                    "cer": work_order_cer,
                    "correct": work_order_correct
                },
                "total_cost": {
                    "predicted": pred_total_cost,
                    "ground_truth": gt_total_cost,
                    "error_category": total_cost_error,
                    "correct": total_cost_correct
                }
            })
            
            # Update error categories
            analysis["error_categories"]["work_order"][work_order_error] = \
                analysis["error_categories"]["work_order"].get(work_order_error, 0) + 1
            analysis["error_categories"]["total_cost"][total_cost_error] = \
                analysis["error_categories"]["total_cost"].get(total_cost_error, 0) + 1
            
            total_cer += work_order_cer
            
        else:
            analysis["summary"]["errors"] += 1
            result_analysis["error"] = result.get("error", "Unknown error")
        
        analysis["results"].append(result_analysis)
    
    # Calculate summary statistics
    total_images = analysis["summary"]["completed"]
    if total_images > 0:
        analysis["summary"]["work_order_accuracy"] = work_order_matches / total_images
        analysis["summary"]["total_cost_accuracy"] = total_cost_matches / total_images
        analysis["summary"]["average_cer"] = total_cer / total_images
        analysis["summary"]["confidence_analysis"]["spatial_match_rate"] = spatial_match_count / total_images
        
        # Confidence accuracy rates
        work_order_confident_count = sum(1 for r in analysis["results"] 
                                       if r.get("extraction_confidence", {}).get("work_order_found", False))
        total_cost_confident_count = sum(1 for r in analysis["results"] 
                                       if r.get("extraction_confidence", {}).get("total_cost_found", False))
        
        if work_order_confident_count > 0:
            analysis["summary"]["confidence_analysis"]["work_order_confidence_accuracy"] = confidence_work_order_correct / work_order_confident_count
        if total_cost_confident_count > 0:
            analysis["summary"]["confidence_analysis"]["total_cost_confidence_accuracy"] = confidence_total_cost_correct / total_cost_confident_count
    
    # Calculate document type performance
    for doc_type in ["invoice", "estimate", "unknown"]:
        count = doc_type_counts[doc_type]
        if count > 0:
            analysis["document_type_performance"][doc_type] = {
                "work_order_accuracy": doc_type_work_order_matches[doc_type] / count,
                "total_cost_accuracy": doc_type_total_cost_matches[doc_type] / count,
                "count": count
            }
    
    return analysis

def select_test_results_file() -> Path:
    """Allow user to select a test results file for analysis."""
    # Get all test result files
    results_dir = ROOT_DIR / "results"
    result_files = list(results_dir.glob("test_results_*.json"))
    
    if not result_files:
        raise FileNotFoundError("No test result files found in results directory")
    
    # Sort files by modification time (newest first)
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("\nAvailable test result files:")
    for i, file in enumerate(result_files, 1):
        # Get file modification time
        mod_time = datetime.fromtimestamp(file.stat().st_mtime)
        print(f"{i}. {file.name} (Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    while True:
        try:
            choice = int(input("\nSelect a test result file (1-{}): ".format(len(result_files))))
            if 1 <= choice <= len(result_files):
                selected_file = result_files[choice - 1]
                print(f"\nSelected file: {selected_file.name}")
                return selected_file
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(result_files)}.")
        except ValueError:
            print("Please enter a valid number.")

def run_analysis():
    """Run the analysis and generate comprehensive performance report."""
    try:
        # Get test results file
        results_file = select_test_results_file()
        
        # Generate analysis
        analysis = analyze_results(str(results_file))
        
        # Create analysis directory if it doesn't exist
        analysis_dir = ROOT_DIR / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save analysis to file
        analysis_file = analysis_dir / f"analysis_{analysis['metadata']['test_id']}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Display comprehensive summary
        print("\n" + "="*60)
        print("DOCTR MODEL PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Basic Performance
        print(f"\nBasic Performance:")
        print(f"Total Images: {analysis['summary']['total_images']}")
        print(f"Completed: {analysis['summary']['completed']}")
        print(f"Errors: {analysis['summary']['errors']}")
        print(f"Work Order Accuracy: {analysis['summary']['work_order_accuracy']:.2%}")
        print(f"Total Cost Accuracy: {analysis['summary']['total_cost_accuracy']:.2%}")
        print(f"Average CER: {analysis['summary']['average_cer']:.3f}")
        
        # Document Type Detection
        print(f"\nDocument Type Detection:")
        doc_detection = analysis['summary']['document_type_detection']
        print(f"Invoice Detected: {doc_detection['invoice_detected']}")
        print(f"Estimate Detected: {doc_detection['estimate_detected']}")
        print(f"Type Unknown: {doc_detection['type_unknown']}")
        
        # Confidence Analysis
        print(f"\nConfidence Analysis:")
        confidence = analysis['summary']['confidence_analysis']
        print(f"Spatial Match Rate: {confidence['spatial_match_rate']:.2%}")
        print(f"Work Order Confidence Accuracy: {confidence['work_order_confidence_accuracy']:.2%}")
        print(f"Total Cost Confidence Accuracy: {confidence['total_cost_confidence_accuracy']:.2%}")
        
        # Document Type Performance
        print(f"\nPerformance by Document Type:")
        for doc_type, perf in analysis['document_type_performance'].items():
            if perf['count'] > 0:
                print(f"  {doc_type.title()} (n={perf['count']}):")
                print(f"    Work Order: {perf['work_order_accuracy']:.2%}")
                print(f"    Total Cost: {perf['total_cost_accuracy']:.2%}")
        
        # Error Categories
        print(f"\nWork Order Error Categories:")
        for category, count in analysis['error_categories']['work_order'].items():
            print(f"  {category}: {count}")
        
        print(f"\nTotal Cost Error Categories:")
        for category, count in analysis['error_categories']['total_cost'].items():
            print(f"  {category}: {count}")
        
        print(f"\nDetailed analysis saved to: {analysis_file}")
        print("="*60)
        
        return analysis
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

# %% [markdown]
"""
## Run Analysis
Generate and display analysis of model performance.
"""

# %%
# Run the analysis
analysis_results = run_analysis() 