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

# Determine root directory by finding .gitignore and .gitattributes
def find_project_root() -> Path:
    """Find project root by locating directory containing .gitignore and .gitattributes"""
    try:
        # When running as a script, start from script location
        start_path = Path(__file__).parent
    except NameError:
        # When running in a notebook, start from current working directory
        start_path = Path.cwd()
    
    # Walk up the directory tree to find git markers
    current_path = start_path
    while current_path != current_path.parent:  # Stop at filesystem root
        if (current_path / ".gitignore").exists() and (current_path / ".gitattributes").exists():
            return current_path
        current_path = current_path.parent
    
    raise RuntimeError("Could not find project root (directory containing .gitignore and .gitattributes)")

ROOT_DIR = find_project_root()

# Verify expected files exist in the Deliverables-Code directory
deliverables_dir = ROOT_DIR / "Deliverables-Code"
if not deliverables_dir.exists():
    raise RuntimeError("Could not find Deliverables-Code directory in project root")

def install_doctr_dependencies():
    """Install docTR dependencies with PyTorch version checking."""
    requirements_file = ROOT_DIR / "Deliverables-Code" / "requirements" / "requirements_doctr.txt"
    
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found at {requirements_file}")
    
    # Check if PyTorch is already installed with correct version
    pytorch_compatible = False
    try:
        import torch
        torch_version = torch.__version__
        if torch_version.startswith("2.1.0"):
            print(f"Compatible PyTorch {torch_version} already installed")
            pytorch_compatible = True
        else:
            print(f"PyTorch {torch_version} found but may need update for docTR compatibility")
    except ImportError:
        print("PyTorch not found, will install from requirements")
    
    print(f"Installing docTR dependencies from {requirements_file}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        raise

# Install dependencies
install_doctr_dependencies()

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
## Model Configuration and Selection
"""

# %%
# Global variables to store selected model architectures
SELECTED_DET_ARCH = None
SELECTED_RECO_ARCH = None

def select_detection_architecture() -> str:
    """
    Allow user to select a detection architecture.
    
    Returns:
        str: Selected detection architecture
    """
    detection_options = {
        1: "db_resnet50",
        2: "db_mobilenet_v3_large", 
        3: "linknet_resnet18",
        4: "linknet_resnet34",
        5: "linknet_resnet50"
    }
    
    print("\nAvailable Detection Architectures:")
    for i, arch in detection_options.items():
        print(f"{i}. {arch}")
    
    while True:
        try:
            choice = int(input(f"\nSelect detection architecture (1-{len(detection_options)}) [default: 1]: ") or "1")
            if 1 <= choice <= len(detection_options):
                selected_arch = detection_options[choice]
                print(f"Selected detection architecture: {selected_arch}")
                return selected_arch
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(detection_options)}.")
        except ValueError:
            print("Please enter a valid number.")

def select_recognition_architecture() -> str:
    """
    Allow user to select a recognition architecture.
    
    Returns:
        str: Selected recognition architecture
    """
    recognition_options = {
        1: "crnn_vgg16_bn",
        2: "crnn_mobilenet_v3_small",
        3: "crnn_mobilenet_v3_large",
        4: "master",
        5: "sar_resnet31",
        6: "vitstr_small",
        7: "vitstr_base"
    }
    
    print("\nAvailable Recognition Architectures:")
    for i, arch in recognition_options.items():
        print(f"{i}. {arch}")
    
    while True:
        try:
            choice = int(input(f"\nSelect recognition architecture (1-{len(recognition_options)}) [default: 1]: ") or "1")
            if 1 <= choice <= len(recognition_options):
                selected_arch = recognition_options[choice]
                print(f"Selected recognition architecture: {selected_arch}")
                return selected_arch
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(recognition_options)}.")
        except ValueError:
            print("Please enter a valid number.")

# Select model architectures
SELECTED_DET_ARCH = select_detection_architecture()
SELECTED_RECO_ARCH = select_recognition_architecture()

# %% [markdown]
"""
## Model Initialization
"""

# %%
def initialize_model() -> tuple:
    """
    Initialize docTR OCR model with user-selected architectures.
    
    Returns:
        tuple: (model, device)
    """
    try:
        print(f"\nInitializing docTR model...")
        print(f"Detection Architecture: {SELECTED_DET_ARCH}")
        print(f"Recognition Architecture: {SELECTED_RECO_ARCH}")
        
        # Initialize model with selected architectures
        model = ocr_predictor(
            det_arch=SELECTED_DET_ARCH,
            reco_arch=SELECTED_RECO_ARCH,
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
test_image_path = ROOT_DIR / "Deliverables-Code" / "data" / "images" / "1_curated" / "1017.jpg"
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
                "spatial_extraction_successful": False,
                "parsing_method": "spatial_analysis",
                "work_order_found": False,
                "total_cost_found": False,
                "overall_confidence": 0.0
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
def generate_results_filename(model_name: str, quantization_level: str, results_dir: Path) -> tuple[str, str]:
    """
    Generate a results filename with auto-incrementing counter.
    
    Args:
        model_name: Name of the model (e.g., "pixtral", "llama", "doctr")
        quantization_level: Quantization level (e.g., "bfloat16", "int8", "int4", "none")
        results_dir: Directory where results are stored
        
    Returns:
        tuple: (filename_without_extension, full_filepath)
    """
    # Find existing files with the same model and quantization pattern
    pattern = f"results-{model_name}-{quantization_level}-*.json"
    existing_files = list(results_dir.glob(pattern))
    
    # Extract counter numbers from existing files
    counter_numbers = []
    for file in existing_files:
        try:
            # Extract number from filename like "results-doctr-none-3.json"
            parts = file.stem.split('-')
            if len(parts) >= 4:
                counter_numbers.append(int(parts[-1]))
        except ValueError:
            continue
    
    # Get next counter number
    next_counter = max(counter_numbers, default=0) + 1
    
    # Generate filename
    filename_base = f"results-{model_name}-{quantization_level}-{next_counter}"
    full_filepath = results_dir / f"{filename_base}.json"
    
    return filename_base, str(full_filepath)

def generate_test_id() -> str:
    """Generate a unique test identifier using timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def collect_test_metadata(test_id: str) -> dict:
    """Collect metadata about the current test configuration."""
    # Get GPU information
    gpu_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    
    return {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": "docTR",
            "version": "1.0",
            "model_id": {
                "detection_model": SELECTED_DET_ARCH,
                "recognition_model": SELECTED_RECO_ARCH,
                "combined": f"{SELECTED_DET_ARCH}+{SELECTED_RECO_ARCH}"
            },
            "model_type": "ocr",
            "quantization": {
                "type": "none",
                "config": {}
            },
            "device_info": {
                "device_map": str(device),
                "use_flash_attention": False,
                "gpu_memory_gb": round(gpu_props.total_memory / (1024**3), 2) if gpu_props else None,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}" if gpu_props else None
            }
        },
        "prompt_info": {
            "prompt_type": "N/A",
            "raw_text": "N/A (OCR only)",
            "formatted_text": "N/A (OCR only)",
            "special_tokens": []
        },
        "processing_config": {
            "inference_params": {
                "resolve_blocks": True,
                "det_arch": SELECTED_DET_ARCH,
                "reco_arch": SELECTED_RECO_ARCH
            },
            "image_preprocessing": {
                "max_size": [1024, 1024],
                "format": "RGB",
                "resize_strategy": "maintain_aspect_ratio"
            }
        }
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
    Process a single image for batch processing, returning structured results with raw OCR output only.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Processing results including raw OCR data and metadata
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
        
        # Export to JSON for spatial data
        json_result = doctr_result.export()
        
        # Count OCR statistics
        total_blocks = sum(len(page['blocks']) for page in json_result['pages'])
        total_lines = sum(len(block['lines']) for page in json_result['pages'] for block in page['blocks'])
        total_words = sum(len(line['words']) for page in json_result['pages'] for block in page['blocks'] for line in block['lines'])
        
        # Update result entry with raw OCR output only
        result_entry.update({
            "status": "completed",
            "processing_time_seconds": round(processing_time, 2),
            "raw_output": {
                "ocr_text": rendered_text,
                "spatial_data": json_result,
                "detection_results": {
                    "blocks_detected": total_blocks,
                    "lines_detected": total_lines,
                    "words_detected": total_words
                },
                "page_dimensions": json_result['pages'][0]['dimensions'] if json_result['pages'] else None
            }
        })
        
        print(f"✅ Successfully processed {Path(image_path).name}")
        return result_entry
        
    except Exception as e:
        print(f"❌ Error processing {Path(image_path).name}: {e}")
        result_entry.update({
            "status": "error",
            "error": {
                "type": "processing_error",
                "message": str(e),
                "stage": "inference"
            }
        })
        return result_entry

def process_batch(image_dir: str = None, output_dir: str = None) -> str:
    """
    Process a batch of images and save raw OCR results only.
    
    Args:
        image_dir (str): Directory containing images to process
        output_dir (str): Directory to save results
        
    Returns:
        str: Path to the results file
    """
    try:
        # Set up directories
        if image_dir is None:
            image_dir = ROOT_DIR / "Deliverables-Code" / "data" / "images" / "1_curated"
        if output_dir is None:
            output_dir = ROOT_DIR / "Deliverables-Code" / "results"
        
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
        
        # Generate filename with new naming convention
        test_id, results_file_path = generate_results_filename("doctr", "none", output_dir)
        results_file = Path(results_file_path)
        
        # Collect metadata with the test_id
        metadata = collect_test_metadata(test_id)
        
        print(f"\nStarting docTR batch test")
        print(f"Found {len(image_files)} images to process")
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
        print(f"Raw results saved to: {results_file}")
        
        return str(results_file)
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise

# %%
# Test batch processing
test_batch_results = process_batch()

# %% [markdown]
"""
## Analysis Helper Functions
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

def select_test_results_file() -> Path:
    """Allow user to select a test results file for analysis."""
    # Get all test result files
    results_dir_path = ROOT_DIR / "Deliverables-Code" / "results"
    result_files = list(results_dir_path.glob("results-doctr-*.json"))
    
    if not result_files:
        raise FileNotFoundError("No docTR test result files found in results directory")
    
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

# %% [markdown]
"""
## Analysis Functions - Data Processing Phase
Functions for analyzing raw OCR outputs and generating structured analysis reports.
"""

# %%
def analyze_raw_results(results_file: str, ground_truth_file: str = None) -> dict:
    """Analyze raw OCR results and generate analysis report."""
    import pandas as pd
    
    # Set default ground truth file path
    if ground_truth_file is None:
        ground_truth_file = str(ROOT_DIR / "Deliverables-Code" / "data" / "images" / "metadata" / "ground_truth.csv")
    
    # Load results and ground truth
    with open(results_file, 'r') as f:
        raw_results = json.load(f)
    
    # Read ground truth with explicit string type for Invoice column
    ground_truth = pd.read_csv(ground_truth_file, dtype={'Invoice': str})
    
    # Initialize analysis structure
    analysis = {
        "source_results": results_file,
        "extraction_method": "spatial_analysis_v2",
        "ground_truth_file": ground_truth_file,
        "metadata": raw_results["metadata"],
        "summary": {
            "total_images": len(raw_results["results"]),
            "completed": 0,
            "errors": 0,
            "spatial_extraction_successful": 0,
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
        "extracted_data": [],
        "performance_metrics": {}
    }
    
    # Process each result
    total_cer = 0
    work_order_matches = 0
    total_cost_matches = 0
    spatial_successful = 0
    spatial_match_count = 0
    confidence_work_order_correct = 0
    confidence_total_cost_correct = 0
    
    # Document type counters
    doc_type_counts = {"invoice": 0, "estimate": 0, "unknown": 0}
    doc_type_work_order_matches = {"invoice": 0, "estimate": 0, "unknown": 0}
    doc_type_total_cost_matches = {"invoice": 0, "estimate": 0, "unknown": 0}
    
    for result in raw_results["results"]:
        # Get ground truth for this image - remove .jpg extension for matching
        image_id = result["image_name"].replace(".jpg", "")
        
        gt_row = ground_truth[ground_truth["Invoice"] == image_id]
        
        if gt_row.empty:
            logger.warning(f"No ground truth found for image {image_id}")
            continue
            
        gt_work_order = str(gt_row["Work Order Number/Numero de Orden"].iloc[0]).strip()
        gt_total_cost = normalize_total_cost(str(gt_row["Total"].iloc[0]))
        gt_doc_type = str(gt_row["Type"].iloc[0]).lower()
        
        # Initialize extraction entry
        extraction_entry = {
            "image_name": result["image_name"],
            "status": result["status"],
            "raw_ocr_text": result.get("raw_output", {}).get("ocr_text", ""),
            "spatial_data": result.get("raw_output", {}).get("spatial_data", {}),
            "ground_truth": {
                "work_order_number": gt_work_order,
                "total_cost": gt_total_cost,
                "document_type": gt_doc_type
            }
        }
        
        if result["status"] == "completed":
            analysis["summary"]["completed"] += 1
            
            # Extract structured data from spatial data
            spatial_data = result["raw_output"]["spatial_data"]
            extracted_data = extract_work_order_and_total(spatial_data)
            
            if extracted_data and not extracted_data.get("error"):
                spatial_successful += 1
                
                # Get confidence info
                confidence_info = extracted_data["extraction_confidence"]
                
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
                pred_work_order = extracted_data.get("work_order_number", "")
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
                pred_total_cost = normalize_total_cost(extracted_data.get("total_cost", ""))
                total_cost_error = categorize_total_cost_error(pred_total_cost, gt_total_cost)
                
                total_cost_correct = total_cost_error == "Numeric Match"
                if total_cost_correct:
                    total_cost_matches += 1
                    doc_type_total_cost_matches[doc_type_key] += 1
                
                # Confidence vs accuracy analysis for total cost
                if confidence_info.get("total_cost_found", False) and total_cost_correct:
                    confidence_total_cost_correct += 1
                
                # Update extraction entry
                extraction_entry.update({
                    "extracted_data": {
                        "work_order_number": pred_work_order,
                        "total_cost": pred_total_cost,
                        "document_type": detected_type
                    },
                    "extraction_confidence": confidence_info,
                    "performance": {
                        "work_order_error_category": work_order_error,
                        "total_cost_error_category": total_cost_error,
                        "work_order_cer": work_order_cer,
                        "work_order_correct": work_order_correct,
                        "total_cost_correct": total_cost_correct,
                        "type_match": detected_type == gt_doc_type
                    }
                })
                
                # Update error categories
                analysis["error_categories"]["work_order"][work_order_error] = \
                    analysis["error_categories"]["work_order"].get(work_order_error, 0) + 1
                analysis["error_categories"]["total_cost"][total_cost_error] = \
                    analysis["error_categories"]["total_cost"].get(total_cost_error, 0) + 1
                
                total_cer += work_order_cer
            else:
                extraction_entry.update({
                    "extraction_error": extracted_data.get("error", "Spatial extraction failed"),
                    "extraction_confidence": {
                        "spatial_extraction_successful": False,
                        "parsing_method": "spatial_analysis",
                        "work_order_found": False,
                        "total_cost_found": False,
                        "overall_confidence": 0.0
                    }
                })
        else:
            analysis["summary"]["errors"] += 1
            extraction_entry["processing_error"] = result.get("error", {})
        
        analysis["extracted_data"].append(extraction_entry)
    
    # Calculate summary statistics
    completed = analysis["summary"]["completed"]
    if completed > 0:
        analysis["summary"]["spatial_extraction_successful"] = spatial_successful
        analysis["summary"]["work_order_accuracy"] = work_order_matches / completed
        analysis["summary"]["total_cost_accuracy"] = total_cost_matches / completed
        analysis["summary"]["average_cer"] = total_cer / completed
        analysis["summary"]["confidence_analysis"]["spatial_match_rate"] = spatial_match_count / completed
        
        # Confidence accuracy rates
        work_order_confident_count = sum(1 for e in analysis["extracted_data"] 
                                       if e.get("extraction_confidence", {}).get("work_order_found", False))
        total_cost_confident_count = sum(1 for e in analysis["extracted_data"] 
                                       if e.get("extraction_confidence", {}).get("total_cost_found", False))
        
        if work_order_confident_count > 0:
            analysis["summary"]["confidence_analysis"]["work_order_confidence_accuracy"] = confidence_work_order_correct / work_order_confident_count
        if total_cost_confident_count > 0:
            analysis["summary"]["confidence_analysis"]["total_cost_confidence_accuracy"] = confidence_total_cost_correct / total_cost_confident_count
        
        # Performance metrics
        analysis["performance_metrics"] = {
            "spatial_extraction_rate": spatial_successful / completed,
            "work_order_extraction_rate": work_order_matches / completed,
            "total_cost_extraction_rate": total_cost_matches / completed,
            "average_processing_time": sum(
                r.get("processing_time_seconds", 0) 
                for r in raw_results["results"] 
                if r["status"] == "completed"
            ) / completed
        }
    
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

def run_analysis():
    """Run analysis on raw OCR results and generate comprehensive performance report."""
    try:
        # Get test results file
        results_file = select_test_results_file()
        
        # Generate analysis
        analysis = analyze_raw_results(str(results_file))
        
        # Create analysis directory if it doesn't exist
        analysis_dir = ROOT_DIR / "Deliverables-Code" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate analysis filename with same convention as results
        model_name = "doctr"
        quantization_level = "none"  # docTR doesn't use quantization
        
        # Find existing analysis files with the same model and quantization pattern
        pattern = f"analysis-{model_name}-{quantization_level}-*.json"
        existing_files = list(analysis_dir.glob(pattern))
        
        # Extract counter numbers from existing files
        counter_numbers = []
        for file in existing_files:
            try:
                # Extract number from filename like "analysis-doctr-none-3.json"
                parts = file.stem.split('-')
                if len(parts) >= 4:
                    counter_numbers.append(int(parts[-1]))
            except ValueError:
                continue
        
        # Get next counter number
        next_counter = max(counter_numbers, default=0) + 1
        
        # Generate analysis filename
        analysis_filename = f"analysis-{model_name}-{quantization_level}-{next_counter}.json"
        analysis_file = analysis_dir / analysis_filename
        
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
        print(f"Spatial Extraction Successful: {analysis['summary']['spatial_extraction_successful']}")
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
        
        # Performance Metrics
        print(f"\nPerformance Metrics:")
        for metric, value in analysis['performance_metrics'].items():
            if 'rate' in metric:
                print(f"- {metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"- {metric.replace('_', ' ').title()}: {value:.2f}")
        
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
Generate and display analysis of raw OCR results.
"""

# %%
# Run the analysis
analysis_results = run_analysis() 