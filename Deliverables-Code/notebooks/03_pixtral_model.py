# %% [markdown]
"""
# Pixtral Model Evaluation Notebook

This notebook evaluates the Pixtral-12B model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
Results are saved as pure data collection artifacts with analysis performed separately.
"""

# %% [markdown]
"""
## Setup and Configuration
### Initial Imports
"""

# %%
import os
import sys
import subprocess
from pathlib import Path
import logging
import json
from datetime import datetime
import torch
from PIL import Image
from typing import Union, Dict, Any, List, Literal
import yaml

# %% [markdown]
"""
### Define a Function and Global Variable to Decide and Hold the Prompt
"""

# %%
# Global variable to store selected prompt
SELECTED_PROMPT = None

def load_prompt_files() -> Dict[str, Dict]:
    """Load all prompt YAML files from the config/prompts directory."""
    prompts_dir = ROOT_DIR / "Deliverables-Code" / "config" / "prompts"
    prompt_files = {
        "basic_extraction": prompts_dir / "basic_extraction.yaml",
        "detailed": prompts_dir / "detailed.yaml",
        "few_shot": prompts_dir / "few_shot.yaml",
        "locational": prompts_dir / "locational.yaml",
        "step_by_step": prompts_dir / "step_by_step.yaml"
    }
    
    loaded_prompts = {}
    for name, file_path in prompt_files.items():
        with open(file_path, 'r') as f:
            loaded_prompts[name] = yaml.safe_load(f)
    return loaded_prompts

def select_prompt() -> str:
    """Allow user to select a prompt type and return the prompt text."""
    global SELECTED_PROMPT
    
    prompts = load_prompt_files()
    print("\nAvailable prompt types:")
    for i, name in enumerate(prompts.keys(), 1):
        print(f"{i}. {name.replace('_', ' ').title()}")
    
    while True:
        try:
            choice = int(input("\nSelect a prompt type (1-5): "))
            if 1 <= choice <= len(prompts):
                selected_name = list(prompts.keys())[choice - 1]
                SELECTED_PROMPT = prompts[selected_name]
                print(f"\nSelected prompt type: {selected_name.replace('_', ' ').title()}")
                print("\nPrompt text:")
                print("-" * 50)
                print(SELECTED_PROMPT['prompts'][0]['text'])
                print("-" * 50)
                return selected_name
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")

# %% [markdown]
"""
### Logging Configuration
"""
# %%
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
"""
### Root Directory Determination
"""
# %%
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
logger.info(f"Found project root: {ROOT_DIR}")

# Verify expected files exist in the Deliverables-Code directory
deliverables_dir = ROOT_DIR / "Deliverables-Code"
if not deliverables_dir.exists():
    raise RuntimeError("Could not find Deliverables-Code directory in project root")

sys.path.append(str(ROOT_DIR))

# Create results directory
results_dir = ROOT_DIR / "Deliverables-Code" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Results will be saved to: {results_dir}")

# %% [markdown]
"""
## Install Dependencies
"""

# %%
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
from tqdm import tqdm


def install_dependencies():
    """Install required dependencies with progress tracking."""
    # Check if PyTorch is already installed with correct version
    try:
        import torch
        torch_version = torch.__version__
        if torch_version.startswith("2.1.0") and "cu118" in torch_version:
            logger.info(f"PyTorch {torch_version} already installed, skipping PyTorch installation")
            pytorch_step = None
        else:
            pytorch_step = ("PyTorch", [
                sys.executable, "-m", "pip", "install", "-q",
                "torch==2.1.0",
                "torchvision==0.16.0",
                "torchaudio==2.1.0",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
    except ImportError:
        pytorch_step = ("PyTorch", [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.1.0",
            "torchvision==0.16.0",
            "torchaudio==2.1.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    
    steps = [
        ("Base requirements", [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT_DIR / "Deliverables-Code" / "requirements" / "requirements_pixtral.txt")])
    ]
    
    if pytorch_step:
        steps.append(pytorch_step)
    
    for step_name, command in tqdm(steps, desc="Installing dependencies"):
        try:
            subprocess.check_call(command)
            logger.info(f"Successfully installed {step_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing {step_name}: {e}")
            raise

# Install dependencies
install_dependencies()

# Clear CUDA cache to prevent conflicts
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
"""
## Memory Resource Check
"""

# %%
def check_memory_resources():
    """
    Check available GPU memory, system RAM, and compare with Pixtral model requirements.
    Returns a dictionary with memory information and recommendations.
    """
    memory_info = {
        "gpu_available": False,
        "gpu_memory": None,
        "system_ram": None,
        "recommendations": []
    }
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        memory_info["gpu_available"] = True
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        memory_info["gpu_memory"] = round(gpu_mem, 2)
        logger.info(f"GPU Memory Available: {memory_info['gpu_memory']} GB")
    else:
        logger.warning("No GPU available. This will significantly impact model performance.")
        memory_info["recommendations"].append("No GPU detected. Consider using a GPU-enabled environment.")
    
    # Check system RAM
    import psutil
    system_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    memory_info["system_ram"] = round(system_ram, 2)
    logger.info(f"System RAM Available: {memory_info['system_ram']} GB")
    
    # Model requirements and recommendations
    model_requirements = {
        "no_quantization": 93.0,  # GB
        "8bit_quantization": 46.0,  # GB
        "4bit_quantization": 23.0   # GB
    }
    
    # Add recommendations based on available resources
    if memory_info["gpu_available"]:
        if memory_info["gpu_memory"] >= model_requirements["no_quantization"]:
            memory_info["recommendations"].append("Sufficient GPU memory for full precision model")
        elif memory_info["gpu_memory"] >= model_requirements["8bit_quantization"]:
            memory_info["recommendations"].append("Consider using 8-bit quantization")
        elif memory_info["gpu_memory"] >= model_requirements["4bit_quantization"]:
            memory_info["recommendations"].append("Consider using 4-bit quantization")
        else:
            memory_info["recommendations"].append("Insufficient GPU memory. Consider using CPU offloading or a different model.")
    
    # Check if system RAM is sufficient for CPU offloading if needed
    if memory_info["system_ram"] < model_requirements["4bit_quantization"]:
        memory_info["recommendations"].append("Warning: System RAM may be insufficient for CPU offloading")
    
    return memory_info

# Check memory resources
memory_status = check_memory_resources()
logger.info("Memory Status:")
for key, value in memory_status.items():
    if key != "recommendations":
        logger.info(f"{key}: {value}")
logger.info("Recommendations:")
for rec in memory_status["recommendations"]:
    logger.info(f"- {rec}")

# %% [markdown]
"""
## Quantization Selection
"""

# %%
def select_quantization() -> Literal["bfloat16", "int8", "int4"]:
    """
    Select quantization level for the model.
    Returns one of: "bfloat16", "int8", "int4"
    """
    print("\nAvailable quantization options:")
    print("1. bfloat16 (full precision, 93GB VRAM)")
    print("2. int8 (8-bit, 46GB VRAM)")
    print("3. int4 (4-bit, 23GB VRAM)")
    
    while True:
        try:
            choice = int(input("\nSelect quantization (1-3): "))
            if choice == 1:
                return "bfloat16"
            elif choice == 2:
                return "int8"
            elif choice == 3:
                return "int4"
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except ValueError:
            print("Please enter a number between 1 and 3.")

# Select quantization
quantization = select_quantization()
logger.info(f"Selected quantization: {quantization}")

# %% [markdown]
"""
## Flash Attention Configuration
"""

# %%
def configure_flash_attention() -> bool:
    """
    Check if Flash Attention is available and configure it.
    Returns True if Flash Attention is enabled, False otherwise.
    """
    try:
        import flash_attn
        
        # Check if GPU supports Flash Attention
        if not torch.cuda.is_available():
            logger.warning("Flash Attention requires CUDA GPU. Disabling Flash Attention.")
            return False
            
        # Get GPU compute capability
        major, minor = torch.cuda.get_device_capability()
        compute_capability = float(f"{major}.{minor}")
        
        # Flash Attention 2 requires compute capability >= 8.0
        if compute_capability >= 8.0:
            logger.info("Flash Attention 2 enabled - GPU supports compute capability 8.0+")
            return True
        else:
            logger.warning(f"GPU compute capability {compute_capability} does not support Flash Attention 2")
            return False
            
    except ImportError:
        logger.warning("Flash Attention not installed. Please install with: pip install flash-attn")
        return False

# Configure Flash Attention
use_flash_attention = configure_flash_attention()
logger.info(f"Flash Attention Status: {'Enabled' if use_flash_attention else 'Disabled'}")

# %% [markdown]
"""
## Device Mapping Configuration
"""

# %%
def configure_device_mapping() -> dict:
    """
    Configure device mapping for GPU.
    Returns device map configuration for model loading.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This notebook requires a GPU to run.")
    
    # Use single GPU setup
    device_map = {"": 0}
    logger.info("Using single GPU setup")
    return device_map

# Configure device mapping
device_map = configure_device_mapping()
logger.info(f"Device Map: {device_map}")

# %% [markdown]
"""
## Model Download and Initialization
"""

# %%
def download_pixtral_model(model_id: str = "mistral-community/pixtral-12b", 
                         max_retries: int = 2,
                         retry_delay: int = 5) -> tuple:
    """
    Download the Pixtral model with retry logic and memory monitoring.
    
    Args:
        model_id: HuggingFace model ID
        max_retries: Maximum number of download attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        tuple: (model, processor) if successful
        
    Raises:
        RuntimeError: If download fails after max retries
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    import time
    import psutil
    
    def log_memory_usage(stage: str):
        """Log current memory usage"""
        gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        ram = psutil.virtual_memory().used / (1024**3)
        logger.info(f"Memory usage at {stage}: GPU={gpu_mem:.2f}GB, RAM={ram:.2f}GB")
    
    # Log initial memory usage
    log_memory_usage("start")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            
            # Configure model loading based on selected quantization
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            if quantization == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif quantization == "int8":
                # Simplified 8-bit config that works more reliably
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                model_kwargs["torch_dtype"] = torch.float16
            elif quantization == "int4":
                # Simplified 4-bit config
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["torch_dtype"] = torch.float16
            
            # Download model and processor
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Log final memory usage
            log_memory_usage("complete")
            
            logger.info("Model and processor downloaded successfully")
            return model, processor
            
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to download model after {max_retries} attempts: {str(e)}")

model, processor = download_pixtral_model()

# Clear CUDA cache after model loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()

logger.info("Model and processor ready for use")

# %% [markdown]
"""
## Prompt Selection
Select a prompt type for the model evaluation.
"""

# %%
# Run the prompt selection
selected_prompt_type = select_prompt()
logger.info(f"Selected prompt type: {selected_prompt_type}")

# The selected prompt is now stored in the global variable SELECTED_PROMPT
# This can be accessed in subsequent cells for model evaluation

# %% [markdown]
"""
## Configuration and Metadata Collection
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
            # Extract number from filename like "results-pixtral-bfloat16-3.json"
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

def collect_test_metadata(test_id: str) -> dict:
    """Collect metadata about the current test configuration."""
    config = yaml.safe_load(open(ROOT_DIR / "Deliverables-Code" / "config" / "pixtral.yaml", 'r'))
    
    # Get GPU information
    gpu_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    
    return {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": "Pixtral-12B",
            "version": "1.0",
            "model_id": "mistral-community/pixtral-12b",
            "model_type": "vision_language_model",
            "quantization": {
                "type": quantization,
                "config": {
                    "load_in_4bit": quantization == "int4",
                    "load_in_8bit": quantization == "int8",
                    "torch_dtype": quantization if quantization == "bfloat16" else None,
                    "bnb_4bit_compute_dtype": "torch.float16" if quantization == "int4" else None,
                    "bnb_4bit_quant_type": "nf4" if quantization == "int4" else None
                }
            },
            "device_info": {
                "device_map": device_map,
                "use_flash_attention": use_flash_attention,
                "gpu_memory_gb": round(gpu_props.total_memory / (1024**3), 2) if gpu_props else None,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}" if gpu_props else None
            }
        },
        "prompt_info": {
            "prompt_type": selected_prompt_type,
            "raw_text": SELECTED_PROMPT['prompts'][0]['text'],
            "formatted_text": format_prompt(SELECTED_PROMPT['prompts'][0]['text']),
            "special_tokens": config['model_params']['special_tokens']
        },
        "processing_config": {
            "inference_params": config['inference'],
            "image_preprocessing": {
                "max_size": config['model_params']['max_image_size'],
                "format": config['model_params']['image_format'],
                "resize_strategy": "maintain_aspect_ratio"
            }
        }
    }

def format_prompt(prompt_text: str) -> str:
    """Format the prompt using the Pixtral template."""
    config = yaml.safe_load(open(ROOT_DIR / "Deliverables-Code" / "config" / "pixtral.yaml", 'r'))
    special_tokens = config['model_params']['special_tokens']
    
    # Format the prompt with special tokens and image token
    formatted_prompt = f"{special_tokens[2]}\n{prompt_text}\n{special_tokens[0]}\n{special_tokens[1]}\n{special_tokens[3]}"
    return formatted_prompt

def load_and_process_image(image_path: str) -> Image.Image:
    """Load and process the image according to Pixtral specifications."""
    config = yaml.safe_load(open(ROOT_DIR / "Deliverables-Code" / "config" / "pixtral.yaml", 'r'))
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != config['model_params']['image_format']:
        image = image.convert(config['model_params']['image_format'])
    
    # Resize if needed
    max_size = config['model_params']['max_image_size']
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return image

# %% [markdown]
"""
## Single Image Test
Run the model on a single image using the selected prompt.
"""

# %%
def run_single_image_test():
    """Run the model on a single image with the selected prompt."""
    # Get the first .jpg file from data/images
    image_dir = ROOT_DIR / "Deliverables-Code" / "data" / "images" / "1_curated"
    image_files = list(image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError("No .jpg files found in data/images/1_curated directory")
    
    image_path = str(image_files[0])
    
    # Load and process image
    image = load_and_process_image(image_path)
    
    # Create a display version of the image with a max size of 800x800
    display_image = image.copy()
    max_display_size = (800, 800)
    display_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
    
    # Format the prompt
    prompt_text = SELECTED_PROMPT['prompts'][0]['text']
    formatted_prompt = format_prompt(prompt_text)
    
    # Display the image
    print("\nInput Image (resized for display):")
    display(display_image)
    
    # Display the prompt
    print("\nFormatted Prompt:")
    print("-" * 50)
    print(formatted_prompt)
    print("-" * 50)
    
    # Prepare model inputs using the original image
    inputs = processor(
        text=formatted_prompt,
        images=[image],  # Pass image as a list
        return_tensors="pt"
    )
    
    # Move inputs to device and convert to model's dtype
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Convert pixel_values to match model dtype for quantized models
    if 'pixel_values' in inputs:
        # Get the actual dtype of the vision model's first layer
        try:
            vision_dtype = next(model.vision_tower.parameters()).dtype
            if vision_dtype != torch.float32:
                inputs['pixel_values'] = inputs['pixel_values'].to(vision_dtype)
        except:
            # Fallback for quantized models
            if quantization in ["int8", "int4"]:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    
    # Get inference parameters from config
    config = yaml.safe_load(open(ROOT_DIR / "Deliverables-Code" / "config" / "pixtral.yaml", 'r'))
    inference_params = config['inference']
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inference_params['max_new_tokens'],
            do_sample=inference_params['do_sample'],
            temperature=inference_params['temperature'],
            top_k=inference_params['top_k'],
            top_p=inference_params['top_p']
        )
    
    # Decode and display response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("\nModel Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    return response

# Run the single image test
try:
    test_response = run_single_image_test()
except Exception as e:
    logger.error(f"Error during single image test: {str(e)}")
    raise

# %% [markdown]
"""
## Batch Processing - Data Collection Only
Run the model on all images and save raw results only.
"""

# %%
def save_incremental_results(results_file: Path, results: list, metadata: dict):
    """Save results incrementally to avoid losing progress."""
    complete_results = {
        "metadata": metadata,
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2)

def process_all_images(results_file: Path, metadata: dict) -> list:
    """Process all images in the data/images directory and collect raw responses only."""
    results = []
    image_dir = ROOT_DIR / "Deliverables-Code" / "data" / "images" / "1_curated"
    image_files = list(image_dir.glob("*.jpg"))
    
    if not image_files:
        raise FileNotFoundError("No .jpg files found in data/images/1_curated directory")
    
    for image_path in image_files:
        try:
            # Load and process image
            image = load_and_process_image(str(image_path))
            
            # Format the prompt
            prompt_text = SELECTED_PROMPT['prompts'][0]['text']
            formatted_prompt = format_prompt(prompt_text)
            
            # Prepare model inputs
            inputs = processor(
                text=formatted_prompt,
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to device and convert to model's dtype
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Convert pixel_values to match model dtype for quantized models
            if 'pixel_values' in inputs:
                # Get the actual dtype of the vision model's first layer
                try:
                    vision_dtype = next(model.vision_tower.parameters()).dtype
                    if vision_dtype != torch.float32:
                        inputs['pixel_values'] = inputs['pixel_values'].to(vision_dtype)
                except:
                    # Fallback for quantized models
                    if quantization in ["int8", "int4"]:
                        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            
            # Get inference parameters from config
            config = yaml.safe_load(open(ROOT_DIR / "Deliverables-Code" / "config" / "pixtral.yaml", 'r'))
            inference_params = config['inference']
            
            # Time the inference
            start_time = datetime.now()
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=inference_params['max_new_tokens'],
                    do_sample=inference_params['do_sample'],
                    temperature=inference_params['temperature'],
                    top_k=inference_params['top_k'],
                    top_p=inference_params['top_p']
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Create result entry with raw output only
            result = {
                "image_name": image_path.name,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "raw_output": {
                    "model_response": response,
                    "model_tokens_used": len(outputs[0]),
                    "generation_parameters_used": {
                        "max_new_tokens": inference_params['max_new_tokens'],
                        "temperature": inference_params['temperature'],
                        "top_k": inference_params['top_k'],
                        "top_p": inference_params['top_p']
                    }
                }
            }
            
            # Add to results
            results.append(result)
            
            # Save incremental results
            save_incremental_results(results_file, results, metadata)
            
            logger.info(f"Processed image: {image_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path.name}: {str(e)}")
            result = {
                "image_name": image_path.name,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": {
                    "type": "processing_error",
                    "message": str(e),
                    "stage": "inference"
                }
            }
            results.append(result)
            # Save incremental results even on error
            save_incremental_results(results_file, results, metadata)
    
    return results

def run_batch_test():
    """Run the model on all images and save raw results only."""
    try:
        # Generate filename with new naming convention
        test_id, results_file_path = generate_results_filename("pixtral", quantization, results_dir)
        results_file = Path(results_file_path)
        
        # Collect metadata with the test_id
        metadata = collect_test_metadata(test_id)
        
        logger.info(f"Starting Pixtral batch test with {quantization} quantization")
        logger.info(f"Results will be saved to: {results_file}")
        
        # Process all images with incremental saving
        results = process_all_images(results_file, metadata)
        
        logger.info(f"Batch test completed. Raw results saved to: {results_file}")
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Error during batch test: {str(e)}")
        raise

# Run the batch test
batch_results_file = run_batch_test()

# %% [markdown]
"""
## Analysis Functions - Data Processing Phase
Functions for analyzing raw model outputs and generating structured analysis reports.
"""

# %%
def extract_json_from_response(response: str) -> tuple[dict, str]:
    """
    Extract and parse JSON from raw model response.
    Returns tuple of (parsed_json, error_message).
    If successful, error_message will be empty.
    """
    try:
        # Find the second occurrence of {
        first_brace = response.find('{')
        if first_brace == -1:
            return None, "No JSON object found in response"
            
        second_brace = response.find('{', first_brace + 1)
        if second_brace == -1:
            return None, "No second JSON object found in response"
        
        # Find the matching closing brace
        brace_count = 1
        end_idx = second_brace + 1
        while brace_count > 0 and end_idx < len(response):
            if response[end_idx] == '{':
                brace_count += 1
            elif response[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
            
        if brace_count != 0:
            return None, "Unmatched braces in JSON object"
            
        # Extract and parse JSON
        json_str = response[second_brace:end_idx].strip()
        parsed_json = json.loads(json_str)
        
        # Validate structure
        if not isinstance(parsed_json, dict):
            return None, "Response is not a JSON object"
        
        # Validate required fields
        required_fields = {"work_order_number", "total_cost"}
        missing_fields = required_fields - set(parsed_json.keys())
        if missing_fields:
            return None, f"Missing required fields: {missing_fields}"
        
        return parsed_json, ""
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

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

def analyze_raw_results(results_file: str, ground_truth_file: str = None) -> dict:
    """Analyze raw model results and generate analysis report."""
    import pandas as pd
    
    # Set default ground truth file path
    if ground_truth_file is None:
        ground_truth_file = str(ROOT_DIR / "Deliverables-Code" / "data" / "images" / "metadata" / "ground_truth.csv")
    
    # Load results and ground truth
    with open(results_file, 'r') as f:
        raw_results = json.load(f)
    
    # Read ground truth with explicit string type for filename column
    ground_truth = pd.read_csv(ground_truth_file, dtype={'filename': str})
    
    # Initialize analysis structure
    analysis = {
        "source_results": results_file,
        "extraction_method": "json_parsing_v2",
        "ground_truth_file": ground_truth_file,
        "metadata": raw_results["metadata"],
        "summary": {
            "total_images": len(raw_results["results"]),
            "completed": 0,
            "errors": 0,
            "json_extraction_successful": 0,
            "work_order_accuracy": 0,
            "total_cost_accuracy": 0,
            "average_cer": 0
        },
        "error_categories": {
            "work_order": {},
            "total_cost": {}
        },
        "extracted_data": [],
        "performance_metrics": {}
    }
    
    # Process each result
    total_cer = 0
    work_order_matches = 0
    total_cost_matches = 0
    json_successful = 0
    
    for result in raw_results["results"]:
        # Get ground truth for this image - use filename directly for matching
        image_filename = result["image_name"]
        
        gt_row = ground_truth[ground_truth["filename"] == image_filename]
        
        if gt_row.empty:
            logger.warning(f"No ground truth found for image {image_filename}")
            continue
            
        gt_work_order = str(gt_row["work_order_number"].iloc[0]).strip()
        gt_total_cost = normalize_total_cost(str(gt_row["total"].iloc[0]))
        
        # Initialize extraction entry
        extraction_entry = {
            "image_name": result["image_name"],
            "status": result["status"],
            "raw_response": result.get("raw_output", {}).get("model_response", ""),
            "ground_truth": {
                "work_order_number": gt_work_order,
                "total_cost": gt_total_cost
            }
        }
        
        if result["status"] == "completed":
            analysis["summary"]["completed"] += 1
            
            # Extract JSON data from raw response
            raw_response = result["raw_output"]["model_response"]
            parsed_json, error_message = extract_json_from_response(raw_response)
            
            if parsed_json:
                json_successful += 1
                
                # Analyze work order
                pred_work_order = parsed_json.get("work_order_number", "")
                work_order_error = categorize_work_order_error(pred_work_order, gt_work_order)
                work_order_cer = calculate_cer(pred_work_order, gt_work_order)
                
                if work_order_error == "Exact Match":
                    work_order_matches += 1
                
                # Analyze total cost
                pred_total_cost = normalize_total_cost(parsed_json.get("total_cost", ""))
                total_cost_error = categorize_total_cost_error(pred_total_cost, gt_total_cost)
                
                if total_cost_error == "Numeric Match":
                    total_cost_matches += 1
                
                # Update extraction entry
                extraction_entry.update({
                    "extracted_data": {
                        "work_order_number": pred_work_order,
                        "total_cost": pred_total_cost
                    },
                    "extraction_confidence": {
                        "json_extraction_successful": True,
                        "parsing_method": "json_extraction",
                        "work_order_found": bool(pred_work_order),
                        "total_cost_found": bool(pred_total_cost),
                        "overall_confidence": 1.0 - work_order_cer
                    },
                    "performance": {
                        "work_order_error_category": work_order_error,
                        "total_cost_error_category": total_cost_error,
                        "work_order_cer": work_order_cer,
                        "work_order_correct": work_order_error == "Exact Match",
                        "total_cost_correct": total_cost_error == "Numeric Match"
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
                    "extraction_error": error_message,
                    "extraction_confidence": {
                        "json_extraction_successful": False,
                        "parsing_method": "json_extraction",
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
        analysis["summary"]["json_extraction_successful"] = json_successful
        analysis["summary"]["work_order_accuracy"] = work_order_matches / completed
        analysis["summary"]["total_cost_accuracy"] = total_cost_matches / completed
        analysis["summary"]["average_cer"] = total_cer / completed
        
        # Performance metrics
        analysis["performance_metrics"] = {
            "json_extraction_rate": json_successful / completed,
            "work_order_extraction_rate": work_order_matches / completed,
            "total_cost_extraction_rate": total_cost_matches / completed,
            "average_processing_time": sum(
                r.get("processing_time_seconds", 0) 
                for r in raw_results["results"] 
                if r["status"] == "completed"
            ) / completed
        }
    
    return analysis

def select_test_results_file() -> Path:
    """Allow user to select a test results file for analysis."""
    # Get all test result files
    results_dir_path = ROOT_DIR / "Deliverables-Code" / "results"
    result_files = list(results_dir_path.glob("results-*.json"))
    
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
    """Run analysis on raw results and generate comprehensive performance report."""
    try:
        # Get test results file
        results_file = select_test_results_file()
        
        # Generate analysis
        analysis = analyze_raw_results(str(results_file))
        
        # Create analysis directory if it doesn't exist
        analysis_dir = ROOT_DIR / "Deliverables-Code" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate analysis filename with same convention as results
        model_name = "pixtral"
        quantization_level = analysis['metadata']['model_info']['quantization']['type']
        
        # Find existing analysis files with the same model and quantization pattern
        pattern = f"analysis-{model_name}-{quantization_level}-*.json"
        existing_files = list(analysis_dir.glob(pattern))
        
        # Extract counter numbers from existing files
        counter_numbers = []
        for file in existing_files:
            try:
                # Extract number from filename like "analysis-pixtral-bfloat16-3.json"
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
        
        # Display summary
        print("\nPixtral Model Analysis Summary:")
        print("-" * 50)
        print(f"Total Images: {analysis['summary']['total_images']}")
        print(f"Completed: {analysis['summary']['completed']}")
        print(f"Errors: {analysis['summary']['errors']}")
        print(f"JSON Extraction Successful: {analysis['summary']['json_extraction_successful']}")
        print(f"Work Order Accuracy: {analysis['summary']['work_order_accuracy']:.2%}")
        print(f"Total Cost Accuracy: {analysis['summary']['total_cost_accuracy']:.2%}")
        print(f"Average CER: {analysis['summary']['average_cer']:.3f}")
        
        print("\nPerformance Metrics:")
        for metric, value in analysis['performance_metrics'].items():
            if 'rate' in metric:
                print(f"- {metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"- {metric.replace('_', ' ').title()}: {value:.2f}")
        
        print("\nWork Order Error Categories:")
        for category, count in analysis['error_categories']['work_order'].items():
            print(f"- {category}: {count}")
        
        print("\nTotal Cost Error Categories:")
        for category, count in analysis['error_categories']['total_cost'].items():
            print(f"- {category}: {count}")
        
        print(f"\nDetailed analysis saved to: {analysis_file}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

# %% [markdown]
"""
## Run Analysis
Generate and display analysis of raw model results.
"""

# %%
# Run the analysis
analysis_results = run_analysis()

