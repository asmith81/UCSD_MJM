# Optimized OCR Model for Construction Industry Invoice Processing with Dual Deployment Strategy

##  UCSD AI/ML Certificate Program Capstone Project Proposal

**Alden Smith**  
**20.6.1**  
**3 February 2025**

This real-world problem set focuses on optimizing a data-extraction workflow for a DC area general contracting firm.  The contractor  faces significant data entry challenges when processing subcontractor invoices. In the current workflow, subcontractors submit invoices to a Google Sheets database by means of an automated form.  The form handles only a basic identification number and total cost of the invoice while the details are captured via a photo of a handwritten invoice that is uploaded to the Google drive and linked to the row in the Google Sheet. Dispersion of funds based on these invoices is a manual process requiring significant time and introducing potential errors in data transcription.

This capstone project focuses on developing an optimized document parsing solution specifically trained on the invoice photos.. The core technical challenge is fine-tuning an existing OCR model to handle industry-specific terminology, varied handwriting styles, and inconsistent photo quality.  Additional work will miniaturize the model to ensure it runs efficiently enough for either local deployment or cost-effective cloud implementation.

The scope of this project encompasses:

* Fine-tuning an OCR model for construction invoice processing  
* Optimizing the model size and performance for dual deployment options  
* Designing a standardized SQL database to house extracted data  
* Creating efficient data pipelines from document capture to database storage

## Dataset Description

The project utilizes a substantial dataset of contractor invoices collected through the company's existing Google Forms workflow. The dataset consists of 797 invoice images totaling 1.81 GB, along with corresponding structured data from Google Sheets. 

The dataset is well-suited for model training due to its built-in validation capabilities. Each invoice image has three corresponding fields in the Google Sheets database that provide ground truth for model training and accuracy validation. Those fields are: Invoice Number, Address, and Total Cost. These fields 

Key dataset characteristics and challenges:

* Miix of handwritten and computer-generated invoices, providing variety for model training  
* Multi-modal data types including:  
  * Handwritten text  
  * Typed text  
  * Numerical values  
  * Address formats  
  * Bilingual content (English and Spanish)  
* Complete metadata through Google Sheets integration  
* Consistent file format and storage structure  
* Real-world data with natural variations in photo quality and lighting

The dataset is clean and ready for use, as it's been actively maintained through the company's existing business processes. All images are already digitized and linked to their corresponding database entries, eliminating the need for extensive preprocessing or data cleaning.

## Project Impact

This project addresses significant challenges across multiple dimensions, making it a compelling candidate for ML/AI implementation.

**Real-World Impact:**

* Addresses a common business problem faced by government contractors  
* Provides a test case for ML model deployment in small business settings  
* Demonstrates practical application of model optimization techniques with quantifiable business value through reduced labor costs and improved accuracy  
* Allows skilled staff to focus on higher-value tasks rather than manual data entry

**Technical Innovation:**

* Dual deployment strategy offers unique flexibility:  
  * Local deployment reduces computing costs for smaller contractors  
  * Cloud deployment enables scalability for larger operations  
  * Both options maintain data security and accessibility  
* Businesses can choose deployment methods based on their specific needs and infrastructure  
* Solution handles complex mixed-format data including multiple languages and data types

**Academic Significance:**

* Advances understanding of OCR model behavior with mixed-format documents  
* Explores critical trade-offs between model size and accuracy  
* Provides insights into practical deployment strategies for ML models in resource-constrained environments

The solution's value extends beyond the initial use case, as similar contractors throughout the construction industry face identical challenges with subcontractor invoice processing. The model's ability to handle both English and Spanish content further broadens its applicability. This combination of immediate practical value, technical innovation, and academic merit makes the project particularly significant for exploration within the course framework.

## Training and Deployment Resource Requirements

This project utilizes transfer learning with Vision Transformers (ViT) optimized for document understanding:

**Model Architecture:**

* Base Model: Pre-trained document understanding transformer (e.g., Donut or LayoutLM)  
* Custom Output Heads:  
  * Invoice Number (text output)  
  * Address (text output)  
  * Total Amount (numeric output)  
* Optimization Targets:  
  * Original model size: \~400-500MB  
  * Optimized target: \<100MB for local deployment

**Computational Requirements:**

* Initial Training Environment:  
  * Standard GPU instance (AWS p2.xlarge sufficient)  
  * 8GB GPU memory  
  * 20GB storage for dataset and model versions  
  * Training time: 18-24 hours per complete run  
  * Expected calendar time: 5-7 days including multiple training iterations  
* Model Miniaturization Phase:  
  * Same GPU instance as training  
  * 4-6GB GPU memory  
  * Processing time:  
    * Pruning: 2-3 hours  
    * Quantization: 1-2 hours  
    * Knowledge distillation: 4-6 hours  
  * Expected calendar time: 2-3 days including evaluation

**Local Deployment Requirements:**

* Minimum Specifications:  
  * 8GB RAM  
  * Single core minimum, dual core preferred  
  * 5GB free storage for:  
    * Optimized model (\~30-50MB)  
    * Support files (\~20MB)  
    * Structured data (\~50MB)  
    * Image archive (\~2GB)

**Cloud Deployment Alternative:**

* AWS SageMaker endpoint  
* Pay-per-inference pricing  
* Automated scaling

The architecture emphasizes minimal resource requirements while maintaining model accuracy, making it accessible for deployment on standard business hardware.

![][image1]  
**Fig. 1: Architecture / Workflow** \- Model Processing can occur on a local device or in a cloud environment.  Structured Data is returned to database that can be queried by any business intelligence tool.

## Implementation Timeline

The project will be completed in three main phases, with specific milestones and deliverables:

**Phase 1: Model Training (2 weeks)**

* Dataset preparation and validation  
  * Convert images to consistent format  
  * Verify ground truth data  
  * Create training/validation split  
* Initial model fine-tuning  
  * Configure base ViT model  
  * Implement custom output heads  
  * Train on full dataset  
* Model evaluation and iteration  
  * Measure accuracy against ground truth  
  * Refine model parameters  
  * Additional training runs as needed

**Phase 2: Model Optimization (1 week)**

* Model miniaturization  
  * Apply pruning techniques  
  * Implement quantization  
  * Validate accuracy maintenance  
* Performance testing  
  * Measure inference speed  
  * Verify memory usage  
  * Test edge cases

**Phase 3: Deployment Setup (1 week)**

* API development  
  * Implement webhook handler  
  * Create model interface  
  * Set up database connections  
* Deployment testing  
  * Verify local deployment  
  * Test cloud alternative  
  * Validate end-to-end flow

Total project duration: 4 weeks with buffer time for iterations and refinements.

[image1]: <data:image/png;base64,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>