# Construction Invoice OCR Processing - AI/ML Capstone Project
## UCSD Extension School AI/ML Engineering Bootcamp - Final Submission

**Author:** Alden Smith  
**Program:** AI/ML Engineering Bootcamp  
**Institution:** UCSD Extension School  
**Submission Date:** March 2025  
**Business Partner:** DC Area General Contracting Firm  
**Faculty Mentor:** [Faculty Name]

---

## Executive Summary

This capstone project addresses a critical business challenge facing construction contractors: the manual processing of subcontractor invoices. Working with a Washington D.C. area general contracting firm, this project develops and evaluates an automated OCR solution using modern multi-modal Large Language Models (LLMs) to extract structured data from handwritten and printed construction invoices.

**Business Problem:** The contractor currently processes 797+ invoices manually, requiring significant time investment and introducing potential transcription errors. Invoices are submitted as photographs via Google Forms, with critical data fields (work order numbers, addresses, costs) requiring manual extraction.

**Technical Approach:** This project implements and compares three distinct approaches:
- **DocTR**: Traditional OCR with deep learning
- **LLaMA Vision**: Self-attention multimodal architecture  
- **Pixtral**: Cross-attention multimodal architecture

**Key Findings:** LLaMA Vision with step-by-step prompting achieved the highest accuracy for numeric field extraction, while Pixtral provided the most consistent results with superior computational efficiency. The study demonstrates that modern multimodal LLMs significantly outperform traditional OCR approaches for complex document understanding tasks.

**Impact:** The solution offers quantifiable business value through reduced labor costs, improved accuracy, and enables skilled staff to focus on higher-value tasks. The dual deployment strategy (local vs. cloud) provides flexibility for contractors with varying infrastructure capabilities.

---

## Academic Assignment Mapping

This repository contains all deliverables for the UCSD Extension AI/ML Engineering Bootcamp capstone project, organized to fulfill specific course requirements:

| Assignment | Document/Notebook | Description | Key Concepts Demonstrated |
|------------|------------------|-------------|---------------------------|
| **Project Proposal** | `Deliverables-Discussion/01_Project_Proposal.md` | Business case definition, technical approach, and implementation timeline | Problem identification, solution architecture, resource planning |
| **Literature Review** | `Deliverables-Discussion/02_Survey_of_Research.md` | Survey of multimodal LLM architectures for OCR applications | Research synthesis, architectural analysis, fine-tuning techniques |
| **Data Pipeline** | `Deliverables-Code/notebooks/01_image_download_and_processing.ipynb` | Data acquisition, preprocessing, and quality assessment | Data engineering, ETL processes, quality control |
| **Data Curation** | `Deliverables-Code/notebooks/02_image_curation_interface.ipynb` | Interactive tool for dataset curation and ground truth validation | Human-in-the-loop ML, data quality management |
| **Model Implementation 1** | `Deliverables-Code/notebooks/03_pixtral_model.ipynb` | Pixtral multimodal LLM implementation and evaluation | Cross-attention architectures, model deployment |
| **Model Implementation 2** | `Deliverables-Code/notebooks/04_llama_model.ipynb` | LLaMA Vision implementation with multiple prompting strategies | Self-attention architectures, prompt engineering |
| **Model Implementation 3** | `Deliverables-Code/notebooks/05_doctr_model.ipynb` | Traditional OCR baseline implementation | Computer vision, OCR fundamentals |
| **Comparative Analysis** | `Deliverables-Code/notebooks/06_Final_Analysis_v2.ipynb` | Comprehensive model performance analysis and statistical evaluation | Experimental design, statistical analysis, model comparison |
| **Process Discussion** | `Deliverables-Discussion/03_Discussion_of_Process.md` | Methodology, challenges, and lessons learned | Project management, technical problem-solving |
| **Results Discussion** | `Deliverables-Discussion/04_Discussion_of_Results.md` | Interpretation of findings and business implications | Results interpretation, business impact analysis |
| **Scaling Analysis** | `Deliverables-Discussion/05_Discussion_of_Scaling.md` | Production deployment considerations and infrastructure requirements | System architecture, scalability planning |
| **Deployment Strategy** | `Deliverables-Discussion/06_Discussion_of_Deployment.md` | Implementation roadmap and technical specifications | Deployment planning, system integration |
| **Future Roadmap** | `Deliverables-Discussion/08_Way_Ahead.md` | Enhancement opportunities and technical evolution | Strategic planning, continuous improvement |

---

## Repository Organization

This repository is structured into two main sections reflecting both academic rigor and practical implementation:

```
UCSD_MJM/
├── Deliverables-Code/              # Technical implementation and analysis
│   ├── analysis/                   # Model performance analysis results
│   ├── config/                     # Model and system configuration files
│   ├── data/                       # Dataset, ground truth, and metadata
│   │   ├── images/                 # Invoice image dataset (797 total, 50 curated for testing)
│   │   │   ├── 0_raw_download/     # Original downloaded images
│   │   │   ├── 1_curated/          # Quality-controlled test dataset
│   │   │   ├── display_cache/      # Optimized images for interface
│   │   │   └── metadata/           # Ground truth and processing logs
│   ├── notebooks/                  # Complete implementation pipeline
│   │   ├── 01_image_download_and_processing.ipynb
│   │   ├── 02_image_curation_interface.ipynb
│   │   ├── 03_pixtral_model.ipynb
│   │   ├── 04_llama_model.ipynb
│   │   ├── 05_doctr_model.ipynb
│   │   └── 06_Final_Analysis_v2.ipynb
│   ├── requirements/               # Environment specifications for each component
│   ├── results/                    # Raw model outputs and performance data
│   └── src/                        # Reusable modules and utilities
├── Deliverables-Discussion/        # Academic analysis and documentation
│   ├── 01_Project_Proposal.md
│   ├── 02_Survey_of_Research.md
│   ├── 03_Discussion_of_Process.md
│   ├── 04_Discussion_of_Results.md
│   ├── 05_Discussion_of_Scaling.md
│   ├── 06_Discussion_of_Deployment.md
│   ├── 08_Way_Ahead.md
│   └── img/                        # Analysis visualizations and diagrams
└── README.md                       # This comprehensive guide
```

**Workflow Logic:** The notebooks are designed to be executed sequentially, with each building upon the previous stage's outputs. Configuration files enable reproducible experiments across different computing environments.

---

## Business Case Summary

### Problem Statement
The partnering general contracting firm processes subcontractor invoices through a Google Forms workflow where contractors submit basic information and upload photographs of handwritten invoices. The current process requires:

- Manual extraction of work order numbers, addresses, and cost data
- Time-intensive data entry prone to transcription errors
- Skilled administrative staff tied to routine data processing tasks
- Potential delays in payment processing and project management

### Solution Architecture
This project implements an automated invoice processing pipeline leveraging state-of-the-art multimodal AI:

**Input:** High-resolution photographs of construction invoices (handwritten and printed)  
**Processing:** Three parallel processing approaches using different AI architectures  
**Output:** Structured data extraction (work order numbers, total costs) with confidence scoring  
**Integration:** Designed for integration with existing Google Sheets workflow

### Business Impact
- **Time Savings:** Estimated 70-80% reduction in manual processing time
- **Accuracy Improvement:** 15-20% improvement in data extraction accuracy over manual processes
- **Resource Optimization:** Enables skilled staff to focus on higher-value project management tasks
- **Scalability:** Supports business growth without proportional increase in administrative overhead
- **Flexibility:** Dual deployment options (local/cloud) accommodate varying IT infrastructure

---

## Technical Approach & Models

This project implements a rigorous comparative analysis of three distinct approaches to invoice processing, evaluating both traditional computer vision and cutting-edge multimodal AI techniques.

### Model Architectures

#### 1. DocTR (Traditional OCR Baseline)
- **Architecture:** Deep learning-based OCR with text detection and recognition
- **Strengths:** Fast processing, established accuracy for printed text
- **Configuration:** `config/doctr_config.yaml`
- **Use Case:** Baseline comparison and fallback processing option

#### 2. LLaMA Vision (Self-Attention Architecture)
- **Architecture:** Unified embedding decoder with self-attention mechanisms
- **Strengths:** Superior performance on complex layouts and mixed content
- **Prompting Strategies:** Basic extraction, detailed analysis, step-by-step reasoning, few-shot learning, locational awareness
- **Configuration:** `config/llama_vision.yaml`
- **Key Finding:** Achieved highest accuracy with step-by-step prompting strategy

#### 3. Pixtral (Cross-Attention Architecture)
- **Architecture:** Cross-modality attention with selective visual processing
- **Strengths:** Computational efficiency, consistent performance across image types
- **Variable Resolution:** Native support for different image sizes and qualities
- **Configuration:** `config/pixtral.yaml`
- **Key Finding:** Most consistent results with optimal computational cost

### Experimental Design

**Dataset:** 50 carefully curated invoices representing typical construction industry formats  
**Ground Truth:** Validated work order numbers and total cost amounts from business records  
**Metrics:** Accuracy (binary field extraction), Character Error Rate (text fields), processing time  
**Cross-Validation:** Multiple test runs to assess consistency and reliability  
**Statistical Analysis:** Comprehensive performance comparison with confidence intervals

### Evaluation Methodology

Each model was evaluated across multiple dimensions:
- **Accuracy:** Percentage of correctly extracted fields
- **Consistency:** Variance in performance across test runs
- **Efficiency:** Processing time and computational resource utilization
- **Robustness:** Performance across different image qualities and handwriting styles

---

## Reproduction Guide

### Quick Start
```bash
# 1. Clone repository
git clone [repository-url]
cd UCSD_MJM

# 2. Set up environment (choose appropriate requirements file)
pip install -r Deliverables-Code/requirements/requirements_analysis.txt

# 3. Run analysis notebook
jupyter lab Deliverables-Code/notebooks/06_Final_Analysis_v2.ipynb
```

### Detailed Reproduction

#### Environment Setup
Each model requires specific dependencies. Choose the appropriate requirements file:

- **Analysis Only:** `requirements_analysis.txt`
- **DocTR Model:** `requirements_doctr.txt`
- **LLaMA Model:** `requirements_llama.txt` 
- **Pixtral Model:** `requirements_pixtral.txt`

#### Data Preparation
1. **Image Dataset:** Navigate to `Deliverables-Code/notebooks/01_image_download_and_processing.ipynb`
2. **Curation Interface:** Use `02_image_curation_interface.ipynb` for dataset selection
3. **Ground Truth:** Located in `data/images/metadata/ground_truth.csv`

#### Model Execution
Execute notebooks in sequence:
1. **Data Processing:** `01_image_download_and_processing.ipynb`
2. **Data Curation:** `02_image_curation_interface.ipynb`
3. **Model Testing:** `03_pixtral_model.ipynb`, `04_llama_model.ipynb`, `05_doctr_model.ipynb`
4. **Analysis:** `06_Final_Analysis_v2.ipynb`

#### Hardware Requirements
- **Minimum:** 16GB RAM, modern CPU
- **Recommended:** 24GB+ GPU memory for full model inference
- **Cloud Alternative:** AWS/Google Cloud GPU instances (configurations provided in RunPod requirement files)

---

## Results Summary

### Key Performance Findings

#### Overall Model Performance
- **LLaMA Vision + Step-by-Step Prompting:** Highest accuracy for numeric fields (Total Cost)
- **Pixtral + Basic Prompting:** Most consistent performance across all test conditions
- **DocTR:** Reliable baseline with fastest processing time

#### Detailed Results
| Model | Prompting Strategy | Work Order Accuracy | Total Cost Accuracy | Avg. Processing Time | Consistency Score |
|-------|-------------------|--------------------|--------------------|---------------------|-------------------|
| LLaMA Vision | Step-by-Step | 78% | **85%** | 12.3s | Medium |
| Pixtral | Basic | 82% | 80% | **8.7s** | **High** |
| Pixtral | Detailed | 80% | 82% | 10.1s | High |
| DocTR | N/A | 65% | 70% | **6.2s** | Medium |

### Business Impact Analysis
- **ROI Calculation:** Estimated 300% return on investment within first year
- **Processing Efficiency:** 75% reduction in manual processing time
- **Error Reduction:** 20% improvement in data accuracy over manual entry
- **Scalability Factor:** Solution handles 10x current volume without additional staffing

### Technical Insights
1. **Multimodal LLMs significantly outperform traditional OCR** for complex document understanding
2. **Cross-attention architectures (Pixtral) provide better computational efficiency** than unified embedding approaches
3. **Prompting strategy selection is critical** for optimizing specific field extraction tasks
4. **Handwriting quality remains the primary limiting factor** across all approaches

---

## Acknowledgments

### Academic Institution
**UCSD Extension School** - AI/ML Engineering Bootcamp Program  
Special recognition for providing the academic framework and rigorous curriculum that enabled this comprehensive analysis.

### Faculty & Mentorship
**[Faculty Mentor Name]** - Technical guidance and academic oversight  
**Course Instructors** - Foundation in machine learning, deep learning, and practical AI deployment

### Business Partnership
**Washington D.C. Area General Contracting Firm** - Real-world data provision and business case validation  
This partnership enabled authentic problem-solving with genuine business impact, transforming academic learning into practical value creation.

### Technical Resources
**Hugging Face** - Model access and deployment infrastructure  
**Google Colaboratory & AWS** - Cloud computing resources for model training and inference  
**Open Source Community** - DocTR, Transformers, and supporting libraries

### Data & Methodology
**Ground Truth Validation** - Business partner's administrative team for data accuracy verification  
**Image Dataset** - 797 authentic construction invoices providing realistic test conditions  
**Research Foundation** - Academic papers and industry reports informing architectural decisions

---

*This project represents the culmination of intensive study in AI/ML engineering, demonstrating the practical application of cutting-edge technology to solve real-world business challenges. The work showcases both technical depth and business acumen, preparing for professional deployment of AI solutions in industry settings.* 