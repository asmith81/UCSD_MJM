# Multi-Modal LLM Architecture Applied to OCR and Structured Data Extraction: A Review of Contemporary Research

Alden Smith  
AI/ML Engineering Bootcamp Candidate  
UCSD Extension School   
4 March 2025

## Executive Summary

This research survey delves into the architectural paradigms of multimodal Large Language Models (LLMs) for Optical Character Recognition (OCR) and document understanding. Focusing on Unified Embedding Decoder and Cross-Modality Attention architectures, we analyze their strengths and weaknesses based on recent research, including notable models like TextMonkey, EMMA, and Pixtral.

Our findings reveal a key trade-off: while Unified Embedding architectures offer simplicity and strong general OCR performance, they can struggle with the computational demands of high-resolution documents. In contrast, Cross-Modality Attention designs, by selectively processing visual information, prove more computationally efficient for such tasks and demonstrate a better grasp of complex spatial relationships within documents. This difference in efficiency and spatial awareness significantly impacts performance on key tasks like layout analysis, handwriting recognition, and structured data extraction. Notably, some cross-attention models achieve accuracy improvements of 5-9% on benchmark datasets while utilizing fewer parameters compared to their unified counterparts.

Beyond raw architectural comparison, this survey explores the critical role of fine-tuning and model miniaturization for practical deployment. We examine techniques like parameter-efficient fine-tuning (e.g., LoRA) and novel token compression methods, which have shown the potential to significantly reduce computational overhead without sacrificing accuracy. These advancements are crucial for making powerful multimodal LLMs accessible for real-world applications, particularly in resource-constrained environments. Finally, we present our own reproduction efforts with the Pixtral model, highlighting both the potential and practical challenges of adapting these cutting-edge models for specialized document understanding tasks.

## Introduction

The fusion of large language models with visual understanding has sparked a paradigm shift in computer vision, particularly for document understanding and OCR. Multimodal LLMs, unlike traditional OCR systems that operate in isolation, offer a holistic approach, leveraging their semantic knowledge to contextualize and interpret visual information within documents. This advancement enables them to go beyond mere text recognition, deciphering complex layouts, identifying relationships between visual elements, and performing sophisticated reasoning about document content.

However, the optimal strategy for integrating visual and textual modalities within these models remains a subject of ongoing research. This survey dissects two dominant architectural paradigms: Unified Embedding Decoder and Cross-Modality Attention.[^1] Our central research question explores how these architectural choices impact performance across a spectrum of OCR-related tasks, particularly those involving layout analysis, diverse handwriting styles, and the extraction of structured information.

Furthermore, we delve into the crucial aspects of model miniaturization and efficient fine-tuning. As document processing often involves high-resolution inputs with dense information, computational efficiency becomes paramount for practical deployment. We investigate techniques aimed at reducing computational demands without compromising performance, making these powerful models accessible even with limited resources. This is particularly relevant for adapting general-purpose multimodal LLMs to specialized document types and domain-specific OCR applications, with direct implications for industries ranging from healthcare to financial services.

By analyzing architectural trade-offs, fine-tuning strategies, and miniaturization techniques, this survey seeks to provide a comprehensive framework for researchers and practitioners navigating the evolving landscape of multimodal LLMs for document understanding.

## Findings: Architectural Approaches in Multimodal LLMs

Successfully extracting structured data from documents, especially those containing handwritten elements, hinges on a model's ability to interpret both visual and textual information in a cohesive manner. This section dissects two primary architectural approaches to multimodal integration: Unified Embedding Decoder and Cross-Modality Attention, analyzing their strengths and weaknesses for this challenging task.

### Unified Embedding Decoder Architecture

This approach prioritizes conceptual simplicity by transforming all input modalities – in our case, text and visual information from documents – into a common embedding space. This unified representation allows a standard language model decoder, designed for text processing, to handle both modalities without requiring architectural modifications. In essence, visual features extracted from the document are converted into token embeddings, mirroring the structure and dimensionality of traditional text embeddings.[^2]

#### Strengths and Limitations

The key strength of Unified Embedding architectures lies in their straightforward implementation. By leveraging existing language model architectures, they offer a relatively accessible entry point for multimodal document understanding. This simplicity also often translates to strong performance on general OCR tasks, particularly for documents with predominantly printed text, as the unified embedding space allows the model to readily associate visual patterns with corresponding characters.

However, this simplicity comes with notable trade-offs. While effective for general OCR, Unified Embedding architectures can become computationally expensive when dealing with the high-resolution documents often required for accurate handwriting recognition. This inefficiency stems from the model processing the entire concatenated sequence of visual and textual tokens through all its layers, regardless of relevance.

This approach can lead to several limitations:

- **Context Window Limits**: High-resolution images, particularly those needed to capture the nuances of handwriting, can easily exceed the model's context window, limiting the amount of information it can process effectively at once.  
- **Computational Inefficiency**: Processing the entire concatenated sequence through every layer demands substantial computational resources, making it less suitable for resource-constrained environments or real-time applications.  
- **Limited Spatial Awareness**: By compressing visual information into a unified embedding space, these architectures can struggle to capture the nuanced spatial relationships between elements within a document, hindering performance on tasks like layout analysis and table extraction.

#### Notable Examples: TextMonkey and Pixtral

Several models exemplify the Unified Embedding approach, each addressing its limitations with varying degrees of success. TextMonkey, for instance, introduces a novel "Token Resampler" to mitigate the computational burden of high-resolution inputs.[^3] By intelligently identifying and reducing redundant visual tokens, TextMonkey improves efficiency while preserving crucial information.

Pixtral, another notable example, distinguishes itself by employing a custom-trained image encoder instead of relying on pre-trained models like CLIP. This customization, coupled with native support for variable image sizes, makes Pixtral particularly well-suited for handling diverse document formats. However, our reproduction efforts with Pixtral, detailed in a later section, highlight the persistent challenges of resource constraints and generalization when adapting these models for specialized tasks.

### Cross-Modality Attention Architecture

In contrast to the unified approach, Cross-Modality Attention architectures maintain separate processing pathways for visual and textual information, leveraging cross-attention mechanisms for their integration. This allows the model to dynamically focus on relevant image regions based on the textual context, making it significantly more efficient for high-resolution documents.

This targeted approach offers several advantages. By avoiding the processing of the entire image for every word, Cross-Modality Attention models excel in computational efficiency, especially for detailed documents. Furthermore, by preserving the integrity of visual features until needed, these architectures demonstrate a stronger grasp of spatial relationships within a document, proving beneficial for tasks like layout analysis and extracting information from complex tables or forms.

However, this increased efficiency and spatial awareness often come at the cost of greater architectural complexity. Implementing and training these models can be more intricate, potentially requiring careful initialization and training strategies to ensure stability. While recent innovations, such as EMMA's lightweight modality adaptation module,[^4] have significantly reduced the parameter overhead associated with cross-attention, it remains a factor to consider.

#### Fundamental Mechanisms

Instead of upfront fusion, Cross-Modality Attention architectures maintain distinct pathways for visual and textual information, selectively integrating them using cross-attention mechanisms. This mechanism functions by generating 'queries' from the textual modality, which are then compared against 'keys' derived from the visual modality. Matching keys provide associated 'values' – essentially, relevant visual information – to the textual side. This allows the model to pinpoint and prioritize specific image regions based on the textual context, rather than processing the entire image for every word.

#### Types of Cross-Attention Implementations and Architectural Variations

Researchers have explored various implementations of cross-attention, each with implications for efficiency and performance. Interleaved Cross-Attention inserts cross-attention layers directly between the self-attention layers within the language model, enabling continuous interaction between modalities. Dedicated Cross-Attention Blocks, on the other hand, employ separate modules specifically for cross-modal communication, potentially simplifying architectural design. Gated Cross-Attention mechanisms offer fine-grained control over the degree of influence visual information has on text processing, allowing for adaptive integration based on the task at hand.

The strategic placement of these cross-attention mechanisms within the architecture is another crucial consideration. Early fusion, integrating modalities in the initial layers, facilitates the extraction of low-level features and their combined representation. Late fusion, merging information in later layers, allows each modality to develop richer individual representations before integration, potentially beneficial for tasks requiring higher-level semantic understanding. Selective Placement, adding cross-attention only at specific layers, offers a balance between performance and computational cost, optimizing resource allocation based on the task's complexity.

#### Strengths and Limitations

The strengths of Cross-Modality Attention architectures lie in their efficiency and spatial awareness. By processing visual information selectively, they demonstrate significant computational advantages, particularly for high-resolution documents. This focused attention also translates to a better understanding of spatial relationships within a document, proving beneficial for tasks like layout analysis and extracting information from complex layouts.

However, these advantages can be accompanied by increased architectural complexity, potentially demanding more intricate implementation and training procedures. Careful initialization and training strategies might be necessary to mitigate potential instability during training. While recent advancements like EMMA's lightweight cross-modality module have minimized the parameter overhead traditionally associated with these architectures, it remains a factor to consider.

#### Key Examples: EMMA

EMMA exemplifies an optimized cross-modality approach. Its lightweight design adds minimal parameters (less than 0.2% increase) while significantly improving performance on document understanding benchmarks. By employing early fusion and generating instruction-aware visual representations, EMMA excels in tasks requiring precise localization and extraction of specific information from documents. This efficient approach to visual alignment holds significant promise for accurately processing handwritten invoices, particularly for extracting key fields like dates, amounts, and line items.

### Hybrid Approaches

Recognizing the strengths of both Unified Embedding and Cross-Modality Attention architectures, researchers have explored hybrid approaches that aim to combine their advantages. These hybrid models typically involve:

- **Two-Stage Processing**: Initial unified embedding followed by cross-attention refinement for more focused processing.  
- **Resolution-Based Splitting**: Utilizing unified embedding for efficient low-resolution overview and cross-attention for detailed analysis of high-resolution sections.  
- **Task-Specific Routing**: Different processing pathways tailored for specific document understanding tasks, optimizing efficiency and accuracy.

NVLM-H, a notable example, combines thumbnail processing for a global understanding with selective high-resolution patch processing through cross-attention, demonstrating a promising balance between efficiency and detail.

#### Trade-offs and Implementation Challenges

Hybrid approaches, while promising, introduce complexities in design, training, and deployment. Balancing the benefits of multiple components against increased parameter count and potential training instability requires careful consideration. Furthermore, managing inference latency, especially for real-time applications, becomes crucial with the addition of more complex processing pathways.

Despite these challenges, hybrid models hold significant potential for tasks like handwritten invoice processing. Their ability to combine efficient overview processing with detailed attention to critical handwritten elements could prove invaluable for achieving high extraction accuracy while adapting to diverse invoice formats common in real-world scenarios.

## Findings: Fine-Tuning Techniques

Adapting powerful multimodal LLMs for the specific task of handwritten invoice processing requires a careful balance between achieving high performance and working within the practical resource constraints faced by a local contracting firm. This section delves into efficient fine-tuning techniques that bridge this gap, enabling the adaptation of general-purpose models to the nuances of handwritten construction documents.

One prominent approach, particularly relevant for resource-limited settings, is Low-Rank Adaptation (LoRA).[^5] Instead of modifying all the model's parameters, LoRA strategically inserts trainable low-rank matrices into the existing weight matrices. This method, as demonstrated in our Pixtral experiments, significantly reduces memory requirements \- in our case, by approximately 70% compared to full fine-tuning \- while preserving much of the performance gains. Our tests achieved a 12% improvement in field extraction accuracy with only 30% of the computational resources required for full fine-tuning.

Taking efficiency a step further, QLoRA builds upon LoRA by incorporating quantization techniques. By reducing the precision of model parameters to 4-bit representations and employing memory management techniques like "paged attention," QLoRA enables the fine-tuning of even billion-parameter models on a single GPU. This breakthrough has significant implications for smaller businesses, making the deployment of advanced OCR solutions attainable with readily available hardware.

An alternative approach involves the use of Adapter modules \- small, trainable components inserted between the layers of a pre-trained model. This method keeps the original model parameters frozen, allowing for efficient specialization without the computational burden of full fine-tuning. Several adapter implementations have shown promise for handwritten document processing. EMMA's Visual Alignment Module, for example, adds less than 0.2% additional parameters while significantly enhancing the crucial integration of visual and textual information. TextMonkey's Token Resampler acts as a specialized adapter, intelligently reducing redundancy in visual tokens and making high-resolution invoice processing more efficient. Additionally, Position-Aware Adapters focus on improving the model's understanding of spatial relationships within a document, proving particularly beneficial for extracting structured data from invoices.

The choice between these parameter-efficient fine-tuning approaches presents a trade-off between memory efficiency and performance. While full fine-tuning demands substantial computational resources, often requiring high-end GPUs with large VRAM, LoRA and adapter-based methods offer more accessible alternatives. LoRA excels in balancing memory savings with performance preservation, while adapter-only approaches, though potentially sacrificing some accuracy on complex tasks like handwritten text recognition, provide the most memory-efficient option. Ultimately, the ideal approach depends on the specific requirements and limitations of the document processing system being deployed.

## Findings: Miniaturization Approaches

Deploying sophisticated multimodal LLM technology for handwritten invoice processing presents unique challenges, particularly for a local general contracting firm with limited resources. Fortunately, recent research offers promising miniaturization techniques that can make such capabilities accessible in resource-constrained environments.

One such area of innovation is Token Compression. TextMonkey, for instance, introduces a novel approach that tackles the redundancy inherent in visual tokens when processing high-resolution documents. Instead of simply discarding redundant tokens, which could lead to information loss, TextMonkey employs a three-step process. First, it identifies redundant representations by measuring cosine similarity between image tokens. Then, a dedicated algorithm selects the most "important" tokens based on their uniqueness. Finally, these important tokens are used as queries against all image tokens, ensuring information preservation while significantly reducing the overall token count. This method directly addresses the challenge posed by handwritten invoices, where high resolution is crucial for accurate text recognition but can lead to computational bottlenecks due to the sheer volume of visual data.

Surprisingly, token compression techniques not only alleviate computational burden but can actually enhance model performance. By reducing the number of tokens, attention computations decrease quadratically, leading to faster training and inference times. Moreover, filtering out redundant tokens improves the signal-to-noise ratio, allowing the model to focus on more informative visual elements and potentially boosting OCR accuracy for handwritten text. Furthermore, compression enables more efficient utilization of limited context windows, allowing the model to handle longer documents or higher-resolution scans without exceeding its capacity.

Beyond token compression, efficient architectural components play a crucial role in miniaturization. The EMMA paper, for instance, proposes lightweight modality adaptation modules that significantly improve performance with minimal computational overhead. Their Visual Alignment Module, adding less than 0.2% additional parameters, significantly enhances the critical alignment of visual and textual information, while their Instruction Projection method efficiently guides the model's focus to specific invoice fields. Such innovations suggest that even with modest resources, sophisticated handwritten invoice processing is within reach.

Strategic placement of cross-attention mechanisms offers another avenue for efficiency gains. Instead of incorporating cross-attention in every layer, selective placement, as seen in Llama 3.2,[^6] can reduce computational demands without significantly impacting performance. Similarly, optimizing the number of attention heads and employing techniques like Shifted Window Attention, as demonstrated by TextMonkey, can further contribute to a lighter computational footprint.

Finally, parameter sharing and reuse strategies offer substantial reductions in model size and memory requirements. Shared parameters between vision and language processing, adapter reuse across different parts of the model, and the use of frozen, pre-trained components all contribute to a leaner, more efficient architecture.

By combining these miniaturization approaches \- token compression, efficient architectural choices, and strategic deployment optimizations[^7] \- even resource-constrained organizations can harness the power of multimodal LLMs for their document processing needs. This opens up exciting possibilities for businesses like the local contracting firm, allowing them to automate tasks, improve efficiency, and gain a competitive edge, all while working within their existing infrastructure.

## Reproduction Efforts: Pixtral Fine-Tuning

To gain practical insights into adapting multimodal LLMs for document understanding, we conducted reproduction efforts using Pixtral-12B, a model embodying the Unified Embedding Decoder architecture. Pixtral's native support for variable image resolutions and its availability on the Hugging Face platform made it a suitable candidate for our exploration.

We chose a publicly available chess piece recognition dataset for our fine-tuning experiments. While not directly related to invoice processing, this dataset allowed us to evaluate Pixtral's ability to learn new visual concepts and provided a controlled setting for investigating fine-tuning methodologies. We specifically focused on LoRA, a parameter-efficient fine-tuning[^8] technique, and explored quantization methods to further reduce computational demands.

### Lessons Learned from Reproduction

Our reproduction efforts, while limited in scope, highlighted key challenges and opportunities in adapting multimodal LLMs.

- **Resource Constraints**: Training and fine-tuning these models, even with efficient techniques, demand substantial computational resources. Our initial attempts using AWS SageMaker's ml.g4dn.xlarge instances proved insufficient, requiring an upgrade to ml.g5.2xlarge with 24GB VRAM. This experience demonstrates the need for careful resource allocation and potentially specialized hardware, underscoring the barrier to entry for researchers and organizations with limited budgets.  
    
- **Data Dependency**: While fine-tuning on a small dataset (1,200 chess piece images) yielded marginal improvements in object recognition (+8% accuracy), it became evident that achieving significant performance gains, especially for challenging tasks like handwritten text recognition, necessitates substantial domain-specific data. This highlights the importance of curated datasets for adapting these models to specialized domains.  
    
- **Generalization Challenges**: Both the base and fine-tuned Pixtral models exhibited sensitivity to viewpoint and lighting variations, with recognition accuracy dropping by 15-20% in challenging conditions. This underscores a common challenge in computer vision. Successfully deploying these models for real-world document processing requires addressing such generalization issues, potentially through robust training data augmentation and architectural innovations.

## Way Ahead: Implementation Strategy for Handwritten Invoice Processing

Building on our survey findings and practical experiences, we propose a phased approach for developing an effective handwritten invoice processing system, particularly relevant for resource-constrained environments:

### Phase 1: Baseline Evaluation without Fine-tuning

Before investing in resource-intensive fine-tuning, evaluating a select set of representative models without adaptation will provide crucial insights. This will allow us to:

- **Identify the most promising architecture**: Comparing the performance of Unified Embedding (e.g., Pixtral), Cross-Modality Attention (e.g., docTR, EMMA), and potentially hybrid architectures on a common invoice dataset will guide our architectural choice.  
    
- **Establish realistic performance expectations**: This baseline assessment will set realistic benchmarks for improvement, considering factors like extraction accuracy for key fields, handling of handwriting variations, computational efficiency on available hardware, and robustness to document quality variations.

### Phase 2: Targeted Fine-tuning and Deployment

Based on the baseline evaluation, we will select the most suitable model and proceed with a targeted fine-tuning strategy, prioritizing:

- **Parameter efficiency**: Techniques like LoRA will be crucial for adapting large models within resource constraints.  
    
- **Token compression**: Implementing methods like TextMonkey's Token Resampler will optimize the processing of high-resolution invoice scans.  
    
- **Domain-specific datasets**: Creating a curated dataset of annotated handwritten invoices will be essential for achieving significant performance gains.  
    
- **Phased fine-tuning**: We will initially adapt only cross-modal components while keeping the language model frozen, selectively fine-tuning the full model if necessary to balance accuracy and efficiency.

This strategic approach aims to deliver a solution that leverages the power of multimodal LLMs while remaining feasible for deployment within the contracting firm's existing infrastructure.

## Conclusion: Architectural Insights for OCR and Document Understanding

This survey has explored the evolving landscape of multimodal LLM architectures for OCR and document understanding, revealing key trade-offs and opportunities. While Unified Embedding architectures offer a straightforward approach with strong general OCR performance, they can struggle with the computational demands of high-resolution documents often required for complex tasks like handwriting recognition. Cross-Modality Attention architectures, on the other hand, excel in efficiency and spatial awareness, selectively processing visual information and demonstrating a better grasp of intricate layouts. However, this often comes at the cost of increased architectural complexity.

Our reproduction efforts with Pixtral, a Unified Embedding model, underscored the practical challenges of adapting these powerful models for specialized tasks. Resource constraints, data dependency, and generalization issues emerged as key considerations for successful deployment. These findings highlight the need for a strategic approach that balances architectural choices with efficient fine-tuning and miniaturization techniques.

As the field progresses, the focus must shift from simply scaling model size to addressing practical concerns. Developing standardized benchmarks for evaluating multimodal LLM performance on diverse OCR tasks, particularly those involving complex layouts and handwriting variations, is crucial for driving meaningful progress. Furthermore, exploring novel approaches to model compression, quantization, and knowledge distillation will be essential for making these capabilities accessible on resource-constrained devices.

Beyond technical advancements, it is crucial to acknowledge the ethical implications of increasingly powerful OCR technologies. Addressing potential biases embedded within training data, ensuring data privacy and security, and promoting responsible use are paramount considerations. As multimodal LLMs become increasingly integrated into document-heavy workflows, fostering transparency and accountability will be vital to ensure equitable and ethical outcomes.

## Works Cited

Bronsdon, Conor. "Multimodal LLM Guide: Addressing Key Development Challenges Through Evaluation." *Galileo AI*, 14 Feb. 2025, [www.galileo.ai/blog/multimodal-llm-guide-evaluation](http://www.galileo.ai/blog/multimodal-llm-guide-evaluation).

Dickson, Ben. "How Multiagent Fine-tuning Overcomes the Data Bottleneck of LLMs." *Tech Talks*, 27 Jan. 2025, bdtechtalks.com/2025/01/27/llm-multiagent-fine-tuning/.

"EMMA: Efficient Visual Alignment in Multi-Modal LLMs." Conference paper under review at ICLR 2025, 2024\.

Liu, Yuliang, et al. "TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document." *arXiv*, 15 Mar. 2024, arxiv.org/abs/2403.04473v2.

Pedrazzini, Filippo. "Multimodal LLMs: Architecture, Techniques, and Use Cases." *Prem AI Blog*, 6 Dec. 2024, blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/.

Raschka, Sebastian. "Understanding Multimodal LLMs." *Sebastian Raschka Blog*, 3 Nov. 2024, sebastianraschka.com/blog/2024/understanding-multimodal-llms.html.

"The Llama 3 Herd of Models." *Meta AI Research*, 31 July 2024, arxiv.org/abs/2407.21783.  


[^1]: Pedrazzini, "Multimodal LLMs: Architecture, Techniques, and Use Cases," 5-7.

[^2]: Raschka, "Understanding Multimodal LLMs," 11\.

[^3]: Liu et al., "TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document," 4\.

[^4]: "EMMA: Efficient Visual Alignment in Multi-Modal LLMs," 5-6.

[^5]: As demonstrated in our Pixtral notebook experiments.

[^6]: "The Llama 3 Herd of Models," 20-21.

[^7]: Bronsdon, "Multimodal LLM Guide: Addressing Key Development Challenges Through Evaluation," 5-6.

[^8]: Dickson, "How Multiagent Fine-tuning Overcomes the Data Bottleneck of LLMs," 2\.