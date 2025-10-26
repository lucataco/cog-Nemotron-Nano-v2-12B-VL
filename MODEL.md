Model Overview
Description:
Nemotron Nano v2 12B VL enables multi-image and video understanding, along with visual Q&A and summarization capabilities. Nemotron Nano v2 12B VL was developed by NVIDIA as a part of the Nemotron family of models.

It leverages Nemotron Nano V2 LLM backbone with reasoning and 128K context length to achieve state-of-the-art Pareto-optimal efficiency-accuracy. The model supports multiple modalities, including text, multiple images, and video. Specific use cases include visual Q&A and summarization, as well as being employed in multi-modal agents, Retrieval Augmented Generation (RAG), and data extraction tasks for documents like invoices, receipts, manuals, and financial/legal contracts. The video functionality includes 512*512 resolution for summarization and visual Q&A, with frame rates inversely proportional to video length (e.g., 2 minutes for 2 frames per second). .

This model is ready for non-commercial use.

License/Terms of Use
Use of this model is governed by the NVIDIA Software and Model Evaluation License Agreement

Deployment Geography:
North America (NAM)

Use Case:
Nemotron Nano 12B V2 VL is a model for multi-modal document intelligence. It would be used by individuals or businesses that need to process documents such as invoices, receipts, and manuals. The model is capable of handling multiple images of documents, up to four images at a resolution of 1k x 2k each, along with a long text prompt. The expected use is for tasks like summarization and Visual Question Answering (VQA). The model is also expected to have a significant advantage in throughput.

Release Date:
NGC [09/16/2025] via [URL]

Model Architecture:
Architecture Type: Transformer
Network Architecture: Vision Encoder: CRadioV2-H Language Encoder: Nemotron 5.5H -12B

** Number of model parameters: 12.6B

Input:
Input Type(s): Image, Video, Text Input Format: Image (jpg,png), Text (String) and mp4
Input Parameters: Image (2D),Video, Text (1D)
Other Properties Related to Input:

Input Images Supported: 4
Language Supported: English only
Input + Output Token: 128K
Minimum Resolution: 32 × 32 pixels
Maximum Resolution: Determined by a 12-tile layout constraint, with each tile being 512 × 512 pixels. This supports aspect ratios such as:
4 × 3 layout: up to 2048 × 1536 pixels
3 × 4 layout: up to 1536 × 2048 pixels
2 × 6 layout: up to 1024 × 3072 pixels
6 × 2 layout: up to 3072 × 1024 pixels
Other configurations allowed, provided total tiles ≤ 12
Channel Count: 3 channels (RGB)
Alpha Channel: Not supported (no transparency)
Output:
Output Type(s): Text
Output Format: String
Output Parameters: 1D
Other Properties Related to Output: Input + Output Token: 128K

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

Software Integration:
Runtime Engine(s):

[vLLM]
Supported Hardware Microarchitecture Compatibility:

NVIDIA A100
NVIDIA B200
NVIDIA H100/H200
Preferred/Supported Operating System(s):

[Linux]
The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

Model Version(s):
Nemotron Nano v2 12B VL

Training, Testing, and Evaluation Datasets:
Training / Evaluation Dataset :
Data Modalities

Total datasets: 265
Text-only datasets: 41
Text-and-image datasets: 165
Video-and-text datasets: 59
Sample counts/sizes

Total samples: 33'675'699
Total QA pairs: 79'562'702
Total images: 23'020'879
Total videos frames: 1'750'511
Total size [GB]: 27512.15
Properties ** Data Collection Method by dataset

[Hybrid: Human, Automated, Synthetic]
** Labeling Method by dataset

[Hybrid: Human, Automated, Synthetic]
Additionally, the dataset collection (for training and evaluation) consists of a mix of internal and public datasets designed for training and evaluation across various tasks. It includes:

Internal datasets built with public commercial images and internal labels, supporting tasks like conversation modeling and document analysis.
Public datasets sourced from publicly available images and annotations, adapted for tasks such as image captioning and visual question answering.
Synthetic datasets generated programmatically for specific tasks like tabular data understanding.
Specialized datasets for safety alignment, function calling, and domain-specific tasks (e.g., science diagrams, financial question answering).
Inference:
Acceleration Engine: [vLLM]
Test Hardware:

A100
B200
H100/H200
Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.
