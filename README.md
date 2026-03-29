<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:0f3460,100:16213e&height=220&section=header&text=Machine%20Learning%20Projects&fontSize=48&fontColor=58a6ff&animation=fadeIn&fontAlignY=38&desc=A%20curated%20collection%20of%20ML%20implementations%20%7C%20Quach%20Thien%20Phu&descAlignY=58&descColor=8b949e" />

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=58A6FF&center=true&vCenter=true&width=700&lines=Deep+Learning+%7C+Computer+Vision+%7C+NLP;Generative+AI+%7C+Agentic+AI+%7C+LLMs;From+scratch+implementations+to+production+apps)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

> A comprehensive repository spanning **deep learning fundamentals**, **computer vision**, **NLP**, **generative AI**, and **multi-agent systems** — built and explored by [Quach Thien Phu](https://github.com/quachphu) as part of his ML engineering journey at CSULB.

</div>

---

## Project Index
 
| # | Project | Category | Key Concepts | Tools |
|---|---------|----------|--------------|-------|
| 01 | [Building Your Deep Neural Network](#1-building-your-deep-neural-network) | Deep Learning Fundamentals | Forward/Backprop, Weights Init | NumPy, Python |
| 02 | [Initialization Deep Learning](#2-initialization-deep-learning) | Deep Learning Fundamentals | Zero, Random, He Init | NumPy |
| 03 | [Optimization Algorithms](#3-optimization-algorithms) | Deep Learning Fundamentals | SGD, Momentum, Adam, RMSProp | NumPy |
| 04 | [Regularization](#4-regularization) | Deep Learning Fundamentals | L2, Dropout, Early Stopping | NumPy |
| 05 | [Gradient Checking](#5-gradient-checking--fraud-detection) | Deep Learning Fundamentals | Numerical gradient verification | NumPy |
| 06 | [Planar Data Classification](#6-planar-data-classification) | Neural Networks | Binary classification, decision boundaries | NumPy, Matplotlib |
| 07 | [Rain Prediction in Australia](#7-rain-prediction-in-australia) | Classical ML | EDA, Feature Eng, Classification | Sklearn, Pandas |
| 08 | [Convolutional Neural Network](#8-convolutional-neural-network-application) | Computer Vision | Conv layers, Pooling, Filters | TensorFlow/Keras |
| 09 | [Image Segmentation (U-Net V2)](#9-image-segmentation-u-net-v2) | Computer Vision | Encoder-Decoder, Skip Connections | TensorFlow/Keras |
| 10 | [Sign Language Digits](#10-tensorflow-sign-language-digits) | Computer Vision | Multi-class classification | TensorFlow |
| 11 | [MobileNet — Alpaca Classification](#11-mobilenet--alpaca-classification) | Transfer Learning | Fine-tuning, MobileNetV2 | TensorFlow/Keras |
| 12 | [Face Recognition](#12-face-recognition) | Computer Vision | FaceNet, Triplet Loss, Embeddings | TensorFlow, OpenCV |
| 13 | [Transformer (from Scratch)](#13-transformer-from-scratch) | NLP / Deep Learning | Attention, Positional Encoding, BERT | PyTorch |
| 14 | [LLM ChatBot](#14-llm-chatbot) | Generative AI | Prompt Engineering, RAG basics | LangChain, OpenAI |
| 15 | [Search Engine](#15-search-engine) | Generative AI / IR | Vector Search, Embeddings | LangChain, FAISS |
| 16 | [Agentic AI Hackathon](#16-agentic-ai-hackathon) | Agentic AI | Multi-agent orchestration | Fetch.ai, uAgents |
| 17 | [Crew AI](#17-crew-ai) | Agentic AI | Role-based agents, task delegation | CrewAI |
| 18 | [FellowAI](#18-fellowai) | Agentic AI | Agent pipelines | LangGraph |
| 19 | [NVIDIA — Accelerated Computing](#19-nvidia--accelerated-computing) | Systems / Optimization | CUDA, GPU programming | NVIDIA toolkit |
 
---
 
## Deep Learning Fundamentals
 
### 1. Building Your Deep Neural Network
> Implementing a deep L-layer neural network **from scratch** using only NumPy.
 
- Built forward propagation, backward propagation, and parameter updates without any ML framework
- Implemented L-layer architecture with ReLU hidden layers and sigmoid output
- Applied to cat image binary classification task
- **Key insight**: Understanding the math behind gradient flow in deep networks
 
```
Input → [LINEAR → RELU] × (L-1) → LINEAR → SIGMOID → Output
```
 
---
 
### 2. Initialization Deep Learning
> Exploring how weight initialization affects training stability and speed.
 
| Method | Problem | When to use |
|--------|---------|------------|
| Zero Init | Symmetry breaking fails | Never |
| Random Init | Exploding/vanishing gradients | Small networks |
| He Init | Optimal for ReLU layers | Deep networks |
 
---
 
### 3. Optimization Algorithms
> Implementing and comparing gradient descent variants from scratch.
 
- **Mini-batch GD** — balances speed vs. accuracy
- **Momentum** — dampens oscillations, accelerates convergence
- **RMSProp** — adaptive learning rates per parameter
- **Adam** — combines Momentum + RMSProp, the go-to optimizer
 
---
 
### 4. Regularization
> Techniques to reduce overfitting in deep neural networks.
 
- L2 regularization (weight decay) — penalizes large weights
- Dropout — randomly deactivates neurons during training
- Early stopping — monitors validation loss to stop training
- **Result**: Significant reduction in overfitting on test datasets
 
---
 
### 5. Gradient Checking — Fraud Detection
> Verifying backpropagation is implemented correctly using numerical gradient approximation.
 
- Applies fraud detection as the domain problem
- Computes numerical gradients via finite differences and compares with analytical gradients
- **Use case**: Debugging custom loss functions and layers
 
---
 
### 6. Planar Data Classification
> Training a shallow neural network to learn non-linear decision boundaries.
 
- Visualized how hidden layer size affects decision boundary complexity
- Compared logistic regression vs neural network on non-linear data
- **Key takeaway**: Even 1 hidden layer can approximate complex functions
 
---
 
## Classical ML & EDA
 
### 7. Rain Prediction in Australia
> End-to-end ML pipeline to predict next-day rainfall using weather data.
 
- Extensive EDA with Matplotlib and Seaborn visualizations
- Feature engineering, handling missing values, encoding categoricals
- Trained and compared Logistic Regression, Decision Tree, Random Forest, XGBoost
- Evaluated with ROC-AUC, Precision-Recall curves
 
---
 
## Computer Vision
 
### 8. Convolutional Neural Network Application
> Implementing CNNs for image classification tasks.
 
- Built Conv → Pool → Flatten → Dense architectures in TensorFlow/Keras
- Explored filter visualizations to understand what each layer learns
- Applied data augmentation to improve generalization
 
---
 
### 9. Image Segmentation — U-Net V2
> Semantic segmentation using the U-Net architecture.
 
- Encoder path: progressively downsamples with conv + maxpool
- Decoder path: upsamples with transposed convolutions
- **Skip connections**: preserve spatial information lost during downsampling
- Applied to biomedical image segmentation
 
```
Input Image
     ↓
[Encoder: Conv → MaxPool] × 4   ← Captures "what"
     ↓
   Bottleneck
     ↓
[Decoder: UpConv + Skip] × 4    ← Recovers "where"
     ↓
Segmentation Mask
```
 
---
 
### 10. TensorFlow Sign Language Digits
> Multi-class classification of sign language digit images (0–9).
 
- Built CNN in TensorFlow from raw image data
- Preprocessing pipeline: normalization, one-hot encoding
- Achieved strong accuracy on 10-class gesture recognition
 
---
 
### 11. MobileNet — Alpaca Classification
> Transfer learning with MobileNetV2 for custom image classification.
 
- Loaded pre-trained MobileNetV2 weights (trained on ImageNet)
- Froze base layers, added custom classification head
- Fine-tuned top layers on the alpaca dataset
- **Key insight**: Transfer learning achieves high accuracy with minimal data
 
---
 
### 12. Face Recognition
> Building a face recognition system using deep embeddings.
 
- Implemented FaceNet architecture producing 128-dim face embeddings
- **Triplet Loss**: trains model to minimize distance between same-identity pairs
- One-shot learning: recognizes faces from a single reference image
- Real-time inference with OpenCV face detection
 
---
 
## NLP & Sequence Models
 
### 13. Transformer (from Scratch)
> Full Transformer architecture implementation in PyTorch.
 
```python
# Core components implemented:
├── Multi-Head Self-Attention
├── Positional Encoding
├── Feed-Forward Networks
├── Layer Normalization
├── Encoder & Decoder stacks
└── Masked attention for autoregressive decoding
```
 
- Built encoder-decoder Transformer following "Attention Is All You Need"
- Explored BERT-style pre-training concepts
- **Foundation for understanding GPT, BERT, T5, and modern LLMs**
 
---
 
## Generative AI
 
### 14. LLM ChatBot
> Conversational AI chatbot powered by large language models.
 
- Integrated OpenAI API for response generation
- Built with LangChain for conversation memory and chain orchestration
- Prompt engineering for consistent persona and context retention
- Deployed as an interactive interface
 
---
 
### 15. Search Engine
> Semantic search engine using vector embeddings.
 
- Generates dense embeddings for documents using sentence transformers
- Indexes with FAISS for fast approximate nearest-neighbor search
- Returns semantically relevant results (not just keyword matches)
- **Stack**: LangChain + FAISS + OpenAI Embeddings
 
---
 
## Agentic AI
 
### 16. Agentic AI Hackathon
> Multi-agent system built at a hackathon using Fetch.ai's uAgents framework.
 
- Designed specialized agents with distinct roles and communication protocols
- Agents coordinate to complete complex multi-step tasks autonomously
- Explored decentralized agent architecture
 
---
 
### 17. Crew AI
> Role-based multi-agent collaboration using CrewAI framework.
 
- Defined AI agents with specific roles (researcher, writer, analyst)
- Orchestrated task delegation between agents
- Built end-to-end pipelines with autonomous decision-making
 
---
 
### 18. FellowAI
> Advanced agent pipelines using LangGraph for stateful workflows.
 
- Built directed acyclic graph (DAG) agent workflows
- Implemented conditional routing and human-in-the-loop checkpoints
- Explored state management across long-running agentic tasks
 
---
 
### 19. NVIDIA — Accelerated Computing
> GPU-accelerated ML using NVIDIA CUDA and optimized libraries.
 
- Explored CUDA programming fundamentals for parallel computation
- Applied cuDNN-accelerated deep learning operations
- Performance benchmarking: GPU vs CPU training times
 
---
 
## Learning Roadmap
 
```
Phase 1 — Foundations          Phase 2 — Vision & NLP         Phase 3 — Modern AI
────────────────────           ───────────────────────         ──────────────────────
[x] Neural nets from scratch   [x] CNNs & image segmentation   [x] Transformers & LLMs
[x] Optimization algorithms    [x] Transfer learning           [x] RAG & vector search
[x] Regularization             [x] Face recognition            [x] Agentic frameworks
[x] Gradient checking          [x] Sign language detection     [x] Multi-agent systems
[x] Classical ML + EDA         [x] Object classification       [ ] CGEV thesis (ongoing)
```
 
---
 
## How to Run
 
```bash
# Clone the repository
git clone https://github.com/quachphu/Machine-Learning-Project.git
cd Machine-Learning-Project
 
# Install dependencies
pip install -r requirements.txt
 
# Launch any notebook
jupyter notebook <ProjectFolder>/<notebook>.ipynb
```
 
**Common dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow keras torch torchvision
pip install langchain openai faiss-cpu opencv-python
pip install crewai fetch-uagents langgraph
```
 
---
 
## Connect
 
<div align="center">
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Quach%20Thien%20Phu-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/quachthienphu)
[![GitHub](https://img.shields.io/badge/GitHub-quachphu-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/quachphu)
[![Portfolio](https://img.shields.io/badge/VoiceBridge-voicebridges.net-FF6B6B?style=for-the-badge&logo=vercel&logoColor=white)](https://voicebridges.net)
 
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:0f3460,100:0d1117&height=120&section=footer" />
 
</div>
