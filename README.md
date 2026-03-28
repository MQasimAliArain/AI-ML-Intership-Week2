# AI & ML Internship

## Task 1: News Topic Classifier Using BERT

### Task Objective
To fine-tune a pre-trained Transformer model (**BERT**) to automatically categorize news headlines into four distinct topic categories: World, Sports, Business, and Sci/Tech.

### Dataset
* **Source:** AG News Dataset (via Hugging Face Datasets).
* **Features:** Raw News Headlines (Text).
* **Target:** Topic Category (4 classes).

### Models Used
* **BERT-base-uncased:** A state-of-the-art transformer model used for sequence classification through transfer learning.
* **Gradio Interface:** A web-based deployment tool used to create a live, interactive environment for real-time headline prediction.

### Methodology
* **Preprocessing:** Data was tokenized using the BERT fast tokenizer with truncation and padding to a 128-token limit.
* **Optimization:** Training was conducted on a 10,000-sample subset for 1 epoch, specifically optimized for **CPU-based environments** using the `accelerate` library.
* **Deployment:** A functional pipeline was integrated into a Gradio UI to allow for immediate inference and user interaction.

### Results
The model achieved high predictive performance, reaching over **90% accuracy** on the test subset. Evaluation via a **Normalized Confusion Matrix** confirmed the model's ability to distinguish between diverse categories, with particularly strong performance in Sports and World News. The live demo successfully demonstrates real-time classification with high confidence scores.

### Skills Gained
* NLP architecture using Transformers.
* Transfer learning and model fine-tuning.
* Evaluation metrics for multi-class text classification (Accuracy & F1-Score).
* Model deployment for live interaction.

# Task 4: Context-Aware Chatbot Using LangChain or RAG

## Task Objective
To build a high-performance **Retrieval-Augmented Generation (RAG)** pipeline that allows users to interact with PDF documents through a natural language interface, specifically optimized for comparative data analysis and ranking tasks.

## Dataset
* **Source:** User-provided PDF documents (placed in a local `/data` directory).
* **Features:** Unstructured text extracted from multi-page PDF files.
* **Target:** Context-aware responses and data-driven insights.

## Models & Technologies Used
* **Llama-3.3-70b-versatile:** A high-reasoning LLM accessed via **Groq** for near-instant inference.
* **all-MiniLM-L6-v2:** A lightweight **HuggingFace** transformer model used to generate semantic vector embeddings.
* **ChromaDB:** A vector database used for local storage and efficient similarity searches.
* **LangChain (LCEL):** The orchestration framework used to chain the retriever, prompt templates, and LLM together.

## Methodology
* **Preprocessing:** Documents are parsed using `PyPDFLoader` and split into 1000-character chunks with a 200-character overlap using `RecursiveCharacterTextSplitter`.
* **Optimization:** Implemented a **Senior Economic Data Analyst** system prompt to ensure the model performs numerical ranking and cross-country comparisons with high precision.
* **Retrieval:** Uses a similarity-based retriever configured to fetch the top 3 most relevant context windows ($k=3$).
* **UI Deployment:** An interactive, stylized chat interface built with `ipywidgets` and custom HTML/JavaScript to handle auto-scrolling and message history within Jupyter.

## Results
The system demonstrates exceptional performance in extracting specific data points from complex documents. By using a **temperature of 0**, the chatbot maintains high factual integrity, making it ideal for professional economic or technical analysis. The integration of **Groq** allows for complex 70B model reasoning with sub-second response times, significantly outperforming standard cloud-based APIs.

## Skills Gained
* **RAG Architecture:** Implementing the full lifecycle of document ingestion, embedding, and retrieval.
* **Prompt Engineering:** Designing specialized personas for structured data extraction and ranking.
* **Vector Databases:** Managing local vector stores with ChromaDB for semantic search.
* **Interactive UI Design:** Creating front-end components within Python environments using `ipywidgets`.


