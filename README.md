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
