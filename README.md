# C++ to Pseudocode — Transformer-Based Code Translation  

## Overview  
This project builds a **Transformer-based model** that converts **C++ code back into pseudocode** for better readability and understanding. The model is trained on the **SPoC dataset** and deployed via **Streamlit** for real-time usage.  

## Project Highlights  
- **Dataset:** SPoC dataset (reversed mapping: C++ as input, pseudocode as output).  
- **Architecture:** A **Transformer encoder-decoder** model with token embeddings and positional encodings.  
- **Training:** Implemented in **PyTorch**, with checkpointing and progress tracking.  
- **Deployment:** A **Streamlit web app** for real-time C++ to pseudocode conversion.  

## Installation  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/cpp-to-pseudocode.git
cd cpp-to-pseudocode
pip install -r requirements.txt
```

Usage
1. Train the Model
To train the model from scratch, run:

python train.py
2. Run the Streamlit App
To launch the interactive app, use:

streamlit run app.py
Dataset
The SPoC dataset is used, where the roles are reversed (C++ as input, pseudocode as output).

Model Architecture
Transformer-based encoder-decoder
Custom token embeddings
Positional encodings
Trained in PyTorch
Deployment
The trained model is hosted on Hugging Face Spaces, allowing users to input C++ code and receive a pseudocode explanation.

Challenges and Learnings
Handling the complexity of C++ syntax and converting it into simple pseudocode.
Addressing ambiguities in mapping code to pseudocode.
Managing training efficiency while preserving accuracy.
Future Improvements
Fine-tune on larger datasets for better results.
Implement explainability tools to highlight code logic.
Expand support to multiple programming languages.

Read the full article: https://medium.com/@sultanularfeen/converting-c-to-pseudocode-building-another-transformer-model-3986e189cb89

# Dataset Download
To train the model, you must download the SPoC dataset or any other dataset of your choice.
## SPoC Dataset Download: [https://drive.google.com/file/d/1AqL7FlyVwqNKLyFZOGGhvwhSMqx7hBYG/view?usp=sharing](https://drive.google.com/file/d/17v5rId0QeC-i6DmbiodrAKOOL8Wy3yhW/view?usp=drive_link)
Disclaimer: Due to the size being too large, the dataset is to be downloaded seperately.
