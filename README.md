# Optimizing-RoBERTa-Fine-Tuning-Strategies-for-Efficient-Sentiment-Classification

## Overview
This project focuses on optimizing the RoBERTa model for sentiment classification using the IMDB dataset. It compares multiple fine-tuning strategies, including Full Fine-Tuning, LoRA, QLoRA, and IA3, to evaluate their performance in terms of accuracy, training time, trainable parameters, and GPU memory usage. The implementation is provided in a Jupyter Notebook (`Optimizing_RoBERTa.ipynb`).

## Repository Contents
- **Optimizing_RoBERTa.ipynb**: Jupyter Notebook containing the implementation of data loading, preprocessing, and fine-tuning strategies for RoBERTa using the Hugging Face `transformers`, `datasets`, and `peft` libraries.
- **README.md**: This file, providing an overview and instructions for running the code.

## Prerequisites
To run the code, ensure the following dependencies are installed:
- Python 3.10+
- PyTorch
- transformers
- datasets
- peft
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

Install the required packages using:
```bash
pip install torch transformers datasets peft numpy matplotlib seaborn scikit-learn jupyter
```
**Note:** Ensure a GPU is available for faster computation, as the notebook is configured to use a GPU (e.g., NVIDIA GeForce RTX 3080).

## Dataset

**IMDB Dataset:** Sourced from Hugging Face (imdb), containing `25,000 training samples`, `25,000 test samples`, and `50,000 unsupervised samples` for sentiment classification (positive/negative reviews).

## Project Components
The Jupyter Notebook (`Optimizing_RoBERTa.ipynb`) includes the following components:

**Environment Setup:**
- Verifies GPU availability using PyTorch.
- Installs necessary libraries (e.g., scikit-learn for metrics).
- Imports essential libraries: PyTorch, transformers, datasets, peft, matplotlib, seaborn, and scikit-learn.


**Data Loading and Preprocessing:**
- Loads the IMDB dataset using the datasets library.
- Tokenizes the dataset using AutoTokenizer from bert-base-uncased (`used as a proxy for RoBERTa in this implementation`) with a maximum sequence length of `512`.
- Formats the dataset for PyTorch compatibility by removing unnecessary columns and setting the format to torch.

**Model Setup and Fine-Tuning:**

- `Full Fine-Tuning:` Fine-tunes the entire bert-base-uncased model using **AutoModelForSequenceClassification.**
- `LoRA Fine-Tuning:` Applies Low-Rank Adaptation (LoRA) using the peft library with configuration (**r=8, lora_alpha=16, targeting query and value modules**).
- `QLoRA and IA3:` Additional fine-tuning methods (partially implemented in the notebook, with results provided).

Configures training using TrainingArguments and Trainer from transformers with parameters like **learning rate (2e-5), batch size (16), and 3 epochs**.

**Evaluation:**

- Uses accuracy_score from scikit-learn to compute accuracy.
- Visualizes results (accuracy, training time, trainable parameters, GPU memory usage) across methods using matplotlib and seaborn bar plots.

## Results

The notebook compares four fine-tuning methods with the following results:

**Full Fine-Tuning:**

- Accuracy: 94.15%
- Training Time: 2657.18 seconds
- Trainable Parameters: ~250,000
- GPU Memory Usage: 9.8 GB


**LoRA:**
- Accuracy: 93.26%
- Training Time: 2306.86 seconds
- Trainable Parameters: 296,450
- GPU Memory Usage: 10.3 GB


**QLoRA:**
- Accuracy: 93.88%
- Training Time: 4029.82 seconds
- Trainable Parameters: 800,000
- GPU Memory Usage: 11.4 GB



**IA3:**
- Accuracy: 90.03%
- Training Time: 2366.51 seconds
- Trainable Parameters: 75,266
- GPU Memory Usage: 7.8 GB


## Expected Outcomes
- **Sentiment Classification:** Fine-tuned models classify IMDB reviews as positive or negative with high accuracy (up to 94.15% for Full Fine-Tuning).
- **Efficiency Comparison:** LoRA and IA3 reduce trainable parameters and memory usage compared to Full Fine-Tuning, with a slight trade-off in accuracy.
- **Visualization:** Bar plots comparing accuracy, training time, trainable parameters, and GPU memory usage across methods.


## Future Work
- Implement and fine-tune additional methods like QLoRA and IA3 fully within the notebook.
- Experiment with RoBERTa-specific models (e.g., roberta-base) instead of bert-base-uncased.
- Optimize hyperparameters (e.g., learning rate, lora_alpha, r) for better performance.
- Extend to other datasets or tasks (e.g., multi-class sentiment analysis).


## Author
- M Burhan ud din
