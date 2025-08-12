# Sentiment-Analysis-NLP
## Project Overview

This project implements a real-time sentiment analysis application that classifies customer feedback or product reviews into **Positive**, **Negative**, or **Neutral** sentiments. It leverages state-of-the-art transformer models to deliver accurate and insightful sentiment predictions.

## Dataset

The model is trained and evaluated using the **Twitter Entity Sentiment Analysis** dataset, publicly available on Kaggle:
[https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)

## Technologies & Tools

* **Transformers**: BERT-based models accessed through Hugging Face's Transformers library for powerful NLP capabilities.
* **PyTorch**: Backend deep learning framework used for model training and inference.
* **Scikit-learn**: For preprocessing, evaluation metrics, and additional ML utilities.
* **Streamlit**: Lightweight and interactive web framework for deploying the sentiment analysis model with a user-friendly interface.
* **Anaconda & VSCode**: Development environment and editor used for coding, debugging, and running the app.

## Features

* Real-time sentiment prediction from user-inputted text.
* Visualization of sentiment probabilities using bar charts.
* Easy-to-use web interface powered by Streamlit.
* Modular and extensible code structure.

## How to Run

1. Clone the repo.
2. Create and activate the Anaconda environment with required dependencies.
3. Run the Streamlit app locally using:

```bash
streamlit run app.py




