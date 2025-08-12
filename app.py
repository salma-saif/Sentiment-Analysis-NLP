import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np

# Load model and tokenizer once, cached by Streamlit
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
    model.eval()
    return tokenizer, model

# Call once at top level
tokenizer, model = load_model()

# Prediction function
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        pred_label = np.argmax(probs)
        return label_map[pred_label], probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "error", [0, 0, 0]

# Plot function
def plot_probs(probs):
    labels = ["negative", "neutral", "positive"]
    colors = ['red', 'gray', 'green']
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probs, color=colors)
    ax.set_ylim(0, 1)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, prob - 0.1, f"{prob:.2f}", ha='center', color='white', fontsize=12)
    st.pyplot(fig)

# Streamlit app layout and interaction
st.title("Sentiment Analysis Demo")
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        label, probs = predict_sentiment(user_input)
        st.write(f"**Predicted sentiment:** {label}")
        plot_probs(probs)
