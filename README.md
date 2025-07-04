
# 📰 LLM News Classifier with Web Scraping & Fine-Tuning

## 🚀 Introduction

This project walks you through the **end-to-end pipeline** of collecting real-world data by scraping websites and using it to **fine-tune a Large Language Model (LLM)** for **news article classification**. It combines **Decodo** for scraping, **BeautifulSoup** for parsing HTML, and **Hugging Face Transformers** for training and inference.

### 🎯 What You'll Learn

- 🌐 How to crawl and scrape structured data from websites using Decodo
- 🔧 Understanding HTML/CSS structure for scraping and navigating the DOM
- 🧠 Basics of LLMs and the fine-tuning process
- 🧹 How to clean and prepare datasets for training
- 📊 How to train/fine-tune a text classifier using Hugging Face
- 🛠️ Switching from a text generation model to a classification model
- 🧪 Evaluating LLMs for accuracy and performance
- 🔁 Using the fine-tuned model for real-time inference

---

## 🔧 Project Pipeline

1. Web scraping with **Decodo** (built-in proxy rotation & JavaScript rendering)
2. Crawl HTML pages, extract content with **BeautifulSoup**
3. Scraped **100 sites each** from: `politics`, `business`, `health`, `science`, `climate` — total **500 articles**
4. Cleaned, labeled, and tokenized the data using Hugging Face tools
5. Converted the data to HF datasets, added padding tokens for LLaMA
6. Fine-tuned the model for classification
7. Achieved:
   - ✅ 85% accuracy on training data
   - ✅ 60% accuracy on test data

---

## 🧠 LLM Training Workflow

1. Get data (scraped from web)
2. Clean and wrangle data (Label encoding, train-test split)
3. Convert to Hugging Face Dataset format
4. Tokenize (with LLaMA tokenizer, padding support)
5. Initialize model head for classification
6. Fine-tune on custom dataset
7. Evaluate performance and run inference

> ⚠️ Note: More epochs = better accuracy, but also more compute

---

## 🖥️ How to Run the App Locally

```bash
# Step 1: Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 2: Install dependencies
pip install gradio transformers torch

# Step 3: Launch the classifier
python3 main.py
```

---

## 🧪 Inference Demo (Gradio UI)

A simple Gradio front-end will open at [http://127.0.0.1:7860](http://127.0.0.1:7860). You can paste any news article or headline and get its predicted category.

---

## 🎥 Demo Video


https://github.com/mgulati3/Fine-Tune/assets/record.mov

(Embed this `.mov` file inside an `/assets` folder in your repo.)

---

## 📁 Project Structure

```
├── main.py                # Gradio app using HF pipeline
├── news_classifier_model/ # Saved tokenizer & model
├── assets/
│   └── record.mov         # Demo video
├── README.md
├── requirements.txt
```

---

## 🤝 Credits

- Built with ❤️ using Hugging Face Transformers, Gradio, Decodo, and BeautifulSoup
- Created by [Manan Gulati](mailto:mgulati3@asu.edu)
