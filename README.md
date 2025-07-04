
# 📰 LLM News Classifier with Web Scraping & Fine-Tuning

---

## 🎥 Demo Video

https://github.com/mgulati3/Fine-Tune/assets/record.gif

---

## 👋 Introduction

This is a complete project I built from scratch that showcases how to collect real-world data by **scraping news articles**, clean and format it, and use it to **fine-tune a Large Language Model (LLM)** for text classification.

I scraped articles directly from **[NPR](https://www.npr.org/)** using Decodo’s smart web scraping tools (with built-in proxy handling and JS rendering). Then, I used **BeautifulSoup** to extract the article links and text from the HTML.

After cleaning and preparing the dataset, I fine-tuned a model on five news categories — `politics`, `business`, `health`, `science`, and `climate`.

🧠 The final model is hosted on Hugging Face:
🔗 [mgulati3/news-classifier-model](https://huggingface.co/mgulati3/news-classifier-model)

You can use the model locally with a Gradio UI to classify custom news inputs. It’s a great starter project for anyone looking to master data scraping, LLM fine-tuning, and inference deployment.

---

## 👋 Introduction

Hi! I’m Manan Gulati, and this is a hands-on project I built to explore the **end-to-end pipeline** of training a Large Language Model (LLM) using real-world data collected via **web scraping**.

I developed a custom text classification model for news articles by:
- Crawling and scraping real news websites using **Decodo**
- Parsing and processing content with **BeautifulSoup**
- Fine-tuning a transformer-based model with **Hugging Face Transformers**

You can check out the final model here on the Hugging Face Hub:  
👉 [mgulati3/news-classifier-model](https://huggingface.co/mgulati3/news-classifier-model)

---

## 🔧 Project Pipeline

1. Web scraping with **Decodo** (built-in proxy rotation & JavaScript rendering)
2. Crawl HTML pages, extract content with **BeautifulSoup**
3. Scraped **1000 sites each** from: `politics`, `business`, `health`, `science`, `climate` — total **5000 articles**
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

## 🤝 Credits

- Built with ❤️ using Hugging Face Transformers, Gradio, Decodo, and BeautifulSoup
- Created by [Manan Gulati](mailto:mgulati3@asu.edu)
