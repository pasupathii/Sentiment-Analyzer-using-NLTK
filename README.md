# 🔮 Sentiment Analyzer Pro

AI-powered sentiment analysis with explainable predictions. Analyzes customer reviews, tweets, and any text to determine **positive**, **negative**, or **neutral** sentiment with detailed explanations.

## ✨ Features

- **Multi-Model Training** — Compares 5 ML models (Naive Bayes, Logistic Regression, SVM, Random Forest, Voting Ensemble) and auto-selects the best
- **Multi-Dataset Training** — Trained on 26K+ samples from airline tweets, movie reviews, and Twitter data
- **Explainable AI** — Every prediction comes with key reasons, word-level sentiment analysis, and confidence scores
- **Batch Analysis** — Upload CSV files for bulk sentiment analysis
- **Premium UI** — Dark-themed Streamlit interface with interactive charts

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_models.py
```

This will:
- Download and combine 3 datasets (~26K+ samples)
- Train 5 different ML models with hyperparameter tuning
- Auto-select the best model
- Save everything to `models/` directory

### 3. Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

> **Note:** Make sure the `models/` directory is committed to Git (it's not in `.gitignore`).

## 📁 Project Structure

```
├── .streamlit/config.toml     # Streamlit dark theme
├── models/                    # Trained model artifacts
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── training_report.json
├── data_loader.py             # Multi-dataset loading & preprocessing
├── train_models.py            # Multi-model training pipeline
├── explainer.py               # Prediction explanation engine
├── app.py                     # Streamlit UI (main entry)
├── dataset.csv                # Twitter US Airline dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🤖 Models Trained

| Model | Description |
|---|---|
| Multinomial Naive Bayes | Classic NLP baseline |
| Logistic Regression | Strong linear text classifier |
| Linear SVM | Best for high-dimensional sparse data |
| Random Forest | Ensemble of decision trees |
| Voting Ensemble | Soft-voting combination of top 3 |

## 📊 Datasets Used

| Dataset | Source | Samples |
|---|---|---|
| Twitter US Airline | `dataset.csv` | ~14.6K |
| NLTK Movie Reviews | `nltk.corpus` | ~2K |
| NLTK Twitter Samples | `nltk.corpus` | ~10K |

## 📝 License

MIT License
