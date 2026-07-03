"""
explainer.py – Generate human-readable explanations for sentiment predictions.

Uses:
- TF-IDF feature weights to find top contributing words
- NLTK VADER for per-word sentiment scoring
- NLTK POS tagging for key phrase extraction
"""

import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from data_loader import LABEL_NAMES, ensure_nltk_data

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

_vader = None


def _get_vader():
    global _vader
    if _vader is None:
        ensure_nltk_data()
        _vader = SentimentIntensityAnalyzer()
    return _vader


# ---------------------------------------------------------------------------
# Core explanation logic
# ---------------------------------------------------------------------------


def get_top_tfidf_features(text: str, vectorizer, model, top_n: int = 10):
    """
    Return the top_n TF-IDF features from *text* that most influenced
    the model's prediction, along with their weights.
    """
    vec = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = vec.toarray().flatten()

    # Get non-zero features
    nonzero_idx = scores.nonzero()[0]
    if len(nonzero_idx) == 0:
        return []

    # Sort by TF-IDF weight descending
    sorted_idx = nonzero_idx[np.argsort(scores[nonzero_idx])[::-1]][:top_n]

    results = []
    for idx in sorted_idx:
        results.append({
            "word": feature_names[idx],
            "tfidf_weight": float(scores[idx]),
        })
    return results


def get_vader_word_scores(text: str):
    """
    Score each word in the text using VADER's lexicon.
    Returns list of dicts with word, compound score, and category.
    """
    vader = _get_vader()
    tokens = word_tokenize(text.lower())
    word_scores = []

    for token in tokens:
        if len(token) <= 2:
            continue
        score = vader.polarity_scores(token)
        compound = score["compound"]
        if compound >= 0.05:
            category = "positive"
        elif compound <= -0.05:
            category = "negative"
        else:
            category = "neutral"
        word_scores.append({
            "word": token,
            "score": compound,
            "category": category,
        })
    return word_scores


def get_key_phrases(text: str, top_n: int = 5):
    """
    Extract key noun phrases and verb phrases using POS tagging.
    """
    ensure_nltk_data()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Extract meaningful phrases: adjective+noun, adverb+verb, etc.
    phrases = []
    i = 0
    while i < len(tagged):
        word, tag = tagged[i]
        if len(word) <= 2:
            i += 1
            continue

        # Adjective + Noun pattern
        if tag.startswith("JJ") and i + 1 < len(tagged) and tagged[i + 1][1].startswith("NN"):
            phrases.append(f"{word} {tagged[i + 1][0]}")
            i += 2
            continue
        # Adverb + Adjective pattern
        if tag.startswith("RB") and i + 1 < len(tagged) and tagged[i + 1][1].startswith("JJ"):
            phrases.append(f"{word} {tagged[i + 1][0]}")
            i += 2
            continue
        # Standalone adjective or important verb
        if tag.startswith("JJ") or tag in ("VBG", "VBN", "VBD"):
            phrases.append(word)
        i += 1

    return phrases[:top_n]


def get_prediction_confidence(text: str, vectorizer, model):
    """
    Get prediction probabilities if model supports predict_proba.
    Falls back to decision_function or returns None.
    """
    vec = vectorizer.transform([text]).toarray()

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        return {
            "predicted_label": int(pred_idx),
            "predicted_sentiment": LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)
            },
        }
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(vec)[0]
        if isinstance(decision, np.ndarray):
            pred_idx = np.argmax(decision)
            # Convert to pseudo-probabilities via softmax
            exp_d = np.exp(decision - np.max(decision))
            probs = exp_d / exp_d.sum()
            return {
                "predicted_label": int(pred_idx),
                "predicted_sentiment": LABEL_NAMES[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {
                    LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)
                },
            }
        else:
            # Binary case
            pred_idx = int(model.predict(vec)[0])
            return {
                "predicted_label": pred_idx,
                "predicted_sentiment": LABEL_NAMES[pred_idx],
                "confidence": None,
                "probabilities": None,
            }
    else:
        pred_idx = int(model.predict(vec)[0])
        return {
            "predicted_label": pred_idx,
            "predicted_sentiment": LABEL_NAMES[pred_idx],
            "confidence": None,
            "probabilities": None,
        }


def explain_prediction(text: str, vectorizer, model):
    """
    Full explanation for a single prediction.

    Returns a dict with:
    - sentiment, confidence, probabilities
    - key_phrases: meaningful extracted phrases
    - top_features: TF-IDF contributing words
    - word_sentiments: VADER per-word scores
    - positive_words, negative_words, neutral_words
    - key_reasons: human-readable bullet points
    """
    from data_loader import clean_text

    cleaned = clean_text(text)
    prediction = get_prediction_confidence(cleaned, vectorizer, model)

    # Feature analysis
    top_features = get_top_tfidf_features(cleaned, vectorizer, model, top_n=8)

    # VADER word-level analysis (on original text for better readability)
    word_sentiments = get_vader_word_scores(text)
    positive_words = [ws for ws in word_sentiments if ws["category"] == "positive"]
    negative_words = [ws for ws in word_sentiments if ws["category"] == "negative"]
    neutral_words = [ws for ws in word_sentiments if ws["category"] == "neutral"]

    # Key phrases from original text
    key_phrases = get_key_phrases(text, top_n=5)

    # Build human-readable key reasons
    key_reasons = []
    vader = _get_vader()
    sentiment_label = prediction["predicted_sentiment"]

    # Reason from key phrases
    for phrase in key_phrases[:3]:
        score = vader.polarity_scores(phrase)["compound"]
        if score >= 0.05:
            key_reasons.append(f'"{phrase}" → positive indicator (score: {score:+.2f})')
        elif score <= -0.05:
            key_reasons.append(f'"{phrase}" → negative indicator (score: {score:+.2f})')
        else:
            key_reasons.append(f'"{phrase}" → neutral context')

    # Reason from strong VADER words
    for ws in sorted(positive_words + negative_words, key=lambda x: abs(x["score"]), reverse=True)[:3]:
        reason = f'"{ws["word"]}" → {ws["category"]} sentiment (score: {ws["score"]:+.2f})'
        if reason not in key_reasons:
            key_reasons.append(reason)

    # Reason from top TF-IDF features
    for feat in top_features[:2]:
        reason = f'"{feat["word"]}" → significant in model vocabulary (weight: {feat["tfidf_weight"]:.3f})'
        if not any(feat["word"] in r for r in key_reasons):
            key_reasons.append(reason)

    # Trim to top 5
    key_reasons = key_reasons[:5]

    # Overall VADER score as additional context
    overall_vader = vader.polarity_scores(text)

    return {
        **prediction,
        "cleaned_text": cleaned,
        "key_phrases": key_phrases,
        "top_features": top_features,
        "word_sentiments": word_sentiments,
        "positive_words": [w["word"] for w in positive_words],
        "negative_words": [w["word"] for w in negative_words],
        "neutral_words": [w["word"] for w in neutral_words],
        "key_reasons": key_reasons,
        "vader_overall": overall_vader,
    }
