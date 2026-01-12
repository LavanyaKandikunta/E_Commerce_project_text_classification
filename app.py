#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import gdown

# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)

# ============================================================
# Environment & memory optimizations
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

# ============================================================
# Model & metadata locations (lazy download)
# ============================================================
MODEL_DIR = "DL_models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE_ID = "1JsCHylh1wStYpPUQCHfMFRrcPYdBM11T"
METADATA_FILE_ID = "1a_txG7ddnCViQ8bAuI_UPkrBRTqpsOEq"

MODEL_FILE = os.path.join(MODEL_DIR, "gru_model.keras")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

# ============================================================
# Lazy-load GRU model + metadata + download only when needed
# ============================================================
gru_model = None
tokenizer = None
label_encoder = None

def load_gru_model():
    """Downloads and loads the GRU model + metadata only when needed."""
    global gru_model, tokenizer, label_encoder

    # ---- Lazy download ----
    if not os.path.exists(MODEL_FILE):
        print("⚡ Downloading GRU model (.keras) from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_FILE, quiet=False)

    if not os.path.exists(METADATA_FILE):
        print("⚡ Downloading metadata.json from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={METADATA_FILE_ID}", METADATA_FILE, quiet=False)

    # ---- Lazy load ----
    if gru_model is None:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        import tensorflow as tf

        tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')

        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)

        tokenizer = tokenizer_from_json(json.dumps(metadata["tokenizer"]))
        label_encoder = metadata["label_encoder"]
        gru_model = load_model(MODEL_FILE)

        print("✅ GRU model loaded successfully.")
    return gru_model

def predict_text(text):
    load_gru_model()  # ensures both download + load happen only once
    if tokenizer is None or label_encoder is None:
        return "⚠️ Model metadata missing"
    seq = tokenizer.texts_to_sequences([text])
    seq = np.array(seq)
    pred_probs = gru_model.predict(seq)
    idx = np.argmax(pred_probs, axis=1)[0]
    return label_encoder.get(str(idx), "Unknown")

# ============================================================
# Load user-product matrix (kept local for speed)
# ============================================================
user_product_matrix = None
if os.path.exists("user_product_matrix.csv"):
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)
elif os.path.exists("user_product_matrix.zip"):
    with zipfile.ZipFile("user_product_matrix.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)

def recommend_products(user_name, top_n=5):
    if user_product_matrix is None or user_product_matrix.empty:
        return ["⚠️ user_product_matrix not loaded"]
    if user_name not in user_product_matrix.index:
        return [f"⚠️ User '{user_name}' not found"]

    user_vec = user_product_matrix.loc[user_name].values.reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, user_product_matrix)[0]
    sim_df = pd.DataFrame({"user": user_product_matrix.index, "similarity": sim_scores}).sort_values(by="similarity", ascending=False)
    similar_users = sim_df[sim_df["user"] != user_name].head(10)["user"]
    similar_user_data = user_product_matrix.loc[similar_users]
    recommended_scores = similar_user_data.sum(axis=0)
    user_products = user_product_matrix.loc[user_name]
    not_bought = recommended_scores[user_products == 0]
    return not_bought.sort_values(ascending=False).head(top_n).index.tolist()

# ============================================================
# Review summary + feature importance
# ============================================================
def summarize_reviews(texts):
    results = []
    for text in texts:
        sentiment = predict_text(text)
        results.append({"review": text, "sentiment": sentiment})

    df = pd.DataFrame(results)

    tfidf = TfidfVectorizer(max_features=5, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    top_features = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray()[0]
        top_idx = row.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_idx if row[idx] > 0]
        top_features.append(", ".join(top_words))

    df["top_features"] = top_features
    return df

# ============================================================
# Flask Routes
# ============================================================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    recommendations = []
    summary_df = None

    if request.method == "POST":
        text_input = request.form.get("text_input")
        user_name = request.form.get("user_name")
        file = request.files.get("file_input")

        # File upload handling
        if file:
            lines = [line.decode("utf-8") for line in file.readlines()]
            summary_df = summarize_reviews(lines)
            prediction = summary_df.to_dict(orient="records")

        # Single text input handling
        if text_input:
            prediction = predict_text(text_input)

        # Product recommendations
        if user_name:
            recommendations = recommend_products(user_name)

    return render_template("index.html",
                           prediction=prediction,
                           recommendations=recommendations,
                           summary=summary_df)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify({"text": text, "prediction": predict_text(text)})

@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json(force=True)
    user_name = data.get("user")
    top_n = data.get("top_n", 5)
    recs = recommend_products(user_name, top_n)
    return jsonify({"user": user_name, "recommendations": recs})

@app.route("/status")
def status():
    return jsonify({
        "model_downloaded": os.path.exists(MODEL_FILE),
        "metadata_downloaded": os.path.exists(METADATA_FILE),
        "gru_loaded": gru_model is not None,
        "users": len(user_product_matrix) if user_product_matrix is not None else 0,
        "products": user_product_matrix.shape[1] if user_product_matrix is not None else 0
    })

# ============================================================
# Run app
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)


# In[ ]:




