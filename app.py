#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gdown, zipfile

# ============================================================
# 0️⃣ Flask setup
# ============================================================
app = Flask(__name__)

# ============================================================
# 0️⃣ Environment setup for Render (CPU-only)
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================
# 1️⃣ GRU model setup (lazy load)
# ============================================================
MODEL_DIR = "DL_models"
os.makedirs(MODEL_DIR, exist_ok=True)

GRU_MODEL_FILE_ID = "1JsCHylh1wStYpPUQCHfMFRrcPYdBM11T"
GRU_METADATA_FILE_ID = "1a_txG7ddnCViQ8bAuI_UPkrBRTqpsOEq"

GRU_MODEL_FILE = os.path.join(MODEL_DIR, "gru_model.keras")
GRU_METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

if not os.path.exists(GRU_MODEL_FILE):
    gdown.download(f"https://drive.google.com/uc?id={GRU_MODEL_FILE_ID}", GRU_MODEL_FILE, quiet=False)

if not os.path.exists(GRU_METADATA_FILE):
    gdown.download(f"https://drive.google.com/uc?id={GRU_METADATA_FILE_ID}", GRU_METADATA_FILE, quiet=False)

gru_model = None
tokenizer = None
label_encoder = None

def load_gru_model():
    global gru_model, tokenizer, label_encoder
    if gru_model is None:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open(GRU_METADATA_FILE, "r") as f:
            metadata = json.load(f)
        tokenizer = tokenizer_from_json(json.dumps(metadata["tokenizer"]))
        label_encoder = metadata["label_encoder"]

        gru_model = load_model(GRU_MODEL_FILE)
        print("✅ GRU model loaded.")
    return gru_model

def predict_text_gru(text):
    load_gru_model()
    seq = tokenizer.texts_to_sequences([text])
    seq = np.array(seq)
    pred_probs = gru_model.predict(seq)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    return label_encoder.get(str(pred_index), "Unknown")

# ============================================================
# 2️⃣ BERT model setup (lazy load, .safetensors)
# ============================================================
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

BERT_DIR = "DL_models/bert_finetuned"
bert_model = None
bert_tokenizer = None
bert_label_map = None

def load_bert_model():
    global bert_model, bert_tokenizer, bert_label_map
    if bert_model is None:
        print("⚡ Loading BERT model...")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
        bert_model = BertForSequenceClassification.from_pretrained(
            BERT_DIR,
            from_tf=False,
            torch_dtype=torch.float32,
        )
        bert_model.eval()
        # Load label encoder
        bert_label_map = joblib.load(os.path.join(BERT_DIR, "label_encoder.pkl"))
        print("✅ BERT model loaded.")
    return bert_model

def predict_text_bert(text):
    load_bert_model()
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
    return bert_label_map.get(pred_id, "Unknown")

# ============================================================
# 3️⃣ Recommendation system
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
# 4️⃣ Flask routes
# ============================================================
@app.route("/", methods=["GET", "POST"])
def home():
    pred_gru, pred_bert, recs = None, None, []
    if request.method == "POST":
        text = request.form.get("text_input")
        user = request.form.get("user_name")
        if text:
            pred_gru = predict_text_gru(text)
            pred_bert = predict_text_bert(text)
        if user:
            recs = recommend_products(user)
    return render_template("index.html", prediction_gru=pred_gru, prediction_bert=pred_bert, recommendations=recs)

@app.route("/status")
def status():
    return jsonify({
        "gru_loaded": gru_model is not None,
        "bert_loaded": bert_model is not None,
        "users": len(user_product_matrix) if user_product_matrix is not None else 0
    })

# ============================================================
# 5️⃣ Run Flask
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

