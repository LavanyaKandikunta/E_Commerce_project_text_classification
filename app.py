#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import zipfile
import gdown


# ============================================================
# Add these optimizations to Reduce memory footprint
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU completely

# ============================================================
# Recommended setup for Render Free Tier
# ============================================================

tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')



app = Flask(__name__)

# ============================================================
# 0️⃣ Download GRU model (.keras) and metadata.json from Google Drive
# ============================================================
MODEL_FILE_ID = "1JsCHylh1wStYpPUQCHfMFRrcPYdBM11T"   # GRU model file ID
METADATA_FILE_ID = "1a_txG7ddnCViQ8bAuI_UPkrBRTqpsOEq"             # metadata.json file ID

MODEL_DIR = "DL_models"
MODEL_FILE = os.path.join(MODEL_DIR, "gru_model.keras")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# Download model file if missing
if not os.path.exists(MODEL_FILE):
    print("⚡ Downloading GRU model (.keras) from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_FILE, quiet=False)
    print("✅ GRU model downloaded successfully.")
else:
    print("✅ GRU model already exists locally.")

# Download metadata if missing
if not os.path.exists(METADATA_FILE):
    print("⚡ Downloading metadata.json from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={METADATA_FILE_ID}", METADATA_FILE, quiet=False)
    print("✅ Metadata downloaded successfully.")
else:
    print("✅ Metadata already exists locally.")

# ============================================================
# 1️⃣ Load the GRU model
# ============================================================
gru_model = load_model(MODEL_FILE)

# ============================================================
# 2️⃣ Load metadata (tokenizer + label encoder)
# ============================================================
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    tokenizer = tokenizer_from_json(metadata["tokenizer"])
    label_encoder = metadata["label_encoder"]
else:
    tokenizer = None
    label_encoder = None
    print("⚠️ metadata.json not found. Text classification may not work.")

# ============================================================
# 3️⃣ Prediction function for GRU model
# ============================================================
def predict_text(text):
    if tokenizer is None or label_encoder is None:
        return "⚠️ Model metadata missing"
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    pred_probs = gru_model.predict(sequence)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    return label_encoder.get(str(pred_index), "Unknown")

# ============================================================
# 4️⃣ Load user-product matrix for recommendations
# ============================================================
user_product_matrix = None
if os.path.exists("user_product_matrix.csv"):
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)
elif os.path.exists("user_product_matrix.zip"):
    with zipfile.ZipFile("user_product_matrix.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)

# ============================================================
# 5️⃣ Recommendation function
# ============================================================
def recommend_products(user_name, top_n=5):
    if user_product_matrix is None or user_product_matrix.empty:
        return ["⚠️ user_product_matrix not loaded"]
    if user_name not in user_product_matrix.index:
        return [f"⚠️ User '{user_name}' not found"]

    user_vector = user_product_matrix.loc[user_name].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, user_product_matrix)[0]

    similarity_df = pd.DataFrame({
        "user": user_product_matrix.index,
        "similarity": similarity_scores
    }).sort_values(by="similarity", ascending=False)

    similar_users = similarity_df[similarity_df["user"] != user_name].head(10)["user"]

    similar_user_data = user_product_matrix.loc[similar_users]
    recommended_scores = similar_user_data.sum(axis=0)

    user_products = user_product_matrix.loc[user_name]
    not_bought = recommended_scores[user_products == 0]

    recommendations = not_bought.sort_values(ascending=False).head(top_n).index.tolist()
    return recommendations

# ============================================================
# 6️⃣ Flask Routes
# ============================================================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        text = request.form.get("text_input")
        if text:
            prediction = predict_text(text)
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    label = predict_text(text)
    return jsonify({"text": text, "prediction": label})

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
        "text_model_loaded": gru_model is not None,
        "recommendation_matrix_loaded": user_product_matrix is not None and not user_product_matrix.empty,
        "users": len(user_product_matrix) if user_product_matrix is not None else 0,
        "products": user_product_matrix.shape[1] if user_product_matrix is not None else 0
    })

# Disable GPU for Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================
# 7️⃣ Run app
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port, debug=False)


# In[ ]:




