#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

app = Flask(__name__)

# -----------------------------
# Load user-product matrix safely
# -----------------------------
user_product_matrix = None

try:
    if os.path.exists("user_product_matrix.csv"):
        user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)
        print("✅ Loaded user_product_matrix.csv successfully")
    elif os.path.exists("user_product_matrix.zip"):
        with zipfile.ZipFile("user_product_matrix.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0).sample(n=500, axis=0)
        print("✅ Loaded 500 sample users for Render demo")
    else:
        print("⚠️ user_product_matrix file not found — API will still run but recommend() will be inactive.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    user_product_matrix = pd.DataFrame()

# -----------------------------
# Recommendation logic
# -----------------------------
def recommend_products(user_name, top_n=5):
    """
    Recommend products for a given user based on user-user collaborative filtering.
    """
    if user_product_matrix is None or user_product_matrix.empty:
        return ["⚠️ user_product_matrix.csv not loaded on server"]

    if user_name not in user_product_matrix.index:
        return [f"⚠️ User '{user_name}' not found in dataset"]

    # Compute similarity
    user_vector = user_product_matrix.loc[user_name].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, user_product_matrix)[0]

    similarity_df = pd.DataFrame({
        "user": user_product_matrix.index,
        "similarity": similarity_scores
    }).sort_values(by="similarity", ascending=False)

    similar_users = similarity_df[similarity_df["user"] != user_name].head(10)["user"]

    # Aggregate and recommend
    similar_user_data = user_product_matrix.loc[similar_users]
    recommended_scores = similar_user_data.sum(axis=0)

    user_products = user_product_matrix.loc[user_name]
    not_bought = recommended_scores[user_products == 0]
    recommendations = not_bought.sort_values(ascending=False).head(top_n).index.tolist()

    return recommendations

# -----------------------------
# API Routes
# -----------------------------
@app.route("/")
def home():
    return "✅ E-Commerce Recommendation API is running successfully!"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    user_name = data.get("user")
    top_n = data.get("top_n", 5)

    recs = recommend_products(user_name, top_n)
    return jsonify({"user": user_name, "recommendations": recs})

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


# In[ ]:




