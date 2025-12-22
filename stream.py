import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================
# JUDUL
# =====================
st.title("Klasifikasi Berita Hoax Menggunakan Naive Bayes")

# =====================
# LOAD DATASET
# =====================
dataset = pd.read_csv(r"dataset/Cleaned_Antaranews_v1.csv")
dataset.dropna(inplace=True)

st.subheader("Preview Dataset")
st.dataframe(dataset.head())

# =====================
# FITUR & LABEL (FIXED)
# =====================
X = dataset["clean_text"]
y = dataset["label"]

# =====================
# TF-IDF
# =====================
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# =====================
# SPLIT DATA
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# =====================
# TRAIN MODEL
# =====================
model = MultinomialNB()
model.fit(X_train, y_train)

# =====================
# EVALUASI
# =====================
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

st.write(f"📊 Akurasi Model: **{akurasi:.2%}**")

# =====================
# INPUT USER
# =====================
st.subheader("Prediksi Berita Baru")

input_text = st.text_area(
    "Masukkan teks berita:",
    height=200
)

if st.button("Prediksi"):
    if input_text.strip() == "":
        st.warning("Teks berita tidak boleh kosong")
    else:
        input_tfidf = vectorizer.transform([input_text])
        hasil = model.predict(input_tfidf)

        if hasil[0] == "hoax":
            st.error("🚨 BERITA HOAX")
        else:
            st.success("✅ BERITA NON-HOAX")
