import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Klasifikasi Berita Hoaks Menggunakan Naive Bayes")

# LOAD & GABUNG DATASET
antara = pd.read_csv("dataset/Cleaned_Antaranews_v1.csv")
detik = pd.read_csv("dataset/Cleaned_Detik_v2.csv")
kompas = pd.read_csv("dataset/Cleaned_Kompas_v2.csv")
turnback = pd.read_csv("dataset/Cleaned_TurnBackHoax_v3.csv")

dataset = pd.concat([antara, detik, kompas, turnback], ignore_index=True)
dataset.dropna(subset=["clean_text", "label"], inplace=True)

st.subheader("Preview Dataset Gabungan")
st.write("Total data:", dataset.shape[0])
st.dataframe(dataset.head())

# FITUR & LABEL
X = dataset["clean_text"]
y = dataset["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# EVALUASI
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

st.write(f"📊 Akurasi Model: **{akurasi:.2%}**")

# INPUT USER
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

        if hasil[0] == 1:
            st.error("🚨 BERITA HOAKS")
        else:
            st.success("✅ BERITA NON-HOAKS")

# DISTRIBUSI LABEL
# st.subheader("Distribusi Berita Hoaks vs Non-Hoaks")

# label_counts = dataset["label"].value_counts()

# st.bar_chart(label_counts)
