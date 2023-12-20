import re
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import os

# Baca file CSV
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'data.csv')
df = pd.read_csv(file_path)
df = df.drop_duplicates()

nltk.download('punkt')

# Fungsi untuk pra-pemrosesan teks
def preprocess_text(text):
    # Menghapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Menghapus tagar dan simbol @
    text = re.sub(r'\@\w+|\#','', text)
    # Menghapus karakter non-alfanumerik
    text = re.sub(r'\W', ' ', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    # Mengubah teks menjadi huruf kecil
    return text.lower()

# Fungsi untuk tokenisasi teks dan penghapusan stopwords
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

# Fungsi untuk mengganti kata singkatan
def fix_abbreviations(text, abbreviations):
    # Buat pola regex dari daftar kata singkatan
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b')
    # Gantilah kata singkatan dengan kata lengkapnya
    fixed_text = pattern.sub(lambda match: abbreviations[match.group(0)], text)
    return fixed_text

# Kamus singkatan dan kata lengkap dalam bahasa Indonesia
abbreviations = {
    'bgt': 'banget',
    'bkn': 'bukan',
    'cm': 'cuma',
    'dgn': 'dengan',
    'jg': 'juga',
    'krn': 'karena',
    'lg': 'lagi',
    'sdh': 'sudah',
    'ttg': 'tentang',
    'yng': 'yang',
    'blm': 'belum',
    'gimana': 'bagaimana',
    'gtw': 'gatau',
    'hrs': 'harus',
    'jd': 'jadi',
    'kl': 'kalau',
    'knp': 'kenapa',
    'nnti': 'nanti',
    'pdhl': 'padahal',
    'tp': 'tapi',
    'ttg': 'tentang',
    'udh': 'udah',
    'utk': 'untuk',
    'yg' : 'yang',
    'gk' : 'gak',
}

# Menerapkan fungsi pra-pemrosesan ke data
df['full_text'] = df['full_text'].apply(preprocess_text)

# Menerapkan fungsi tokenisasi dan penghapusan stopwords ke data
df['full_text'] = df['full_text'].apply(tokenize_and_remove_stopwords)

# Menerapkan fungsi untuk mengganti kata singkatan
df['full_text'] = df['full_text'].apply(lambda text: fix_abbreviations(text, abbreviations))

# Pisahkan data menjadi data pelatihan dan data pengujian
X = df['full_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vektorisasi teks
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Prediksi sentimen
y_pred = rf_model.predict(X_test_tfidf)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Fungsi untuk memprediksi sentimen
def predict_sentiment(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = rf_model.predict(text_tfidf)
    return prediction[0]

# Judul aplikasi
st.title('Analisis Sentimen Kualitas Layanan Transportasi Online di Indonesia dengan Random Forest')

# Tambahkan tautan ke sidebar untuk navigasi antar halaman
st.sidebar.header('Navigasi Halaman') 
page = st.sidebar.selectbox("Halaman", ["Tabel", "Prediksi Sentimen", "Word Cloud", "Metrik Evaluasi"])

if page == "Tabel":
    # Prediksi sentimen untuk semua data
    df['predicted_sentiment'] = rf_model.predict(tfidf_vectorizer.transform(df['full_text']))
    # Halaman 1: Tabel dengan komentar Twitter dan akurasi
    st.header('Tabel Komentar Twitter dan Label Sentimen')
    st.write(df[['full_text', 'predicted_sentiment']].rename(columns={'full_text': 'Tweet', 'predicted_sentiment': 'Label'}))

    # Halaman 1: Tabel dengan komentar Twitter dan akurasi
    st.write('### Filter Komentar berdasarkan Label Sentimen')
    selected_sentiment = st.selectbox("Pilih Label Sentimen:", df['predicted_sentiment'].unique())

    # Filter data berdasarkan label sentimen yang dipilih
    filtered_df = df[df['predicted_sentiment'] == selected_sentiment]

    # Tampilkan tabel dengan komentar Twitter yang sesuai dengan label sentimen yang dipilih
    st.write('### Tabel Komentar Twitter dengan Label Sentimen yang Dipilih')
    st.write(filtered_df[['full_text', 'predicted_sentiment']].rename(columns={'full_text': 'Tweet', 'predicted_sentiment': 'Label'}))

    # Tampilkan jumlah komentar yang sesuai dengan label sentimen yang dipilih
    num_comments = len(filtered_df)
    st.write(f'Jumlah Komentar dengan Label Sentimen "{selected_sentiment}": {num_comments}')

elif page == "Prediksi Sentimen":
    # Halaman 2: Prediksi sentimen
    st.write('### Prediksi Sentimen Komentar Twitter')
    input_text = st.text_input('Masukkan komentar Twitter:')
    if st.button('Prediksi Sentimen'):
        sentiment = predict_sentiment(input_text)
        st.write(f'Sentimen: {sentiment}')

elif page == "Word Cloud":

    # Melihat distribusi sentimen
    st.subheader('Distribusi Sentimen')
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax)
    st.pyplot(fig)

    # Halaman 3: Word Cloud
    st.write('### Word Cloud')

        # Menerapkan fungsi pra-pemrosesan ke data
    df['full_text'] = df['full_text'].apply(preprocess_text)

    # Menerapkan fungsi tokenisasi dan penghapusan stopwords ke data
    df['full_text'] = df['full_text'].apply(tokenize_and_remove_stopwords)

    all_words = ' '.join([text for text in df['full_text']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)


elif page == "Metrik Evaluasi" :
    # Menampilkan metrik evaluasi
    st.subheader('Metrik Evaluasi Model')
    st.metric('Akurasi', accuracy)
    st.metric('Presisi', precision)
    st.metric('Recall', recall)
    st.metric('F1-score', f1)