import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cityblock
from gensim.models import Word2Vec
import pandas as pd

# Fungsi untuk mengambil abstrak dari URL
def fetch_abstracts_from_urls(urls):
    """
    Mengambil abstrak dari URL.
    """
    all_abstracts = []

    for url in urls:
        print(f"Fetching abstracts from {url}...")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Cari div dengan id 'articleAbstract'
            article_abstract = soup.find('div', id='articleAbstract')
            if article_abstract:
                abstract_text = article_abstract.text.strip()
                all_abstracts.append(abstract_text)
            else:
                print(f"Abstract not found in {url}.")

        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")

    return all_abstracts

# Fungsi untuk memroses abstrak
def preprocess_abstracts(abstracts):
    """
    Memproses abstrak menjadi lowercase.
    """
    return [abstract.lower() for abstract in abstracts]

# Representasi fitur 1: TF-IDF Vectorizer
def calculate_tfidf_cosine_similarity(abstracts):
    """
    Menghitung kemiripan cosine menggunakan representasi TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts)  # Representasi fitur TF-IDF
    similarity_matrix = cosine_similarity(tfidf_matrix)  # Menghitung cosine similarity
    return similarity_matrix

# Representasi fitur 2: Word2Vec + Manhattan Distance
def calculate_word2vec_manhattan_similarity(abstracts):
    """
    Menghitung kemiripan menggunakan Word2Vec dan Manhattan Distance.
    """
    # Membuat model Word2Vec dengan gensim
    tokenized_abstracts = [abstract.split() for abstract in abstracts]
    word2vec_model = Word2Vec(sentences=tokenized_abstracts, vector_size=100, min_count=1, window=5)

    # Merepresentasikan setiap abstrak sebagai rata-rata vektor kata
    def vectorize_sentence(sentence):
        vectors = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    sentence_vectors = np.array([vectorize_sentence(sentence) for sentence in tokenized_abstracts])

    # Menghitung matriks kemiripan menggunakan Manhattan Distance
    n_abstracts = len(sentence_vectors)
    similarity_matrix = np.zeros((n_abstracts, n_abstracts))

    for i in range(n_abstracts):
        for j in range(n_abstracts):
            similarity_matrix[i, j] = -cityblock(sentence_vectors[i], sentence_vectors[j])  # Negasi karena distance

    return similarity_matrix

# Fungsi untuk menyimpan matriks similarity ke file CSV
def save_similarity_matrix(similarity_matrix, filename="similarity_matrix.csv"):
    """
    Simpan matriks similarity dalam format CSV.
    """
    rounded_matrix = np.round(similarity_matrix, 4)
    similarity_df = pd.DataFrame(
        rounded_matrix,
        columns=[f"Abstract {i+1}" for i in range(len(similarity_matrix))],
        index=[f"Abstract {i+1}" for i in range(len(similarity_matrix))]
    )
    similarity_df.to_csv(filename, index=True)
    print(f"Similarity matrix saved to '{filename}'.")

# Fungsi utama
def main():
    urls = [
        "https://jurnal.ugm.ac.id/ijccs/article/view/73334",
        "https://jurnal.ugm.ac.id/ijccs/article/view/78537",
        "https://jurnal.ugm.ac.id/ijccs/article/view/79623",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80776",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80077",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80214",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80049",
        "https://jurnal.ugm.ac.id/ijccs/article/view/90437",
        "https://jurnal.ugm.ac.id/ijccs/article/view/81178",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80956"
    ]

    abstracts = fetch_abstracts_from_urls(urls)
    print(f"Fetched {len(abstracts)} abstracts.")

    if abstracts:
        preprocessed_abstracts = preprocess_abstracts(abstracts)

        # Perhitungan menggunakan TF-IDF dan Cosine Similarity
        tfidf_cosine_similarity_matrix = calculate_tfidf_cosine_similarity(preprocessed_abstracts)
        save_similarity_matrix(tfidf_cosine_similarity_matrix, filename="tfidf_cosine_similarity.csv")

        # Perhitungan menggunakan Word2Vec dan Manhattan Distance
        word2vec_manhattan_similarity_matrix = calculate_word2vec_manhattan_similarity(preprocessed_abstracts)
        save_similarity_matrix(word2vec_manhattan_similarity_matrix, filename="word2vec_manhattan_similarity.csv")

    else:
        print("No abstracts to process.")

if __name__ == "__main__":
    main()