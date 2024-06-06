import cv2
import math
import nltk
import numpy as np
import os
import pandas as pd
import pytesseract
import re
import yaml
import pickle
import keras
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from spire.doc import *
from spire.doc.common import *
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
nltk.download('stopwords')
wnl = WordNetLemmatizer()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
words = set(nltk.corpus.words.words())
stop_words = set(nltk.corpus.stopwords.words('english'))

def main_code(file_path):
    """Main function to process a single file and send the results to show_results."""
    try:
        print('main_code')
        root, config = load_config()
        poppler_path = 'D:/users/poppler-24.02.0/Library/bin'
        output_path = 'D:/Github/PiSAScan/data/output.xlsx'
        
        embeddings, processed_data = process_file(file_path, poppler_path=poppler_path)
        df_embeddings, df_cvs = save_to_df(embeddings, [processed_data], output_path)
        
        models_path = 'D:/Github/PiSAScan/models'
        clf_clusters, keras_model, kmeans, lda_model, pca, clusters_coeff, words_coeff = load_models_and_data(models_path)
        
        # if not all([clf_clusters, keras_model, kmeans, lda_model, pca, clusters_coeff, words_coeff]):
        #     print("Error: One or more models or data files could not be loaded.")
        #     return
        
        print('Before enter to test_model')
        results, X = test_model(df_embeddings, df_cvs, [0], lda_model, keras_model, clf_clusters, pca, None, kmeans.n_clusters, clusters_coeff, words_coeff)
        print('After enter to test model')
        return results, X

    except Exception as e:
        print(f"Error in main execution block: {e}")

# Functions

def concat_chunks(translated_list):
    """Concatenate list of translated chunks into a single string."""
    return "".join(translated_list)

def detect_language(text):
    """
    Detect the language of the given text.

    Args:
    text (str): Text to detect the language of.

    Returns:
    str: Detected language code or "Unknown" if detection fails.
    """
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "Unknown"

def get_project_root():
    """Return the root directory of the project as a Path object."""
    return Path(__file__).parent 

def load_config():
    """Load configuration from a YAML file located in the project root directory."""
    project_root = get_project_root()
    config_path = project_root / "config.yml"
    try:
        with open(config_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return project_root, params
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return project_root, {}

def preprocess_by_fixed_words(text, label, index, num_words=7):
    """
    Preprocess text by translating, normalizing, lemmatizing, and extracting embeddings. 

    Args:
    text (str): Text to preprocess.
    label (str): Label for the data. 
    index (int): Data index.
    num_words (int): Number of words per chunk for embeddings.

    Returns:
    Tuple: Embeddings and the processed text.
    """
    try:
        text = translate_text(text)
        regex = re.compile('[^a-zA-Z]')
        text = unidecode(text)
        text = regex.sub(' ', text).lower()
        filtered_text = " ".join(wnl.lemmatize(word) for word in nltk.wordpunct_tokenize(text) if word in words and word not in stop_words and len(word) > 1)
        text_chunks = filtered_text.split()
        embeddings = []
        for i in range(0, len(text_chunks), num_words):
            sentence = ' '.join(text_chunks[i:i + num_words])
            embedding = model.encode(sentence)
            embeddings.append({'label': label, 'index': index, 'sentence': sentence, 'embedding': embedding, 'len': math.ceil(len(text_chunks) / num_words)})
        return embeddings, filtered_text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return [], " "

def process_doc(file_path):
    """Process DOC/DOCX files to extract text."""
    try:
        document = Document(file_path)
        txt = document.GetText()
        document.Close()
        return txt
    except Exception as e:
        print(f"Error processing DOC/DOCX file {file_path}: {e}")
        return ""

def process_pdf(file_path, poppler_path):
    """Process PDF files to extract text."""
    try:
        pages = convert_from_path(file_path, dpi=200, poppler_path=poppler_path)
        text = ""
        for page in pages:
            img = np.array(page.convert('RGB')).astype(np.uint8)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            page_text = pytesseract.image_to_string(thr, lang='spa+eng')
            text += page_text
        return text
    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")
        return ""

def save_to_df(embeddings, cvs, filename):
    """Save embeddings and CVs to 2 df's.""" 
    try:
        df_embeddings = pd.DataFrame(embeddings)
        df_cvs = pd.DataFrame(cvs)
        return df_embeddings, df_cvs
    except Exception as e:
        print(f"Error saving to file {filename}: {e}")

def split_text(text, chunk_size=4999):
    """
    Split text into chunks of specified size, ensuring that each chunk is within a limit.

    Args: 
    text (str): Text to split.
    chunk_size (int): Maximum size of each chunk.

    Returns:
    List[str]: List of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def translate_text(text):
    """
    Translate text to English if it not already in English.

    Args:
    text (str): Text to translate.

    Returns:
    str: Translated English text. 
    """
    try:
        language = detect_language(text)
        if language != 'en':
            chunks = split_text(text)
            translated = [GoogleTranslator(source='auto', target='en').translate(chunk) for chunk in chunks]
            return concat_chunks(translated)
        else:
            return text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

def process_file(file_path, label='cv', num_words=7, poppler_path='/usr/local/opt/poppler/bin'):
    """Process a single PDF, DOC, or DOCX file to extract embeddings and text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = process_pdf(file_path, poppler_path)
    elif ext in ['.doc', '.docx']:
        text = process_doc(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    embeddings, processed_text = preprocess_by_fixed_words(text, label, 0, num_words)
    return embeddings, {'label': label, 'index': 0, 'text': processed_text}

def get_clusters_features(df, clusters_coeff):
    """
    This function gets the features for the clusters
    """
    for column in df.columns:
        if str(column) in clusters_coeff.columns:
            df[column] *= clusters_coeff[str(column)].iloc[0]

    return df

def get_individual_words_features(df, words_coeff):
    """
    This function extract features
    """
    # We get the features for c_idf
    df_copy = df.copy()
    df_copy['combined_sentences'] = df.groupby('index')['sentence'].transform(lambda x: " ".join(x))
    corpus = df_copy.drop_duplicates(subset=['index', 'combined_sentences'])
    print('Corpus Shape')
    print(corpus.shape)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus['combined_sentences'])

    columns = vectorizer.get_feature_names_out()
    cols = words_coeff.columns
    print('Cols')
    print(cols)
    X = X.toarray()
    result = np.zeros((X.shape[0], words_coeff.shape[1]))
    for i, col in enumerate(cols):
        if col in columns:
            index = -1
            for it in range(len(columns)):
                if columns[it] == col:
                    index = it
                    break
            if index != -1:
                result[:, i] = (X[:, index] * words_coeff[col].iloc[0])

    return result

def get_test_matrix(df, df_norm, num_clusters, clusters_coeff, words_coeff):
    """
    """
    keys = df['index'].unique()
    X = {k: np.zeros(num_clusters+1) for k in keys}
    
    for _, row in df.iterrows():
        X[row['index']][row['cluster']] += 1

    for key, value in X.items():
        X[key][:-1] = X[key][:-1] / df_norm.iloc[key] if df_norm.iloc[key] != 0 else X[key][:-1]
        X[key][-1] = df_norm.iloc[key] / 100
    
    df_result = pd.DataFrame(X).T

    clusters_features = np.array(get_clusters_features(df_result, clusters_coeff))
    words_features = np.array(get_individual_words_features(df, words_coeff))

    print('Cluster features')
    print(clusters_features.shape)
    print('Words features')
    print(words_features.shape)
    # Merge clusters_features and words_features together
    df_result = np.concatenate([clusters_features, words_features], axis=1)
    print('DfResult')
    print(df_result.shape)

    return df_result

def load_models_and_data(models_path):
    """Function to load the required models and data files from the specified path."""
    try:
        print('load_models')
        with open(f'{models_path}/clf_clusters.pkl', 'rb') as file:
            clf_clusters = pickle.load(file)
        # with open(f'{models_path}/clf_model.pkl', 'rb') as file:
        #     clf_model = pickle.load(file)
        keras_model = keras.saving.load_model(f'{models_path}/classification_model.keras')
        with open(f'{models_path}/kmeans.pkl', 'rb') as file:
            kmeans = pickle.load(file)
        with open(f'{models_path}/lda.pkl', 'rb') as file:
            lda_model = pickle.load(file)
        with open(f'{models_path}/pca.pkl', 'rb') as file:
            pca = pickle.load(file)

        print('load_models::Getting coeff')
        clusters_coeff = pd.read_csv(f'{models_path}/clusters_coeff.csv')
        print('load_models::clusters coeff')
        print(clusters_coeff)
        words_coeff = pd.read_csv(f'{models_path}/words_coeff.csv')
        print('load_models::words coeff')
        print(words_coeff)
        
        return clf_clusters, keras_model, kmeans, lda_model, pca, clusters_coeff, words_coeff

    except Exception as e:
        print(f"Error loading models or data: {e}")
        return None, None, None, None, None, None, None

def test_model(df_embeddings, df_cvs, test_index, lda_model, keras_model, clf_clusters, pca, mask, n_clusters, clusters_coeff, words_coeff):
    """
    This function evaluates a model
    """
    print('test_model')
    X_test = df_embeddings.copy()
    y_test = df_cvs.copy()['label']

    cluster_prediction = clf_clusters.predict(X_test['embedding'].to_list())
    X_test['cluster'] = cluster_prediction

    X = get_test_matrix(X_test, X_test['len'], n_clusters, clusters_coeff, words_coeff)

    print('Step::Apply PCA')
    X = pca.transform(X)

    # print('Step::Apply mask')
    # X_pro = X.loc[:, mask]

    print('Step::Apply LDA')
    # X_points = lda_model.transform(np.array(X_pro))
    X_points = lda_model.transform(np.array(X))

    results = keras_model.predict(X_points)
    print("Resultados:",results)
    return results, X_test