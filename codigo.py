import cv2
import math
import nltk
import numpy as np
import os
import pandas as pd
import pytesseract
import re
import yaml
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from spire.doc import *
from spire.doc.common import *
from unidecode import unidecode
from tempfile import NamedTemporaryFile 

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
        root, config = load_config()
        poppler_path = 'D:/users/poppler-24.02.0/Library/bin'
        output_path = 'D:/Github/PiSAScan/data/output.xlsx'
        
        embeddings, processed_data = process_file(file_path, poppler_path=poppler_path)
        save_to_excel(embeddings, [processed_data], output_path)
        
        from main import show_results
        show_results(output_path)

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

def save_to_excel(embeddings, cvs, filename):
    """Save embeddings and CVs to an Excel file.""" 
    try:
        df_embeddings = pd.DataFrame(embeddings)
        df_cvs = pd.DataFrame(cvs)
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_embeddings.to_excel(writer, sheet_name='Embeddings', index=False)
            df_cvs.to_excel(writer, sheet_name='CVs', index=False)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to Excel file {filename}: {e}")

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
