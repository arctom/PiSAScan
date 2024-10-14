import cv2
import math
import multiprocessing
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

# Setting the path to the Tesseract command
pytesseract.pytesseract.tesseract_cmd = '/usr/local/opt/tesseract/bin/tesseract'

# Initialize constants
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
nltk.download('stopwords')
wnl = WordNetLemmatizer()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
words = set(nltk.corpus.words.words())
stop_words = set(nltk.corpus.stopwords.words('english'))

# Functions

def concat_chunks(translated_list):
    """Concatenate list of translated chunks into a single string."""
    return "".join(translated_list)

def convert2txt(input_path, txt_path, poppler_path):
    """
    Convert PDF and DOC/DOCX files in specified directories to text format.

    Args:
    input_path (str): Directory containing folders of PDFs and DOCs.
    txt_path (str): Output directory for saving text files.
    poppler_path (str): Path to the Poppler library binaries.
    """
    dir_list = [f for f in os.listdir(input_path) if f[0] != '.']
    file_info_list = [(folder, cv, input_path, txt_path, poppler_path) for folder in dir_list for cv in os.listdir(os.path.join(input_path, folder)) if cv[0] != '.']

    with multiprocessing.Pool() as pool:
        pool.map(worker_convert, file_info_list)

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
            params = yaml.load(f, Loader = yaml.FullLoader)
        return project_root, params
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return project_root, {}
    
def load_file(file_info):
    """Process a single file to extract embeddings and text."""
    file, dir, label, index, num_words = file_info
    try:
        path = os.path.join(dir, file)
        print(f'Processing file: {path}')
        with open (path, 'r', encoding='utf-8', errors='ignore') as tmp:
            embeddings, text = preprocess_by_fixed_words(tmp.read(), label, index, num_words)
            return embeddings,{'label': label, 'index': index, 'text': text}, index + 1 
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return [], "", index

def load_data(root_dir, index = 0, label = '', num_words=10):
    """
    Load and process data from text files, returning embeddings and processed data frames.

    Args:
    dir (str): Directory to load data from.
    index (str): Index to start from for labelling.
    label (str): Label to assign to the data.
    num_words (int): Number of words to consider for each chunk.

    Returns:
    Tuple: A tuple containing embeddings, data frames, and the last index used.
    """
    df_embeddings, df_cvs = [], []

    file_info_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            label = dirname
            for filename in os.listdir(os.path.join(dirpath, dirname)):
                if not filename.startswith('.'):
                    file_info_list.append((filename, os.path.join(dirpath, dirname), label, index + len(file_info_list), num_words))
            
    with multiprocessing.Pool() as pool:
        results = pool.map(load_file, file_info_list)

    for result in results:
        embeddings, cv, new_index = result
        if embeddings:
            df_embeddings.extend(embeddings)
            df_cvs.append(cv)
            index = new_index
    
    return df_embeddings, df_cvs, index

def preprocess_by_fixed_words(text, label, index, num_words=10):
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

def process_doc(file_path, txt_file_path):
    """Process DOC/DOCX files to extract text."""
    try:
        document = Document(file_path)
        document.SaveToFile(txt_file_path, FileFormat.Txt)
        document.Close()
        # Clean up unnecessary top lines from DOC/DOCX text output
        with open(txt_file_path, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(txt_file_path, 'w') as fout:
            fout.writelines(data[1:])
    except Exception as e:
        print(f"Error processing DOC/DOCX file {file_path}: {e}")

def process_pdf(file_path, txt_file_path, poppler_path):
    """Process PDF files to extract text."""
    try:
        pages = convert_from_path(file_path, dpi=200, poppler_path=poppler_path)
        for page in pages:
            img = np.array(page.convert('RGB')).astype(np.uint8)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_text = pytesseract.image_to_string(thr, lang='spa+eng')
            with open(txt_file_path, 'a') as text_file:
                text_file.write(image_text)
    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")

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
    
def worker_convert(file_info):
    folder, cv, input_path, txt_path, poppler_path = file_info
    print(f'Converting {cv}')
    folder_path = os.path.join(input_path, folder)
    file_name, ext = os.path.splitext(cv)
    file_path = os.path.join(folder_path, cv)
    txt_file_path = os.path.join(txt_path, folder, file_name + '.txt')
    if ext.lower() in ['.pdf']:
        process_pdf(file_path, txt_file_path, poppler_path)
    elif ext.lower() in ['.docx', '.doc']:
        process_doc(file_path, txt_file_path)
    else:
        print(f'Format {ext} is not supported.')


if __name__ == "__main__":
    # Main execution block to run the script functionalities
    try:
        root, config = load_config()
        poppler_path = '/usr/local/opt/poppler/bin'
        input_path = os.path.join(root, config['data']['input_path'])
        txt_path = os.path.join(root, config['txt'])
        output_path = os.path.join(root, config['data']['output_path'])

        if not input_path or not txt_path or not output_path:
            raise ValueError("Input path, text path, or output path is missing in the configuration file.")
        
        convert2txt(input_path, txt_path, poppler_path)
        df_embeddings, df_cvs, _ = load_data(txt_path , num_words=7)
        save_to_excel(df_embeddings, df_cvs, output_path)
    except Exception as e:
        print(f"Error in main execution block: {e}")