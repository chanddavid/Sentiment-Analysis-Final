# model/data_cleaning.py
import os
import re
import tarfile
import pandas as pd
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'nor', 'without', 'neither', 'none', 'nobody', 'nothing', 'cannot', "couldn", "wouldn", "shouldn", "won", "don", "didn", "isn", "aren", "wasn", "weren"}
stop_words = stop_words - negations
extra_stopwords = {'br', 'u', 'im', 'dont', 'didnt', 'couldnt', 'wouldnt'}
stop_words.update(extra_stopwords)

lemmatizer = WordNetLemmatizer()

def extract_dataset():
    """Extract the dataset if not already extracted."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    tar_path = os.path.join(project_root, 'data', 'domain_sentiment_data.tar.gz')
    extract_path = os.path.join(project_root, 'data')
    
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Dataset not found at {tar_path}")
    
    # After extraction, we expect a folder 'sorted_data_acl' inside data/
    expected_folder = os.path.join(extract_path, 'sorted_data_acl')
    if not os.path.exists(expected_folder):
        print(f"Extracting dataset from {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def load_reviews():
    """Load all labelled positive/negative reviews from extracted dataset."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(project_root, 'data', 'sorted_data_acl')
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Extracted dataset folder not found at {data_root}. Run extract_dataset() first.")
    
    review_data = []
    label_map = {"positive.review": 1, "negative.review": 0}
    
    for root, _, files in os.walk(data_root):
        for file_name in files:
            if file_name in label_map:
                label = label_map[file_name]
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                matches = re.findall(r"<review_text>(.*?)</review_text>", content, re.DOTALL)
                for rev in matches:
                    review_data.append({
                        "review": rev.strip(),
                        "sentiment": label,
                        "source_file": file_path
                    })
    
    df = pd.DataFrame(review_data)
    print(f"Loaded {len(df)} reviews.")
    if len(df) == 0:
        raise ValueError("No reviews loaded. Check dataset structure.")
    return df

def prepare_data():
    extract_dataset()
    df = load_reviews()
    print(f"Raw dataset size: {len(df)}")
    df["clean_review"] = df["review"].apply(clean_text)
    df["processed_review"] = df["clean_review"].apply(preprocess_text)
    df["word_count"] = df["processed_review"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 5].reset_index(drop=True)
    print(f"After cleaning and outlier removal: {len(df)} reviews")
    return df