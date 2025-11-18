import nltk
import sys

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK punkt tokenizer already downloaded")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        print("Downloaded punkt tokenizer")
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK punkt_tab tokenizer already downloaded")
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)
        print("Downloaded punkt_tab tokenizer")

if __name__ == "__main__":
    print("Setting up text summarizer project...")
    download_nltk_data()
    print("Setup complete!")

