import re
import nltk
from typing import List
import html

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.sentence_tokenizer = nltk.tokenize.sent_tokenize
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
    
    def remove_html_tags(self, text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        return self.url_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        return self.emoji_pattern.sub('', text)
    
    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def preprocess(self, text: str) -> str:
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.normalize_whitespace(text)
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        sentences = self.sentence_tokenizer(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def preprocess_and_segment(self, text: str) -> List[str]:
        processed_text = self.preprocess(text)
        sentences = self.segment_sentences(processed_text)
        return sentences

