import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import List
from preprocessing import TextPreprocessor

class TFIDFSummarizer:
    def __init__(self, preprocessor: TextPreprocessor = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = self.preprocessor.segment_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            top_indices = np.argsort(sentence_scores)[-num_sentences:]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
        except:
            return ' '.join(sentences[:num_sentences])

class TextRankSummarizer:
    def __init__(self, preprocessor: TextPreprocessor = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except:
            n = len(sentences)
            return np.eye(n)
    
    def _calculate_pagerank(self, similarity_matrix: np.ndarray, damping: float = 0.85) -> np.ndarray:
        similarity_matrix = np.maximum(similarity_matrix, 0)
        np.fill_diagonal(similarity_matrix, 0)
        
        row_sums = similarity_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums[:, np.newaxis]
        
        n = len(normalized_matrix)
        pagerank = np.ones(n) / n
        
        for _ in range(100):
            new_pagerank = (1 - damping) / n + damping * normalized_matrix.T.dot(pagerank)
            if np.allclose(pagerank, new_pagerank):
                break
            pagerank = new_pagerank
        
        return pagerank
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = self.preprocessor.segment_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            similarity_matrix = self._build_similarity_matrix(sentences)
            scores = self._calculate_pagerank(similarity_matrix)
            
            top_indices = np.argsort(scores)[-num_sentences:]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
        except:
            return ' '.join(sentences[:num_sentences])

class LeadKSummarizer:
    def __init__(self, preprocessor: TextPreprocessor = None):
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = self.preprocessor.segment_sentences(text)
        return ' '.join(sentences[:num_sentences])

