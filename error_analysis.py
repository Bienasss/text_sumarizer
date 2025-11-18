import pandas as pd
from typing import List, Dict
from evaluation import RougeEvaluator

class ErrorAnalyzer:
    def __init__(self):
        self.evaluator = RougeEvaluator()
    
    def find_error_cases(self, results_df: pd.DataFrame, method: str, 
                        threshold: float = 0.3) -> List[Dict]:
        rouge_col = f'{method}_rouge1_f1'
        summary_col = f'{method}_summary'
        
        if rouge_col not in results_df.columns:
            return []
        
        error_cases = results_df[results_df[rouge_col] < threshold].copy()
        
        error_examples = []
        for idx, row in error_cases.head(5).iterrows():
            error_examples.append({
                'article_id': row.get('article_id', ''),
                'title': row.get('title', ''),
                'rouge1_score': row[rouge_col],
                'rouge2_score': row.get(f'{method}_rouge2_f1', 0),
                'rougeL_score': row.get(f'{method}_rougeL_f1', 0),
                'summary': row.get(summary_col, ''),
                'text_length': row.get('text_length', 0),
                'num_sentences': row.get('num_sentences', 0)
            })
        
        return error_examples
    
    def analyze_errors(self, results_df: pd.DataFrame) -> Dict:
        methods = ['TF-IDF', 'TextRank', 'Lead-3', 'BART', 'T5']
        
        analysis = {}
        
        for method in methods:
            rouge_col = f'{method}_rouge1_f1'
            if rouge_col not in results_df.columns:
                continue
            
            method_errors = self.find_error_cases(results_df, method)
            
            low_scores = results_df[results_df[rouge_col] < 0.3]
            high_scores = results_df[results_df[rouge_col] > 0.7]
            
            analysis[method] = {
                'error_examples': method_errors[:2],
                'num_low_scores': len(low_scores),
                'num_high_scores': len(high_scores),
                'avg_score': results_df[rouge_col].mean(),
                'min_score': results_df[rouge_col].min(),
                'max_score': results_df[rouge_col].max()
            }
        
        return analysis

