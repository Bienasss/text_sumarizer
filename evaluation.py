from rouge_score import rouge_scorer
from typing import List, Dict
import numpy as np

class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate(self, reference: str, candidate: str) -> Dict[str, float]:
        scores = self.scorer.score(reference, candidate)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
    
    def evaluate_batch(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        all_scores = {
            'rouge1_precision': [],
            'rouge1_recall': [],
            'rouge1_f1': [],
            'rouge2_precision': [],
            'rouge2_recall': [],
            'rouge2_f1': [],
            'rougeL_precision': [],
            'rougeL_recall': [],
            'rougeL_f1': [],
        }
        
        for ref, cand in zip(references, candidates):
            scores = self.evaluate(ref, cand)
            for key in all_scores:
                all_scores[key].append(scores[key])
        
        avg_scores = {}
        for key in all_scores:
            avg_scores[key] = np.mean(all_scores[key])
            avg_scores[f'{key}_std'] = np.std(all_scores[key])
        
        return avg_scores

