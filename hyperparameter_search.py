import time
from typing import List, Dict
from abstractive_summarizer import AbstractiveSummarizer, T5Summarizer
from evaluation import RougeEvaluator
from preprocessing import TextPreprocessor

class HyperparameterSearch:
    def __init__(self):
        self.evaluator = RougeEvaluator()
        self.preprocessor = TextPreprocessor()
    
    def search_hyperparameters(self, articles: List[Dict], model_type: str = "T5", 
                              num_samples: int = 10) -> Dict:
        if model_type == "T5":
            summarizer_class = T5Summarizer
        else:
            summarizer_class = AbstractiveSummarizer
        
        hyperparameter_configs = [
            {"max_length": 150, "min_length": 50, "num_beams": 4, "do_sample": False, "no_repeat_ngram_size": 3},
            {"max_length": 150, "min_length": 50, "num_beams": 2, "do_sample": False, "no_repeat_ngram_size": 3},
            {"max_length": 150, "min_length": 50, "num_beams": 6, "do_sample": False, "no_repeat_ngram_size": 3},
            {"max_length": 150, "min_length": 50, "num_beams": 4, "do_sample": True, "no_repeat_ngram_size": 3},
            {"max_length": 150, "min_length": 50, "num_beams": 4, "do_sample": False, "no_repeat_ngram_size": 2},
            {"max_length": 150, "min_length": 50, "num_beams": 4, "do_sample": False, "no_repeat_ngram_size": 4},
            {"max_length": 120, "min_length": 40, "num_beams": 4, "do_sample": False, "no_repeat_ngram_size": 3},
            {"max_length": 180, "min_length": 60, "num_beams": 4, "do_sample": False, "no_repeat_ngram_size": 3},
        ]
        
        results = []
        
        for config in hyperparameter_configs:
            print(f"Testing config: {config}")
            summarizer = summarizer_class(model_name="t5-small" if model_type == "T5" else "facebook/bart-large-cnn")
            
            config_results = {
                "config": config,
                "rouge1_scores": [],
                "rouge2_scores": [],
                "rougeL_scores": [],
                "times": []
            }
            
            for i, article in enumerate(articles[:num_samples]):
                text = article.get('text', '')
                if len(text) < 200:
                    continue
                
                processed_text = self.preprocessor.preprocess(text)
                sentences = self.preprocessor.segment_sentences(processed_text)
                
                if len(sentences) < 3:
                    continue
                
                reference_summary = ' '.join(sentences[:3])
                
                start_time = time.time()
                summary = summarizer.summarize(processed_text, **config)
                elapsed_time = time.time() - start_time
                
                scores = self.evaluator.evaluate(reference_summary, summary)
                
                config_results["rouge1_scores"].append(scores['rouge1_f1'])
                config_results["rouge2_scores"].append(scores['rouge2_f1'])
                config_results["rougeL_scores"].append(scores['rougeL_f1'])
                config_results["times"].append(elapsed_time)
            
            config_results["avg_rouge1"] = sum(config_results["rouge1_scores"]) / len(config_results["rouge1_scores"]) if config_results["rouge1_scores"] else 0
            config_results["avg_rouge2"] = sum(config_results["rouge2_scores"]) / len(config_results["rouge2_scores"]) if config_results["rouge2_scores"] else 0
            config_results["avg_rougeL"] = sum(config_results["rougeL_scores"]) / len(config_results["rougeL_scores"]) if config_results["rougeL_scores"] else 0
            config_results["avg_time"] = sum(config_results["times"]) / len(config_results["times"]) if config_results["times"] else 0
            
            results.append(config_results)
            print(f"  Avg ROUGE-1: {config_results['avg_rouge1']:.4f}, Avg Time: {config_results['avg_time']:.2f}s")
        
        best_config = max(results, key=lambda x: x['avg_rouge1'])
        
        return {
            "all_results": results,
            "best_config": best_config
        }

