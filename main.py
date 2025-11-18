import json
import os
import argparse
import time
from typing import List, Dict
import pandas as pd
from data_collector import NewsCollector
from preprocessing import TextPreprocessor
from extractive_summarizer import TFIDFSummarizer, TextRankSummarizer, LeadKSummarizer
from abstractive_summarizer import AbstractiveSummarizer, T5Summarizer
from evaluation import RougeEvaluator
from error_analysis import ErrorAnalyzer

def load_data(data_path: str) -> List[Dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_data(articles: List[Dict], train_ratio: float = 0.8) -> tuple:
    split_idx = int(len(articles) * train_ratio)
    return articles[:split_idx], articles[split_idx:]

def evaluate_summarizers(articles: List[Dict], num_samples: int = None):
    if num_samples:
        articles = articles[:num_samples]
    
    preprocessor = TextPreprocessor()
    
    extractive_methods = {
        'TF-IDF': TFIDFSummarizer(preprocessor),
        'TextRank': TextRankSummarizer(preprocessor),
        'Lead-3': LeadKSummarizer(preprocessor)
    }
    
    abstractive_methods = {
        'BART': AbstractiveSummarizer(model_name="facebook/bart-large-cnn"),
        'T5': T5Summarizer(model_name="t5-small")
    }
    
    evaluator = RougeEvaluator()
    
    results = []
    
    for article in articles:
        text = article.get('text', '')
        if len(text) < 200:
            continue
        
        processed_text = preprocessor.preprocess(text)
        sentences = preprocessor.segment_sentences(processed_text)
        
        if len(sentences) < 3:
            continue
        
        reference_summary = ' '.join(sentences[:3])
        
        article_results = {
            'article_id': article.get('url', ''),
            'title': article.get('title', ''),
            'text_length': len(text),
            'num_sentences': len(sentences)
        }
        
        for method_name, summarizer in extractive_methods.items():
            try:
                start_time = time.time()
                summary = summarizer.summarize(processed_text, num_sentences=3)
                elapsed_time = time.time() - start_time
                scores = evaluator.evaluate(reference_summary, summary)
                
                article_results[f'{method_name}_rouge1_f1'] = scores['rouge1_f1']
                article_results[f'{method_name}_rouge2_f1'] = scores['rouge2_f1']
                article_results[f'{method_name}_rougeL_f1'] = scores['rougeL_f1']
                article_results[f'{method_name}_summary'] = summary
                article_results[f'{method_name}_time'] = elapsed_time
            except Exception as e:
                print(f"Error with {method_name}: {e}")
        
        for method_name, summarizer in abstractive_methods.items():
            try:
                start_time = time.time()
                summary = summarizer.summarize(processed_text, max_length=150, min_length=50,
                                               num_beams=4, do_sample=False, no_repeat_ngram_size=3)
                elapsed_time = time.time() - start_time
                scores = evaluator.evaluate(reference_summary, summary)
                
                article_results[f'{method_name}_rouge1_f1'] = scores['rouge1_f1']
                article_results[f'{method_name}_rouge2_f1'] = scores['rouge2_f1']
                article_results[f'{method_name}_rougeL_f1'] = scores['rougeL_f1']
                article_results[f'{method_name}_summary'] = summary
                article_results[f'{method_name}_time'] = elapsed_time
            except Exception as e:
                print(f"Error with {method_name}: {e}")
        
        results.append(article_results)
        print(f"Processed {len(results)} articles")
    
    return results

def generate_report(results: List[Dict], output_path: str = "results/report.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(results)
    
    methods = ['TF-IDF', 'TextRank', 'Lead-3', 'BART', 'T5']
    
    error_analyzer = ErrorAnalyzer()
    error_analysis = error_analyzer.analyze_errors(df)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TEXT SUMMARIZATION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total articles evaluated: {len(results)}\n\n")
        
        f.write("ROUGE-1 F1 Scores:\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            col = f'{method}_rouge1_f1'
            if col in df.columns:
                mean_score = df[col].mean()
                std_score = df[col].std()
                f.write(f"{method:15s}: {mean_score:.4f} ± {std_score:.4f}\n")
        
        f.write("\nROUGE-2 F1 Scores:\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            col = f'{method}_rouge2_f1'
            if col in df.columns:
                mean_score = df[col].mean()
                std_score = df[col].std()
                f.write(f"{method:15s}: {mean_score:.4f} ± {std_score:.4f}\n")
        
        f.write("\nROUGE-L F1 Scores:\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            col = f'{method}_rougeL_f1'
            if col in df.columns:
                mean_score = df[col].mean()
                std_score = df[col].std()
                f.write(f"{method:15s}: {mean_score:.4f} ± {std_score:.4f}\n")
        
        f.write("\nAverage Processing Time (seconds):\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            time_col = f'{method}_time'
            if time_col in df.columns:
                mean_time = df[time_col].mean()
                f.write(f"{method:15s}: {mean_time:.4f}s\n")
        
        f.write("\nERROR ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            if method in error_analysis:
                analysis = error_analysis[method]
                f.write(f"\n{method}:\n")
                f.write(f"  Average ROUGE-1: {analysis['avg_score']:.4f}\n")
                f.write(f"  Low scores (<0.3): {analysis['num_low_scores']}\n")
                f.write(f"  High scores (>0.7): {analysis['num_high_scores']}\n")
                if analysis['error_examples']:
                    f.write(f"  Error examples: {len(analysis['error_examples'])} found\n")
    
    df.to_csv("results/detailed_results.csv", index=False, encoding='utf-8')
    
    error_examples_path = "results/error_examples.json"
    with open(error_examples_path, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"Report saved to {output_path}")
    print(f"Detailed results saved to results/detailed_results.csv")
    print(f"Error analysis saved to {error_examples_path}")

def main():
    parser = argparse.ArgumentParser(description='Text Summarization Evaluation')
    parser.add_argument('--collect', action='store_true', help='Collect news articles')
    parser.add_argument('--data', type=str, default='data/articles.json', help='Path to articles JSON file')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of articles to evaluate')
    parser.add_argument('--output', type=str, default='results/report.txt', help='Output report path')
    
    args = parser.parse_args()
    
    if args.collect:
        print("Collecting news articles...")
        collector = NewsCollector()
        bbc_articles = collector.collect_bbc_news(num_articles=500)
        guardian_articles = collector.collect_guardian_news(num_articles=400)
        fox_articles = collector.collect_fox_news(num_articles=300)
        all_articles = bbc_articles + guardian_articles + fox_articles
        collector.save_articles(all_articles, "articles.json")
        print(f"Collected {len(all_articles)} articles")
        print("Data collection complete. Run without --collect to evaluate.")
        return
    
    if not os.path.exists(args.data):
        print(f"Data file {args.data} not found. Use --collect to collect articles first.")
        return
    
    print("Loading articles...")
    articles = load_data(args.data)
    print(f"Loaded {len(articles)} articles")
    
    if args.num_samples:
        print(f"Limiting to {args.num_samples} articles for evaluation")
        articles = articles[:args.num_samples]
    
    train_articles, test_articles = split_data(articles, train_ratio=0.8)
    print(f"Train: {len(train_articles)}, Test: {len(test_articles)}")
    
    print("Evaluating summarizers on test set...")
    results = evaluate_summarizers(test_articles, num_samples=None)
    
    print("Generating report...")
    generate_report(results, args.output)
    
    print("Generating full report with visualizations...")
    from generate_full_report import generate_full_report
    generate_full_report()
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()

