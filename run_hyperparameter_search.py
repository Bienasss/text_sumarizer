import json
import argparse
from data_collector import NewsCollector
from hyperparameter_search import HyperparameterSearch

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Abstractive Summarization')
    parser.add_argument('--data', type=str, default='data/articles.json', help='Path to articles JSON file')
    parser.add_argument('--model', type=str, default='T5', choices=['T5', 'BART'], help='Model to search')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of articles to test')
    
    args = parser.parse_args()
    
    with open(args.data, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles")
    print(f"Testing {args.num_samples} articles with {args.model} model")
    
    search = HyperparameterSearch()
    results = search.search_hyperparameters(articles, model_type=args.model, num_samples=args.num_samples)
    
    print("\n" + "="*50)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*50)
    
    print(f"\nBest Configuration:")
    best = results['best_config']
    print(f"  Config: {best['config']}")
    print(f"  Avg ROUGE-1: {best['avg_rouge1']:.4f}")
    print(f"  Avg ROUGE-2: {best['avg_rouge2']:.4f}")
    print(f"  Avg ROUGE-L: {best['avg_rougeL']:.4f}")
    print(f"  Avg Time: {best['avg_time']:.4f}s")
    
    output_file = f"results/hyperparameter_search_{args.model.lower()}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

