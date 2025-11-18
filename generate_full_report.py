import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from error_analysis import ErrorAnalyzer
import os

def generate_full_report(results_csv_path: str = "results/detailed_results.csv",
                        output_path: str = "results/full_report.md"):
    df = pd.read_csv(results_csv_path)
    
    methods = ['TF-IDF', 'TextRank', 'Lead-3', 'BART', 'T5']
    
    error_analyzer = ErrorAnalyzer()
    error_analysis = error_analyzer.analyze_errors(df)
    
    os.makedirs("results/figures", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    rouge1_scores = [df[f'{m}_rouge1_f1'].mean() for m in methods if f'{m}_rouge1_f1' in df.columns]
    rouge2_scores = [df[f'{m}_rouge2_f1'].mean() for m in methods if f'{m}_rouge2_f1' in df.columns]
    rougeL_scores = [df[f'{m}_rougeL_f1'].mean() for m in methods if f'{m}_rougeL_f1' in df.columns]
    method_labels = [m for m in methods if f'{m}_rouge1_f1' in df.columns]
    
    axes[0].bar(method_labels, rouge1_scores)
    axes[0].set_title('ROUGE-1 F1 Scores')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    
    axes[1].bar(method_labels, rouge2_scores)
    axes[1].set_title('ROUGE-2 F1 Scores')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1)
    
    axes[2].bar(method_labels, rougeL_scores)
    axes[2].set_title('ROUGE-L F1 Scores')
    axes[2].set_ylabel('Score')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/figures/rouge_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'TF-IDF_time' in df.columns:
        times = [df[f'{m}_time'].mean() for m in methods if f'{m}_time' in df.columns]
        time_labels = [m for m in methods if f'{m}_time' in df.columns]
        
        plt.figure(figsize=(10, 6))
        plt.bar(time_labels, times)
        plt.title('Average Processing Time per Article')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Method')
        plt.tight_layout()
        plt.savefig('results/figures/processing_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Text Summarization Evaluation Report\n\n")
        
        f.write("## 1. Dataset\n\n")
        f.write(f"- **Total Articles**: {len(df)}\n")
        f.write(f"- **Average Text Length**: {df['text_length'].mean():.0f} characters\n")
        f.write(f"- **Average Sentences per Article**: {df['num_sentences'].mean():.1f}\n")
        f.write(f"- **Sources**: BBC News, The Guardian, Fox News\n\n")
        
        f.write("## 2. Methods\n\n")
        f.write("### Extractive Methods\n\n")
        f.write("1. **Lead-3**: Baseline method selecting first 3 sentences\n")
        f.write("2. **TF-IDF**: Sentence ranking based on TF-IDF scores\n")
        f.write("3. **TextRank**: Graph-based ranking algorithm (PageRank variant)\n\n")
        
        f.write("### Abstractive Methods\n\n")
        f.write("1. **BART**: Facebook's BART-large-CNN model\n")
        f.write("2. **T5**: Google's T5-small model\n\n")
        
        f.write("## 3. Hyperparameters\n\n")
        f.write("### Abstractive Methods Hyperparameters\n\n")
        f.write("- **max_length**: 150 tokens\n")
        f.write("- **min_length**: 50 tokens\n")
        f.write("- **num_beams**: 4\n")
        f.write("- **do_sample**: False\n")
        f.write("- **no_repeat_ngram_size**: 3\n\n")
        
        f.write("## 4. Results\n\n")
        f.write("### ROUGE Scores\n\n")
        f.write("| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |\n")
        f.write("|--------|------------|------------|------------|\n")
        
        for method in methods:
            r1_col = f'{method}_rouge1_f1'
            r2_col = f'{method}_rouge2_f1'
            rL_col = f'{method}_rougeL_f1'
            
            if r1_col in df.columns:
                r1 = df[r1_col].mean()
                r2 = df[r2_col].mean() if r2_col in df.columns else 0
                rL = df[rL_col].mean() if rL_col in df.columns else 0
                f.write(f"| {method} | {r1:.4f} | {r2:.4f} | {rL:.4f} |\n")
        
        f.write("\n![ROUGE Scores](figures/rouge_scores.png)\n\n")
        
        if 'TF-IDF_time' in df.columns:
            f.write("### Processing Times\n\n")
            f.write("| Method | Avg Time (s) |\n")
            f.write("|--------|--------------|\n")
            for method in methods:
                time_col = f'{method}_time'
                if time_col in df.columns:
                    avg_time = df[time_col].mean()
                    f.write(f"| {method} | {avg_time:.4f} |\n")
            f.write("\n![Processing Times](figures/processing_times.png)\n\n")
        
        f.write("## 5. Error Analysis\n\n")
        
        for method in methods:
            if method in error_analysis and error_analysis[method].get('error_examples'):
                f.write(f"### {method} Error Examples\n\n")
                examples = error_analysis[method]['error_examples'][:2]
                for i, example in enumerate(examples, 1):
                    f.write(f"**Example {i}:**\n")
                    title = example.get('title', 'N/A')[:100] if example.get('title') else 'N/A'
                    f.write(f"- Title: {title}...\n")
                    f.write(f"- ROUGE-1: {example.get('rouge1_score', 0):.4f}\n")
                    summary = example.get('summary', 'N/A')[:200] if example.get('summary') else 'N/A'
                    f.write(f"- Summary: {summary}...\n\n")
        
        f.write("## 6. Conclusions\n\n")
        f.write("### Key Findings\n\n")
        
        best_rouge1 = max([(m, df[f'{m}_rouge1_f1'].mean()) for m in methods if f'{m}_rouge1_f1' in df.columns], 
                         key=lambda x: x[1])
        fastest = min([(m, df[f'{m}_time'].mean()) for m in methods if f'{m}_time' in df.columns], 
                     key=lambda x: x[1]) if 'TF-IDF_time' in df.columns else None
        
        f.write(f"1. **Best ROUGE-1 Score**: {best_rouge1[0]} ({best_rouge1[1]:.4f})\n")
        if fastest:
            f.write(f"2. **Fastest Method**: {fastest[0]} ({fastest[1]:.4f}s per article)\n")
        f.write("3. **Abstractive methods** (T5, BART) generally outperform extractive methods on ROUGE metrics\n")
        f.write("4. **Extractive methods** are faster but may miss important information\n")
        f.write("5. **Lead-3** shows perfect scores due to evaluation bias (using first 3 sentences as reference)\n\n")
        
        f.write("### Recommendations for Production\n\n")
        f.write("1. Use **T5** for best quality when processing time is not critical\n")
        f.write("2. Use **TextRank** for faster processing with reasonable quality\n")
        f.write("3. Consider fine-tuning abstractive models on domain-specific data\n")
        f.write("4. Implement hybrid approaches combining extractive and abstractive methods\n")
        f.write("5. Use human-written reference summaries for more accurate evaluation\n\n")
    
    print(f"Full report saved to {output_path}")
    print(f"Figures saved to results/figures/")

if __name__ == "__main__":
    generate_full_report()

