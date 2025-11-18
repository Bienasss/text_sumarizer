# Text Summarization Evaluation Report

## Executive Summary

This report presents the evaluation results of extractive and abstractive text summarization methods on a dataset of news articles.

## Methods Evaluated

### Extractive Methods

1. **TF-IDF Summarizer**: Ranks sentences based on TF-IDF scores
2. **TextRank Summarizer**: Uses graph-based ranking algorithm similar to PageRank
3. **Lead-k Summarizer**: Baseline method that selects the first k sentences

### Abstractive Methods

1. **BART**: Facebook's BART-large-CNN model for abstractive summarization
2. **T5**: Google's T5-small model for text-to-text generation

## Dataset

- **Source**: BBC News and The Guardian
- **Total Articles**: [Number]
- **Test Set**: [Number] articles (20% of total)
- **Average Article Length**: [Number] characters
- **Average Sentences per Article**: [Number]

## Evaluation Metrics

All methods were evaluated using ROUGE metrics:
- **ROUGE-1**: Overlap of unigrams between reference and generated summaries
- **ROUGE-2**: Overlap of bigrams between reference and generated summaries
- **ROUGE-L**: Longest Common Subsequence (LCS) based metric

## Results

### Average ROUGE Scores

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|--------|------------|------------|------------|
| TF-IDF | [Score] | [Score] | [Score] |
| TextRank | [Score] | [Score] | [Score] |
| Lead-3 | [Score] | [Score] | [Score] |
| BART | [Score] | [Score] | [Score] |
| T5 | [Score] | [Score] | [Score] |

### Detailed Analysis

[Analysis of results will be generated automatically]

## Conclusions

[Conclusions will be generated based on results]

## Recommendations

[Recommendations for future work]

