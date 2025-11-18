# Text Summarizer

Text summarization project implementing both extractive and abstractive methods with ROUGE evaluation.

## Features

- **Extractive Methods:**
  - TF-IDF sentence ranking
  - TextRank algorithm
  - Lead-k baseline

- **Abstractive Methods:**
  - BART (facebook/bart-large-cnn)
  - T5 (t5-small)

- **Evaluation:**
  - ROUGE-1, ROUGE-2, ROUGE-L metrics
  - Batch evaluation and comparison

- **Data Collection:**
  - Real news articles from BBC and The Guardian and Fox
  - Automatic HTML/URL/emoji removal
  - Sentence segmentation

## Installation

```bash
pip install -r requirements.txt
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Usage

### Collect News Articles

```bash
python main.py --collect
```

This will collect articles from BBC and The Guardian and save them to `data/articles.json`.

### Run Evaluation

```bash
python main.py --data data/articles.json --num-samples 200
```

This will:
1. Load articles from the specified file
2. Split into train/test sets (80/20)
3. Evaluate all summarization methods on the test set
4. Generate a report in `results/report.txt`
5. Save detailed results to `results/detailed_results.csv`

### Interactive Demo

Open `demo.ipynb` in Jupyter Notebook for an interactive demonstration.

## Project Structure

```
text_sumarizer/
├── data_collector.py          # News article collection
├── preprocessing.py            # Text preprocessing utilities
├── extractive_summarizer.py   # Extractive methods (TF-IDF, TextRank, Lead-k)
├── abstractive_summarizer.py  # Abstractive methods (BART, T5)
├── evaluation.py              # ROUGE evaluation
├── main.py                    # Main evaluation script
├── demo.ipynb                 # Interactive demo notebook
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.10+
- See `requirements.txt` for full list of dependencies

## Data

The project uses real news articles from:
- BBC News
- The Guardian

Articles are collected automatically and stored in JSON format with title, text, URL, and source fields.

## Results

Evaluation results are saved to:
- `results/report.txt` - Summary report with average ROUGE scores
- `results/detailed_results.csv` - Detailed per-article results

## Notes

- First run will download transformer models (may take time)
- Abstractive models require significant memory
- CPU mode is supported but GPU is recommended for abstractive methods