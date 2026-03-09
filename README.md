## Metrics Analysis Notebook

This repository contains a single Jupyter notebook, `Analysis.ipynb`, used to explore and analyze a collection of evaluation metrics for question–answering / LLM responses. The analysis operates on an `evaluation_results.csv` file containing model responses, reference answers, contextual fields, and a wide range of automatic metrics (e.g. token-based, BLEU/METEOR, BERTScore, NLI-style scores, embedding-based metrics, retrieval metrics, and ROUGE).

### Repository structure

- **`Analysis.ipynb`**: Main analysis notebook. Loads `evaluation_results.csv`, computes additional ROUGE scores, filters/cleans the data, and produces descriptive statistics and visualizations (such as distribution plots of correct vs. incorrect (`Analysis` = A/N) by `Aspect`, and per-metric summaries).
- **`evaluation_results.csv`** (expected): Input data file with columns including:
  - Text fields: `context`, `supporting_context`, `answer`, `Response`, `Analysis`, `Aspect`, `context_token_count`
  - Token/string metrics: `EM`, `Token_F1`, `Token_Precision`, `Token_Recall`, `Token_Accuracy`, `Fuzzy_Accuracy`, `LCS`, `Jaccard_Similarity`, `ROUGE_1`, `ROUGE_2`, `ROUGE_L`, `METEOR`, `BLEU`
  - Embedding / semantic metrics: `BERTScore_P`, `BERTScore_R`, `BERTScore_F1`, `Meaning_Accuracy_LLM`, `Meaning_Accuracy_Emb`, `Emb_Precision`, `Emb_Recall`, `Emb_F1`
  - Retrieval / faithfulness metrics: `Faithfulness`, `Context_Precision`, `Context_Recall`, `Retrieval_Precision`, `Retrieval_Recall`, `Retrieval_F1`, `Retrieval_TP`, `Retrieval_FP`, `Retrieval_FN`, `Answer_Relevance`, `NLI_Precision`, `NLI_Recall`, `NLI_F1`, `GEval_Precision`, `GEval_Recall`, `GEval_F1`, `Is_Error`

> Note: The notebook assumes that `evaluation_results.csv` is present in the same directory as `Analysis.ipynb`.

### Environment & dependencies

The notebook is written for Python and uses common scientific/ML libraries. A minimal environment will need:

- Python 3.9+ (recommended)
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `rouge-score`

You can install these into a virtual environment with:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install pandas numpy matplotlib seaborn scipy scikit-learn rouge-score
```

### How to run the analysis

1. **Clone / open this folder** in your environment or IDE (e.g. Cursor, VS Code, JupyterLab).
2. **Ensure the data file is available**:
   - Place `evaluation_results.csv` in the root of this repo (alongside `Analysis.ipynb`), or adjust the path in the first data-loading cell of the notebook.
3. **Create and activate a virtual environment** (optional but recommended), then install the dependencies listed above.
4. **Start a Jupyter server**:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```
5. **Open `Analysis.ipynb`** and run the cells from top to bottom.

### What the notebook does

At a high level, the notebook:

- Imports the CSV into a `pandas` DataFrame.
- Computes ROUGE-1/2/L F-measure scores using `rouge-score` from each `Response` to its corresponding `answer`.
- Selects and casts a curated subset of metric columns to numeric types.
- Optionally filters rows for certain aspects or labels (e.g. dropping specific outliers, handling missing context/support).
- Computes summary statistics (e.g. `.describe()` per metric).
- Visualizes:
  - Distribution of correct vs. incorrect (`Analysis` A/N) counts per `Aspect`.
  - Additional per-metric summaries and comparisons (e.g. per-metric means grouped by `Aspect`/`Analysis`).

The notebook is intended as an exploratory analysis tool to compare different evaluation metrics and understand how they behave across aspects and correctness labels.

### Extending the analysis

Some ideas for extending or customizing the notebook:

- Add new metrics to `evaluation_results.csv` and incorporate them into the `METRICS` list to include in aggregations and plots.
- Create ROC curves or calibration plots for specific metrics (e.g. BERTScore, NLI, faithfulness) to better understand their discriminative power between A vs. N.
- Perform correlation analysis between metrics to identify redundancy or complementarities.
- Segment the analysis by additional dimensions (e.g. difficulty buckets, context length, domain) if those columns are available in the CSV.

### Reproducibility notes

- If you modify the CSV schema (e.g. rename columns or change types), be sure to update:
  - The `cols` list used to subset the DataFrame.
  - The `METRICS` list and any code that assumes specific column names.
- Randomness is minimal (most metrics are deterministic given the data), but if you introduce stochastic components, consider setting random seeds for reproducibility.

