# AG News Multiclass Text Classification

An end-to-end NLP project comparing a classical TF-IDF + Linear SVM classifier with a fine-tuned DistilBERT model on the AG News dataset.

## Dataset

- **Dataset:** AG News Classification
- **Training samples:** 120,000
- **Test samples:** 7,600
- **Classes:** World, Sports, Business, Sci/Tech
- **Class distribution:** balanced across the four classes

## Benchmark Results

The following results are from the experiments represented in this repository.

| Model | Approach | Test Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Uniform Random Baseline | Random baseline | 25.13% | 0.2500 | 0.2513 | 0.2512 |
| Linear SVM (Tuned) | TF-IDF + LinearSVC | 90.71% | 0.9070 | 0.9071 | 0.9067 |
| DistilBERT (2 Epochs) | Fine-tuned Transformer | 91.03% | 0.9107 | 0.9103 | 0.9102 |

> **Note:** The metrics above are retained as the reported experimental results. The exact training configuration used to produce those numbers should be kept with the corresponding experiment artifacts/notebook.

## Project Structure

```text
.
├── src/
│   ├── preprocess.py
│   └── predict.py
├── requirements.txt
├── tests/
│   └── test_preprocess.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/payamhabibi/nlp-news-classification.git
cd nlp-news-classification

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Tests

```bash
pytest -q
```

## Inference

The prediction pipeline expects two locally available Joblib artifacts:

- `tfidf_vectorizer.joblib`
- a trained `LinearSVC` model, e.g. `best_linear_svc.joblib`

These model artifacts are intentionally excluded from Git via `.gitignore` because binary model files are not part of the source repository.

Example:

```python
from src.predict import NewsClassificationPipeline

pipeline = NewsClassificationPipeline(
    vectorizer_path="models/tfidf_vectorizer.joblib",
    model_path="models/best_linear_svc.joblib",
)

sample = ["NASA successfully launches a new deep-space telescope."]
print(pipeline.predict(sample))
```

If the required model artifacts are not present, inference cannot run until the artifacts are generated from the training experiment.

## Preprocessing

The shared preprocessing function:

1. HTML-unescapes text.
2. Removes URLs and HTML tags.
3. Removes backslashes.
4. Keeps English alphabetic characters and whitespace.
5. Lowercases text.
6. Removes tokens shorter than three characters.

The preprocessing is intentionally simple and is designed for the English AG News corpus.

## Model Interpretability

For the Linear SVM pipeline, the prediction output includes the highest-weight present TF-IDF features for the predicted class. These are **feature contributions/evidence keywords**, not a general explanation of the model and not a causal interpretation.

## Reproducibility

For reproducible experiments, keep the following together with each experiment:

- random seed
- train/validation/test split
- TF-IDF configuration
- SVM hyperparameters
- Transformer checkpoint
- tokenizer configuration
- learning rate
- batch size
- number of epochs
- maximum sequence length
- evaluation metric definitions

The repository deliberately does not commit trained model binaries.

## License

This project is released under the MIT License.
