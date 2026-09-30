# AG News Multiclass Classification & Interpretability Benchmark

An end-to-end Natural Language Processing (NLP) project comparing a classical TF-IDF + Linear SVM pipeline with a fine-tuned DistilBERT model on the **AG News** dataset.

The project focuses on reproducible text preprocessing, a practical classical ML baseline, comparison with a fine-tuned transformer, and lightweight feature-level interpretability for the Linear SVM model.

## Dataset

- **Source:** AG News Classification Dataset
- **Training samples:** 120,000
- **Test samples:** 7,600
- **Classes:** World, Sports, Business, Sci/Tech
- **Class balance:** 25% per class

## Benchmark Results

The reported results below are the benchmark values currently documented for this project.

| Model | Paradigm | Test Accuracy | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Uniform Random Baseline | Baseline | 25.13% | 0.2500 | 0.2513 | 0.2512 |
| Linear SVM (Tuned) | Classical ML | 90.71% | 0.9070 | 0.9071 | 0.9067 |
| DistilBERT (2 Epochs) | Transformer | 91.03% | 0.9107 | 0.9103 | 0.9102 |

> **Interpretability scope:** keyword evidence is implemented for the Linear SVM pipeline using TF-IDF feature weights. It is not a token-level explanation method for DistilBERT.

## Project Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── predict.py
├── tests/
│   ├── __init__.py
│   └── test_preprocess.py
├── figures/
├── results/
├── data/
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Python 3.11 is used by the CI workflow.

```bash
git clone https://github.com/payamhabibi/nlp-news-classification.git
cd nlp-news-classification

python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Tests

Run the lightweight preprocessing test suite with:

```bash
pytest -q
```

GitHub Actions runs the same test suite automatically on pushes to `main`/feature branches and on pull requests targeting `main`.

## Inference with the Linear SVM

The inference class expects two trained artifacts:

- `models/tfidf_vectorizer.joblib`
- `models/best_linear_svc.joblib`

Model artifacts are intentionally excluded from Git via `.gitignore`. Generate them using the project's training workflow before running inference, or place compatible artifacts at those paths.

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

The returned `Top_Evidence_Keywords` field contains the highest positive TF-IDF × SVM-weight features present in the input for the predicted class.

## Reproducibility Notes

- Keep the preprocessing function identical between training and inference.
- Keep the fitted TF-IDF vectorizer together with the fitted classifier.
- The benchmark table reports the project's existing documented results; rerunning training may produce slightly different values unless the original random seeds and environment are reproduced exactly.
- Large model/data artifacts are intentionally not committed to the repository.

## License

No open-source license is currently declared for this repository. Until a license is added by the owner, reuse and redistribution should not be assumed to be permitted.
