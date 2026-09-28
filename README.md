# AG News Multiclass Classification & Interpretability Benchmark

An end-to-end Natural Language Processing (NLP) pipeline benchmarking Classical Machine Learning versus Fine-tuned Transformers on the **AG News** corpus (120,000 training and 7,600 test articles).

## Dataset Specifications
- **Source**: AG News Classification Dataset
- **Samples**: 120,000 Train / 7,600 Test
- **Target Classes**: World, Sports, Business, Sci/Tech (Balanced 25% each)
- **Length**: Median text length of 37 tokens.

## Benchmark & Results

| Architecture | Paradigm | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline (Uniform) | Rule-based | 25.13% | 0.2500 | 0.2513 | 0.2512 |
| Linear SVM (Tuned) | Classical ML | 90.71% | 0.9070 | 0.9071 | 0.9067 |
| DistilBERT (2 Epochs) | Deep Learning | 91.03% | 0.9107 | 0.9103 | 0.9102 |

## How to Run

```bash
git clone [https://github.com/](https://github.com/)<payamhabibi>/nlp-news-classification.git
cd nlp-news-classification
pip install -r requirements.txt
```

```python
from src.predict import NewsClassificationPipeline

pipeline = NewsClassificationPipeline(
    vectorizer_path='models/tfidf_vectorizer.joblib',
    model_path='models/best_linear_svc.joblib'
)
sample = ['NASA successfully launches new deep-space telescope.']
print(pipeline.predict(sample))
```
