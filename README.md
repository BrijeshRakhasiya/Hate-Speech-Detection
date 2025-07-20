# 🛑 Hate Speech Detection using Machine Learning

This project focuses on building a machine learning pipeline to detect **hate speech** in text data using `SGDClassifier` with `CountVectorizer` and `TfidfTransformer`. The model demonstrates strong performance with a **96.91% accuracy**, making it suitable for applications like social media moderation and content filtering.


## 🚀 Overview

- **Objective**: Automatically detect and classify hate speech in text data.
- **Type**: Binary classification (0: Non-Hate, 1: Hate Speech)
- **ML Approach**: Traditional ML pipeline using Scikit-learn
- **Pipeline**:
  - CountVectorizer
  - TfidfTransformer
  - SGDClassifier

---

## 🧠 Tech Stack

- Python 3.x
- Scikit-learn
- Natural Language Processing (NLP)
- Jupyter Notebook

---

## 🔧 Model Pipeline

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', SGDClassifier())
])
```

# 📊 Model Performance

- Accuracy: 96.91%

**📈 Classification Report**
- Metric	Class 0 (Non-Hate)	Class 1 (Hate Speech)
- Precision	0.99	0.95
- Recall	0.95	0.99
- F1-score	0.97	0.97
- Support	7490	7370
- Macro Avg F1-Score: 0.97
- Weighted Avg F1-Score: 0.97

True Positives (TP): 7271 (Hate Speech correctly identified)

True Negatives (TN): 7130 (Non-Hate correctly identified)

False Positives (FP): 360 (Non-Hate misclassified as Hate)

False Negatives (FN): 99 (Hate misclassified as Non-Hate)


## 📄 License

This project is licensed under the MIT License.

---
**Made ❤️ by Brijesh Rakhasiya**

