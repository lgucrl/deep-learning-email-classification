# Machine / Deep Learning classification and NLP analysis of emails

This project builds an end-to-end pipeline to **classify emails as SPAM vs. HAM (non-spam)** and then perform detailed **NLP analyses** on the resulting groups. In addition to supervised classification, it explores the **main topics** of SPAM emails, the **differences/similarities** of SPAM topics, and the **organizations mentioned** in non-spam messages. The full project is contained in the [`email_classification_ml_dl.ipynb`](https://github.com/lgucrl/deep-learning-email-classification/blob/main/email_classification_ml_dl.ipynb) notebook.

---

## Dataset

The dataset is a labeled text corpus containing **>5,000 emails** and the following columns:

- `text`: raw email text (including subject and body)
- `label`: string class (`ham` / `spam`)
- `label_num`: numeric class (`0` = ham, `1` = spam)

Class distribution is **imbalanced (~71% ham and ~29% spam)**, which requires specific choices in modeling and evaluation.

---

## Project workflow

1. **Exploratory Data Analysis**  
   The workflow begins with a structured inspection of dataset shape, schema, and label distribution to quantify imbalance. Text-oriented EDA is then performed, where email length is measured (word counts) both overall and per class, and summary statistics with histograms/boxplots are used to highlight outliers and typical message sizes. This step informs decisions such as token filtering threshold and sequence length caps for neural network models.

2. **Text cleaning, normalization and split**  
   Raw emails are transformed into a “clean” representation suitable for ML. Tokenization, lowercasing, punctuation/digit removal, and de-accenting are applied with **Gensim** preprocessing, setting a maximum token length to 99th percentile to normalize them and reduce noise. **NLTK stopwords** are then removed and a cleaned text field is reconstructed to support different pipelines consistently. To preserve real-world class ratios, the dataset is split into train/test partitions using **stratified sampling**. This ensures that evaluation reflects performance on an imbalanced distribution.

3. **Vectorization to TF-IDF and integer sequences**  
   Two parallel representations are built:
   - **TF-IDF vectors** (sparse, high-dimensional) for classical ML models.
   - **Integer token sequences** using Keras `TextVectorization`, with vocabulary size tuned to cover 95% of token occurrences and sequence length capped at the 95th percentile of email lengths (applying padding/truncation to obtain uniform tensors), for deep learning approaches.

4. **Handling imbalance with SMOTE**  
   Since ham emails dominates, the TF-IDF training set is balanced using **SMOTE** to synthetically oversample the minority class. Importantly, only the training data is oversampled, while test data remains untouched to avoid inflated metrics and to keep evaluation realistic.

5. **Model training (Multinomial Naive Bayes and RNN)**  
   Two classifiers are trained and compared:
   - **Multinomial Naive Bayes** on balanced TF-IDF features.
   - A **Bidirectional GRU-based RNN** (with `Embedding`, `SpatialDropout`, `BiGRU` and `Dense sigmoid`) on token sequences, with early stopping on validation accuracy to control overfitting.

6. **Model evaluation**  
   Models are evaluated on the held-out test set using accuracy, precision, recall, F1, ROC-AUC, and PR-AUC, complemented by confusion matrices and ROC/PR curves. This gives a threshold-based view of false positives vs. false negatives, crucial where misclassification costs for spam filtering must be considered.

7. **NLP: topics, similarity, and entities**  
   After classification, **LDA topic modeling** is applied to SPAM emails to extract dominant themes. To quantify the heterogeneity of SPAM content, topic-level embeddings are generated using **Doc2Vec**, and **cosine similarity** is computed between topics. Finally, **spaCy NER** is run on non-spam emails to extract and rank all mentioned **organization entities (ORG)**.

---

## Tech stack

- **Python**
- **scikit-learn** (train/test split, TF-IDF, metrics, Naive Bayes)
- **TensorFlow / Keras** (TextVectorization, BiGRU model)
- **NLTK** (tokenization, stopwords)
- **Gensim** (preprocessing, LDA, Doc2Vec)
- **imbalanced-learn** (SMOTE)
- **SciPy** (cosine distance/similarity)
- **spaCy** (Named Entity Recognition)
