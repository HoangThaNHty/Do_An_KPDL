# SFAS Project Plan

## 1. Goal
- Build a Student Feedback Analytics System (SFAS) for the data mining course project.
- Process unstructured Vietnamese student feedback data.
- Predict sentiment and analyze feedback content.
- Store and manage feedback in MongoDB.

## 2. Core Requirements From The PDF

### Scoring Rubric
- Data collection/building and analysis: 1.5
- Data cleaning/data augmentation: 1.5
- Model/algorithm for solving the problem: 2.0
- Application development: 3.0
- Report and presentation: 1.0
- Teamwork and task coordination: 1.0

### Required Report Structure
1. Weekly team working schedule
2. Each member's work
3. Table of contents
4. Introduction
5. Data analysis
6. Design
7. Implementation
8. Conclusion and future work
9. References
10. Appendix

### Submission Rules
- Submit before the presentation session.
- Package Word report, source code, and PowerPoint together.
- Demo the application during the presentation.
- Avoid plagiarism completely.

## 3. Agreed Direction
- Topic: student feedback management and analytics system.
- App name: Student Feedback Analytics System (SFAS).
- Main stack:
  - Python
  - FastAPI
  - MongoDB
  - scikit-learn
  - underthesea
  - HTML/CSS/JS + Chart.js
- Main dataset: UIT-VSFC.
- Supplementary dataset: NEU-ESC.
- Augmentation dataset: Vietnamese students feedback synthetic dataset.
- Number of models: 2.
  - Naive Bayes
  - Logistic Regression

## 4. Project Roadmap

### Phase 1: Data Collection And Exploration
- Download 3 datasets from Hugging Face.
- Normalize dataset formats.
- Filter the useful subset from NEU-ESC.
- Perform EDA:
  - sample counts
  - sentiment/topic distributions
  - missing values
  - text length statistics
- Main objective: satisfy the data collection and analysis requirement.

### Phase 2: Data Cleaning And Augmentation
- Inject artificial noise into about 30% of UIT-VSFC to demonstrate cleaning.
- Build a Vietnamese text cleaning pipeline:
  - remove emoji, URLs, special characters
  - normalize unicode
  - lowercase
  - fix common teencode/typos
  - remove stopwords
  - word segmentation with underthesea
- Merge synthetic data to strengthen training data.
- Compare before/after cleaning for the report.
- Main objective: satisfy the cleaning and augmentation requirement.

### Phase 3: Feature Extraction And Model Training
- Use TF-IDF vectorization.
- Train 2 models:
  - Naive Bayes as baseline
  - Logistic Regression as main deployment model
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Test on NEU-ESC for cross-domain generalization.
- Main objective: satisfy the model/algorithm requirement.

### Phase 4: Visualization And Result Analysis
- Build sentiment distribution charts.
- Build topic distribution charts.
- Generate word clouds if useful.
- Extract insights for the report and slides.

### Phase 5: Web Application Development
- Use MongoDB for unstructured feedback storage.
- Build backend with FastAPI.
- Main features:
  - submit feedback
  - predict sentiment/topic
  - save into MongoDB
  - list feedback
  - filter/search
  - analytics dashboard
- Main objective: maximize the application development score.

### Phase 6: Report And Presentation
- Write the Word report with all 10 required sections.
- Prepare PowerPoint focusing on:
  - problem statement
  - dataset
  - cleaning pipeline
  - 2 models
  - results
  - app demo
- Prepare screenshots, charts, and evaluation tables.

### Phase 7: Final Validation And Submission
- Re-test the whole pipeline.
- Verify app behavior and database integration.
- Check submission files.
- Package Word report, code, and PowerPoint.
- Prepare the final demo.

## 5. Dataset Plan

### UIT-VSFC
- Main dataset for training and evaluation.
- Expected role: core sentiment/topic learning source.

### NEU-ESC
- Supplementary real-world style dataset.
- Expected role: extra analysis and cross-domain testing.

### Synthetic Dataset
- Expected role: training augmentation.

## 6. Model Plan
- Model 1: Multinomial Naive Bayes
- Model 2: Logistic Regression
- Vectorizer: TF-IDF
- Deployment target: best-performing model, expected default is Logistic Regression unless results show otherwise.

## 7. Progress Update Rule
After each major cycle or phase completion, report progress in this format:

```text
HOAN THANH GIAI DOAN X: [Ten giai doan]
Da tao: [files or outputs]
Ket qua: [main results]
Bao cao: [related report sections]
Tiep theo: [next step]
```

## 8. Current Status
- Phase 1 completed successfully.
- Phase 2 completed successfully.
- All 3 datasets downloaded, normalized, cleaned, merged, and split.
- Noise injected into 30% of UIT-VSFC (4,852 rows) to demonstrate cleaning.
- Cleaning pipeline (app/cleaning.py) implements 5 steps: URL/mention/emoji removal, unicode normalization, teencode fixing, stopword removal, word segmentation.
- Final split: Train 23,913 / Val 2,658 / Test 3,166 rows.
- Neutral class balanced from 698 → 4,394 (synthetic data).
- Phase 3 completed successfully.
- TF-IDF vectorizer (max_features=10000, ngram_range=(1,2)) trained and saved.
- Logistic Regression selected for deployment (F1 Macro 66.7% vs NB 61.2%).
- Cross-domain test (UIT train → NEU-ESC test) confirms need for multi-domain training.
- Phase 4 completed successfully.
- Charts generated: sentiment pie, sentiment bar, topic bar, word clouds, sentiment by source.
- Key insight: facility has 95.6% negative, lecturer has 62.1% positive.
- Phase 5 completed successfully.
- MongoDB Community Server 8.3.2 installed on D:\MongoDB.
- FastAPI backend with Jinja2 templates running on localhost:8000.
- All 14 features implemented: CRUD, predict, dashboard, search/filter, import/export, pagination, API docs, etc.
- All page endpoints and API endpoints tested OK (200).
- Next step: Phase 6 — Report and presentation.

## 9. Improvement Log
- 2026-05-24: Reduced the number of models from 4 to 2.
  - Reason: keep scope realistic while still satisfying the model comparison requirement.
- 2026-05-24: Confirmed FastAPI + MongoDB instead of ASP.NET.
  - Reason: Python ML/NLP stack fits the project better and reduces integration friction.

## 10. Issue Log
- 2026-05-24: `hung20gg/NEU-ESC` is gated on Hugging Face. Resolved by user providing `HF_TOKEN` saved to `.env`.
- 2026-05-24: `thnran3/vietnamese-students-feedback-synthetic` (wrong ID). The correct repo is `thnhan3/vietnamese-students-feedback-synthetic`. Resolved.
- 2026-05-24: `datasets` library v4.8.5 drops support for dataset scripts (`.py`). Workaround: download raw data via `huggingface_hub` and `requests` instead of `load_dataset`.
- 2026-05-24: `cp1252` terminal encoding cannot print Vietnamese characters. Workaround: use `python -X utf8` or redirect output to file.

## 11. Decision Log
- 2026-05-24: Use UIT-VSFC as the primary dataset.
  - Why: it directly matches the student feedback domain.
- 2026-05-24: Use NEU-ESC as a supplementary dataset.
  - Why: adds noise diversity and supports cross-domain evaluation.
- 2026-05-24: Use synthetic data for augmentation.
  - Why: helps demonstrate the augmentation requirement.
- 2026-05-24: Keep only Naive Bayes and Logistic Regression.
  - Why: enough for a clean comparison without overexpanding scope.
- 2026-05-24: Add noise injection before cleaning.
  - Why: clearly demonstrates the cleaning pipeline in the report and demo.

## 12. Working Notes
- If the direction changes, append the change to Improvement Log and Decision Log.
- If a blocker appears, append it to Issue Log with impact and proposed workaround.
- Re-read this file whenever the project context needs to be refreshed.

## 13. Phase Completion Log
- 2026-05-24: Phase 1 fully completed.
- 2026-05-24: Phase 2 fully completed.
  - Noise injection: 30% of UIT-VSFC (4,852 rows) injected with emoji, uppercase, teencode, special chars, extra whitespace.
  - Cleaning pipeline: 5-step pipeline in `app/cleaning.py` (URL/mention/emoji removal → unicode normalization → teencode fix → stopword removal → word segmentation).
  - Cleaning results: UIT chars 64→46, NEU chars 124→100, Synth chars 46→36.
  - Merge: 3 datasets combined into one corpus (29,737 rows total).
  - Split: Train 23,913 / Val 2,658 / Test 3,166.
  - Neutral balanced: 698 → 4,394 (synthetic augmentation).
  - Outputs: `scripts/inject_noise.py`, `app/cleaning.py`, `scripts/prepare_data.py`, `notebooks/02_cleaning.ipynb`, `data/train_clean.parquet`, `data/val_clean.parquet`, `data/test_clean.parquet`, `data/uit_vsfc_noisy.parquet`, `data/phase2_summary.json`.
  - Outputs created:
    - `data/uit_vsfc_raw.parquet` — 16,175 rows
    - `data/uit_vsfc_normalized.parquet`
    - `data/neu_esc_raw.parquet` — 32,966 rows
    - `data/neu_esc_normalized.parquet`
    - `data/neu_esc_filtered.parquet` — 10,388 rows (academic + non-toxic)
    - `data/synthetic_raw.parquet` — 3,174 rows (all neutral)
    - `data/synthetic_normalized.parquet`
    - `data/phase1_summary.json`
    - `scripts/download_data.py`
    - `notebooks/01_eda.ipynb`
    - `.env` + `.gitignore`
  - Key results:
    - UIT-VSFC: 8,038 positive / 7,439 negative / 698 neutral
    - UIT-VSFC topics: lecturer 11,607 / training_program 3,040 / others 816 / facility 712
    - NEU-ESC: 10,388 academic rows filtered (7,827 negative / 1,383 positive / 1,178 neutral)
    - Synthetic: 3,174 neutral samples (for class imbalance)
  - Issues encountered and resolved:
    - NEU-ESC was gated → user provided HF token
    - Synthetic repo had wrong ID (`thnran3` → `thnhan3`)
    - `datasets.load_dataset` broken for script-based repos → switched to `huggingface_hub` + `requests`
- 2026-05-24: Phase 3 fully completed.
  - TF-IDF vectorizer: max_features=10000, ngram_range=(1,2), vocab size=10000
  - Model comparison (test set):
    | Model | Accuracy | F1 Macro | F1 Weighted |
    |:---|:---:|:---:|:---:|
    | Naive Bayes | 82.1% | 61.2% | 81.3% |
    | Logistic Regression | 81.1% | **66.7%** | 82.8% |
  - LR selected for deployment (higher F1 Macro, better neutral recall)
  - Cross-domain: UIT train → NEU-ESC test = F1 Macro 29.1% (confirms need for mixed training)
  - Outputs: `scripts/train_models.py`, `notebooks/03_model_training.ipynb`, `models/tfidf_vectorizer.pkl`, `models/nb_model.pkl`, `models/lr_model.pkl`, `data/reports/nb_confusion_matrix.png`, `data/reports/lr_confusion_matrix.png`, `data/phase3_summary.json`
- 2026-05-24: Phase 4 fully completed.
  - Charts generated: `data/reports/sentiment_pie.png`, `data/reports/sentiment_bar.png`, `data/reports/topic_bar.png`, `data/reports/word_clouds.png`, `data/reports/sentiment_by_source.png`
  - Key insights:
    - Sentiment overall: negative 51.3%, positive 31.7%, neutral 17.0%
    - Topic "facility" has 95.6% negative sentiment
    - Topic "lecturer" has 62.1% positive sentiment
    - Top negative words: học, không, viên, sinh, môn, thầy, bài, giảng, trường
    - Top positive words: giảng, dạy, tình, nhiệt, sinh, dễ
  - Outputs: `scripts/visualize.py`, `notebooks/04_visualization.ipynb`, `data/phase4_insights.json`, charts in `data/reports/`
- 2026-05-24: Phase 5 fully completed.
  - MongoDB Community Server 8.3.2 installed on D:\MongoDB (877MB download via BITS).
  - mongod.exe running with data directory D:\MongoDB\data\db.
  - FastAPI backend with Jinja2 templates.
  - All 14 features implemented:
    1. CRUD Feedback (Add/Edit/Delete/Read)
    2. Predict Sentiment (Input → Clean → TF-IDF → LR → Result)
    3. MongoDB Storage (All feedback saved to DB)
    4. Dashboard (Pie chart sentiment, bar chart topic, trend line)
    5. Search/Filter (Search by text, filter by sentiment)
    6. Confidence Score (Show % for each class)
    7. Bulk Import (Upload CSV/Parquet)
    8. Export CSV (Download all feedbacks)
    9. Sentiment Trend (Line chart over time)
    10. Recommendation (Insights from negative feedback)
    11. Pagination (20 items per page)
    12. Toast Notification (Alert for CRUD operations)
    13. API Docs (FastAPI auto-doc at /docs)
    14. Responsive Design (Bootstrap 5, mobile-friendly)
  - All endpoints tested OK: GET /, GET /dashboard, GET /feedbacks, GET /api/stats, GET /api/predict, GET /docs
  - Vietnamese interface throughout.
  - Outputs: `app/db.py`, `app/main.py`, `app/templates/base.html`, `app/templates/index.html`, `app/templates/dashboard.html`, `app/templates/list.html`, `app/static/css/style.css`
