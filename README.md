# SFAS - Student Feedback Analytics System

Hệ thống quản lý và khai thác dữ liệu phản hồi sinh viên từ văn bản tiếng Việt
phi cấu trúc. Ứng dụng lưu phản hồi dưới dạng document MongoDB, phân loại cảm
xúc và chủ đề bằng TF-IDF + Logistic Regression, sau đó tổng hợp kết quả trên
dashboard.

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Backend | FastAPI, Pydantic |
| Database | MongoDB, PyMongo |
| Machine Learning | scikit-learn, TF-IDF, Logistic Regression |
| NLP tiếng Việt | underthesea |
| Frontend | Jinja2, Bootstrap 5, Chart.js |
| Dataset | UIT-VSFC (`uitnlp/vietnamese_students_feedback`) |

## Chức năng

- Dự đoán sentiment: `positive`, `neutral`, `negative`.
- Dự đoán topic: `lecturer`, `training_program`, `facility`, `others`.
- Lưu phản hồi và metadata vào MongoDB.
- Tìm kiếm, lọc, phân trang, thêm, sửa, xóa phản hồi.
- Dashboard sentiment, topic, split, phản hồi tiêu cực và khuyến nghị.
- Import CSV/Parquet chống trùng và export CSV.
- REST API có validation và Swagger tại `/docs`.
- Hiển thị cảnh báo thân thiện khi thiếu MongoDB hoặc model.
- Bootstrap và Chart.js được lưu local để giao diện chính vẫn chạy khi mất mạng.

## Cấu trúc chính

```text
app/
  config.py              Đọc cấu hình từ .env
  cleaning.py            Tiền xử lý tiếng Việt và trích từ khóa
  db.py                  MongoDB, schema document, index và CRUD
  ml.py                  Load model và dự đoán
  schemas.py             Pydantic request models
  main.py                Web routes và REST API
  templates/             Giao diện Jinja2
  static/                CSS
data/
  raw/                   Dữ liệu train/validation/test tải về
  processed/feedbacks.csv
  reports/               Metrics và confusion matrix
  sample_feedbacks.csv   30 phản hồi để demo offline
models/
  sentiment_model.pkl
  topic_model.pkl
scripts/
  download_dataset.py
  prepare_data.py
  import_mongodb.py
  train_models.py
  seed_sample_data.py
  smoke_test.py
tests/
REPORT_OUTLINE.md
DEMO_SCRIPT.md
```

## Cài đặt trên Windows

Yêu cầu: Python 3.11 trở lên và MongoDB Community Server.

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Nội dung cấu hình mặc định:

```dotenv
MONGO_URI=mongodb://localhost:27017
MONGO_DB=student_feedback_db
FEEDBACK_COLLECTION=feedbacks
MODEL_DIR=models
DATA_DIR=data
```

`HF_TOKEN` là tùy chọn. Không đưa token thật vào `.env.example` hoặc commit
file `.env`.

## Chuẩn bị dữ liệu và model

### Quy trình đầy đủ

```powershell
python -X utf8 scripts\download_dataset.py
python -X utf8 scripts\prepare_data.py
python -X utf8 scripts\import_mongodb.py
python -X utf8 scripts\train_models.py
```

`download_dataset.py` dùng thư viện `datasets`. Nếu Hugging Face không truy cập
được, script ưu tiên dùng `data/uit_vsfc_raw.parquet` đang có trong repo và in
hướng dẫn tải thủ công khi không có file fallback.

Kết quả được tạo:

- `data/processed/feedbacks.csv`
- `models/sentiment_model.pkl`
- `models/topic_model.pkl`
- `data/reports/metrics.json`
- `data/reports/sentiment_confusion_matrix.png`
- `data/reports/topic_confusion_matrix.png`

### Demo nhanh không cần tải dataset

Bật MongoDB rồi chạy:

```powershell
python -X utf8 scripts\seed_sample_data.py
python -X utf8 scripts\smoke_test.py
```

Script seed nhập 30 phản hồi mẫu tiếng Việt và tự bỏ qua dữ liệu trùng.

## Chạy ứng dụng

```powershell
python -X utf8 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mở:

- Web: [http://localhost:8000](http://localhost:8000)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Dashboard: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

## API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/stats` | Thống kê tổng hợp |
| GET | `/api/feedbacks` | Danh sách, tìm kiếm và lọc |
| POST | `/api/feedbacks` | Tạo phản hồi |
| PUT | `/api/feedbacks/{id}` | Cập nhật phản hồi |
| DELETE | `/api/feedbacks/{id}` | Xóa phản hồi |
| POST | `/api/predict` | Dự đoán sentiment và topic |
| POST | `/api/import` | Import CSV/Parquet |
| GET | `/api/export` | Export CSV |

Ví dụ dự đoán:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/predict `
  -ContentType "application/json" `
  -Body '{"text":"giảng viên dạy dễ hiểu và nhiệt tình"}'
```

## Schema MongoDB

Collection mặc định: `feedbacks`.

```json
{
  "sentence": "giảng viên dạy dễ hiểu",
  "sentiment": {"label": "positive", "code": 2},
  "topic": {"label": "lecturer", "code": 0},
  "processed_text": "giảng_viên dạy dễ hiểu",
  "keywords": ["giảng viên", "dạy"],
  "source": "UIT-VSFC",
  "split": "train",
  "prediction": {
    "sentiment_label": "positive",
    "sentiment_confidence": 0.95,
    "topic_label": "lecturer",
    "topic_confidence": 0.91
  },
  "created_at": "UTC datetime",
  "updated_at": "UTC datetime"
}
```

App tự tạo index cho `sentence`, `sentiment.label`, `topic.label`, `split`,
`created_at` và khóa chống trùng `feedback_key`.

## Kết quả model hiện tại

| Bài toán | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Sentiment | 84.60% | 68.43% | 72.41% | 69.44% |
| Topic | 80.74% | 67.75% | 76.12% | 70.78% |

## Kiểm thử

```powershell
python -X utf8 -m pytest -q
python -X utf8 scripts\smoke_test.py
```

## Checklist tính năng

- [x] Cấu hình bằng `.env`
- [x] MongoDB schema, index và CRUD
- [x] Pipeline tải và tiền xử lý UIT-VSFC
- [x] Model sentiment và topic
- [x] API predict và CRUD
- [x] Import/export dữ liệu
- [x] Trang chủ phân tích và lưu phản hồi
- [x] Trang quản lý có tìm kiếm, lọc, phân trang
- [x] Dashboard Chart.js và khuyến nghị
- [x] Trang giới thiệu đề tài
- [x] Dữ liệu mẫu offline
- [x] Test tự động và smoke test

Xem [DEMO_SCRIPT.md](DEMO_SCRIPT.md) trước khi thuyết trình và
[REPORT_OUTLINE.md](REPORT_OUTLINE.md) khi viết báo cáo.
