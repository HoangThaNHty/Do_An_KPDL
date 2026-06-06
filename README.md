# SFAS - Student Feedback Analytics System

> Đồ án cuối kỳ môn **Khai phá Dữ liệu** (NT213.Q11)
> Hệ thống phân tích phản hồi sinh viên bằng Machine Learning + MongoDB + FastAPI.

## 🎯 Giới thiệu

SFAS là hệ thống web thu thập, lưu trữ và phân tích cảm xúc (sentiment) của các phản hồi sinh viên về:
- Giảng viên
- Chương trình đào tạo
- Cơ sở vật chất
- Học thuật
- Khác

Hệ thống hỗ trợ **2 bài toán**:
1. **Phân loại sentiment** (Tiêu cực / Trung tính / Tích cực)
2. **Phân loại topic** (5 chủ đề)

## 🛠️ Công nghệ

| Layer | Tech |
|:---|:---|
| Backend | FastAPI 0.110+ |
| Database | MongoDB 8.0+ (NoSQL - unstructured) |
| ML | scikit-learn (TF-IDF + Naive Bayes + Logistic Regression) |
| NLP | underthesea (tách từ tiếng Việt) |
| Frontend | Jinja2 + Bootstrap 5 + Chart.js |
| Data | pandas, pyarrow |

## 📁 Cấu trúc

```
project/
├── app/                    # FastAPI application
│   ├── main.py            # Routes & API endpoints
│   ├── db.py              # MongoDB CRUD
│   ├── cleaning.py        # 5-step cleaning pipeline
│   ├── templates/         # Jinja2 HTML
│   └── static/            # CSS/JS
├── data/
│   ├── *.parquet          # Datasets (raw/clean/splits)
│   └── reports/           # EDA charts & confusion matrices
├── models/                # Trained models (.pkl)
├── notebooks/             # Jupyter notebooks (01-EDA → 04-Visualization)
├── scripts/               # Pipeline scripts
├── PLAN_SFAS.md           # Master plan
├── requirements.txt
└── .env.example
```

## 🚀 Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/HoangThaNHty/Do_An_KPDL.git
cd Do_An_KPDL
```

### 2. Cài Python packages
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Cài MongoDB
Tải từ [mongodb.com/try/download/community](https://www.mongodb.com/try/download/community) và cài như service.

Hoặc chạy thủ công:
```bash
mongod --dbpath D:\MongoDB\data\db --port 27017
```

### 4. Cấu hình
```bash
cp .env.example .env
# Sửa .env: điền HF_TOKEN của bạn (lấy tại huggingface.co/settings/tokens)
```

### 5. Chạy app
```bash
python -X utf8 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mở [http://localhost:8000](http://localhost:8000).

## 📊 API Endpoints

| Method | Endpoint | Mô tả |
|:---|:---|:---|
| GET | `/` | Form nhập feedback + predict |
| GET | `/dashboard` | Dashboard thống kê |
| GET | `/feedbacks` | Danh sách + CRUD |
| POST | `/api/predict` | Predict sentiment từ text |
| GET | `/api/stats` | Thống kê tổng quan |
| GET | `/api/feedbacks` | JSON list feedbacks |
| POST | `/api/feedbacks` | Tạo mới |
| PUT | `/api/feedbacks/{id}` | Cập nhật |
| DELETE | `/api/feedbacks/{id}` | Xóa |
| POST | `/api/import` | Bulk import CSV |
| GET | `/api/export` | Export CSV |
| GET | `/docs` | Swagger UI |

## 🧪 Tính năng (14/14)

✅ CRUD feedback • ✅ Predict sentiment • ✅ Confidence score
✅ Dashboard Chart.js • ✅ Search/Filter • ✅ Pagination
✅ Bulk import CSV • ✅ Export CSV • ✅ Sentiment trend
✅ Recommendation • ✅ Toast notification • ✅ API docs
✅ Responsive • ✅ Vietnamese UI

## 📈 Kết quả mô hình

| Model | Accuracy | F1 Macro | Note |
|:---|:---|:---|:---|
| Naive Bayes | 82.1% | 61.2% | Baseline |
| **Logistic Regression** | **81.1%** | **66.7%** | **Selected for deploy** |

Cross-domain test (UIT train → NEU test): F1 Macro 29.1% — chứng minh sự cần thiết của mixed-domain training.

## 👥 Nhóm 6

| MSSV | Họ tên | Phụ trách |
|:---|:---|:---|
| ... | ... | ... |

## 📝 License

MIT License - xem [LICENSE](LICENSE).
