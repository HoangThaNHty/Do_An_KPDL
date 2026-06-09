# Dàn ý báo cáo cuối kỳ SFAS

## 1. Giới thiệu đề tài

- Bối cảnh thu thập phản hồi sinh viên trong trường đại học.
- Vấn đề của văn bản tự do: khối lượng lớn, phi cấu trúc, khó tổng hợp thủ công.
- Tên đề tài: Xây dựng hệ thống quản lý và khai thác dữ liệu phản hồi của sinh
  viên từ văn bản phi cấu trúc.
- Phạm vi: phản hồi tiếng Việt về giảng viên, chương trình đào tạo, cơ sở vật
  chất và các nội dung khác.

## 2. Mục tiêu

- Thiết kế cách lưu trữ phản hồi dạng document trong MongoDB.
- Xây dựng pipeline tiền xử lý văn bản tiếng Việt.
- Phân loại cảm xúc và chủ đề của phản hồi.
- Cung cấp ứng dụng quản lý dữ liệu, API và dashboard trực quan.
- Đánh giá mô hình bằng Accuracy, Precision, Recall và F1 macro.

## 3. Cơ sở lý thuyết

### 3.1. Dữ liệu phi cấu trúc

- Khái niệm và đặc điểm của dữ liệu văn bản.
- Khó khăn: không có schema cố định, từ đồng nghĩa, lỗi chính tả, teencode.
- Lý do cần tiền xử lý và khai phá dữ liệu.

### 3.2. MongoDB

- Cơ sở dữ liệu NoSQL hướng document.
- BSON, collection, document, field lồng và index.
- So sánh ngắn với mô hình quan hệ.
- Lý do MongoDB phù hợp với phản hồi sinh viên và metadata dự đoán.

### 3.3. Phân loại văn bản

- Bag of Words và TF-IDF.
- Logistic Regression cho bài toán đa lớp.
- Ma trận nhầm lẫn và các chỉ số đánh giá.

## 4. Dataset UIT-VSFC

- Nguồn: `uitnlp/vietnamese_students_feedback`.
- Tổng số dữ liệu sau làm sạch: 16.169 phản hồi.
- Split: train, validation, test.
- Nhãn sentiment: negative, neutral, positive.
- Nhãn topic: lecturer, training_program, facility, others.
- Trình bày phân bố lớp và nhận xét mất cân bằng, đặc biệt lớp neutral/facility.
- Nêu phương án dữ liệu mẫu offline khi không truy cập Hugging Face.

## 5. Thiết kế hệ thống

- Kiến trúc 3 phần: trình duyệt, FastAPI, MongoDB/model.
- Luồng dự đoán: input -> tiền xử lý -> hai model -> kết quả -> lưu MongoDB.
- Luồng quản trị: tìm kiếm/lọc -> CRUD -> thống kê.
- Công nghệ: FastAPI, Jinja2, Bootstrap, Chart.js, scikit-learn, underthesea.

## 6. Thiết kế cơ sở dữ liệu

- Collection `feedbacks`.
- Giải thích các nhóm field: câu gốc, nhãn, text đã xử lý, từ khóa, nguồn,
  split, prediction, thời gian.
- Trình bày document JSON mẫu từ README.
- Index: `sentence`, `sentiment.label`, `topic.label`, `split`, `created_at`.
- `feedback_key` dùng hash của sentence + split để tránh import trùng.
- Giải thích lợi ích của field lồng `sentiment`, `topic`, `prediction`.

## 7. Pipeline khai phá dữ liệu

1. Tải UIT-VSFC bằng Hugging Face `datasets`.
2. Chuẩn hóa tên cột và ánh xạ nhãn số sang nhãn chữ.
3. Chuẩn hóa Unicode, xóa URL/ký tự nhiễu, xử lý teencode.
4. Tách từ tiếng Việt bằng underthesea.
5. Lưu `data/processed/feedbacks.csv`.
6. Import document vào MongoDB.
7. Huấn luyện và lưu model.
8. Tích hợp model với FastAPI và dashboard.

## 8. Mô hình sentiment và topic

### 8.1. Đặc trưng

- `TfidfVectorizer`, unigram + bigram.
- Tối đa 15.000 đặc trưng, `min_df=2`, `sublinear_tf=True`.

### 8.2. Thuật toán

- Logistic Regression đa lớp.
- `class_weight="balanced"` để giảm ảnh hưởng mất cân bằng.
- `random_state=42` để tái lập kết quả.

### 8.3. Hai bài toán

- Sentiment classification: 3 lớp.
- Topic classification: 4 lớp.
- Mỗi bài toán có một pipeline độc lập.

## 9. Kết quả thực nghiệm

| Bài toán | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Sentiment | 84,60% | 68,43% | 72,41% | 69,44% |
| Topic | 80,74% | 67,75% | 76,12% | 70,78% |

- Chèn hai confusion matrix trong `data/reports/`.
- Phân tích lớp dự đoán tốt/chưa tốt.
- Giải thích Accuracy cao hơn F1 macro do dữ liệu mất cân bằng.
- Nêu rằng model cổ điển phù hợp phạm vi môn học và chạy nhanh trên laptop.

## 10. Giao diện ứng dụng

- Trang chủ: nhập câu, xem hai kết quả và độ tin cậy, lưu MongoDB.
- Trang phản hồi: tìm kiếm, lọc sentiment/topic/split, CRUD, phân trang.
- Dashboard: thống kê, ba biểu đồ, phản hồi tiêu cực, khuyến nghị.
- Trang giới thiệu: mô tả đề tài, dataset, MongoDB và pipeline.
- Swagger `/docs`: chứng minh REST API và validation.
- Chèn ảnh chụp từng màn hình.

## 11. Kết luận và hướng phát triển

### Kết luận

- Hoàn thành quản lý dữ liệu phi cấu trúc bằng MongoDB.
- Hoàn thành hai bài toán phân loại văn bản tiếng Việt.
- Tích hợp model với web, API và dashboard.

### Hạn chế

- Lớp neutral và facility có ít mẫu.
- Độ tin cậy của model chưa được hiệu chỉnh chuyên sâu.
- Khuyến nghị đang dựa trên luật và số lượng phản hồi tiêu cực.
- Chưa có đăng nhập, phân quyền và triển khai production.

### Hướng phát triển

- Thu thập thêm dữ liệu thực tế và cân bằng lớp.
- So sánh thêm SVM hoặc mô hình ngôn ngữ tiếng Việt nhẹ.
- Trích xuất khía cạnh và từ khóa nâng cao.
- Thêm phân quyền, audit log, Docker và triển khai cloud.
- Theo dõi model drift và cho phép giảng viên sửa nhãn để tái huấn luyện.
