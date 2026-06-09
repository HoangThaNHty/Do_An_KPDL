# Kịch bản demo SFAS 5-7 phút

## Chuẩn bị trước khi vào lớp

```powershell
venv\Scripts\activate
python -X utf8 scripts\seed_sample_data.py
python -X utf8 scripts\smoke_test.py
python -X utf8 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở sẵn bốn tab:

1. `http://localhost:8000`
2. `http://localhost:8000/feedbacks`
3. `http://localhost:8000/dashboard`
4. `http://localhost:8000/docs`

## 0:00-0:45 - Giới thiệu

Lời nói gợi ý:

> Đề tài của nhóm là xây dựng hệ thống quản lý và khai thác dữ liệu phản hồi
> sinh viên từ văn bản phi cấu trúc. Hệ thống có ba phần chính: MongoDB để lưu
> document phản hồi, hai mô hình học máy để phân loại cảm xúc và chủ đề, và ứng
> dụng FastAPI để quản lý, dự đoán và trực quan hóa.

Chỉ nhanh phần thống kê và pipeline ở trang chủ.

## 0:45-2:15 - Phân tích phản hồi

1. Bấm ví dụ: `giảng viên dạy dễ hiểu và nhiệt tình`.
2. Bấm **Phân tích phản hồi**.
3. Chỉ kết quả sentiment, topic và hai độ tin cậy.
4. Chỉ dòng văn bản sau tiền xử lý và keywords.
5. Bấm **Lưu vào MongoDB**.
6. Thử thêm câu `phòng máy quá cũ và mạng rất yếu`.

Lời nói gợi ý:

> Văn bản đầu vào được chuẩn hóa Unicode, loại nhiễu, xử lý một số teencode và
> tách từ tiếng Việt. Sau đó hai pipeline TF-IDF + Logistic Regression dự đoán
> độc lập sentiment và topic. Kết quả được lưu kèm câu gốc, text đã xử lý, nhãn,
> confidence và metadata.

## 2:15-3:30 - Quản lý dữ liệu MongoDB

1. Mở trang **Phản hồi**.
2. Tìm từ khóa `phòng`.
3. Lọc cảm xúc **Tiêu cực** và chủ đề **Cơ sở vật chất**.
4. Mở modal sửa một phản hồi rồi đóng hoặc lưu.
5. Chỉ nút thêm, xóa, import và export.

Lời nói gợi ý:

> MongoDB lưu mỗi phản hồi dưới dạng một document BSON. Cấu trúc document có
> các object lồng như sentiment, topic và prediction nên phù hợp với dữ liệu
> phi cấu trúc và dễ mở rộng. Collection được tạo index cho nội dung, nhãn,
> split và thời gian; import dùng hash để tránh trùng.

## 3:30-4:50 - Dashboard khai thác dữ liệu

1. Mở **Dashboard**.
2. Chỉ bốn thẻ thống kê tổng quan.
3. Giải thích biểu đồ sentiment, topic và split.
4. Chỉ danh sách phản hồi tiêu cực gần đây.
5. Chỉ phần khuyến nghị.

Lời nói gợi ý:

> Dashboard biến dữ liệu phản hồi thành thông tin hỗ trợ ra quyết định. Ví dụ,
> nếu phản hồi tiêu cực tập trung ở cơ sở vật chất, hệ thống đề xuất ưu tiên
> kiểm tra phòng học, thiết bị và mạng. Phần khuyến nghị hiện được xây dựng bằng
> luật đơn giản trên kết quả khai phá.

## 4:50-5:40 - API và kiến trúc

1. Mở Swagger `/docs`.
2. Mở `POST /api/predict` và cho xem request schema.
3. Chỉ các endpoint CRUD, stats, import, export.

Lời nói gợi ý:

> FastAPI tự sinh tài liệu Swagger và dùng Pydantic để kiểm tra dữ liệu đầu vào.
> Vì vậy hệ thống vừa có giao diện web vừa có API để tích hợp với ứng dụng khác.
> Khi input rỗng, ID sai, thiếu model hoặc MongoDB tắt, API trả lỗi rõ ràng thay
> vì làm toàn bộ ứng dụng bị crash.

## 5:40-6:30 - Kết quả model và kết luận

Lời nói gợi ý:

> Sentiment đạt Accuracy 84,6% và F1 macro 69,44%. Topic đạt Accuracy 80,74% và
> F1 macro 70,78%. Nhóm chọn mô hình cổ điển vì phù hợp phạm vi môn học, dễ giải
> thích và chạy nhanh trên máy cá nhân. Hệ thống đã hoàn thành luồng từ dataset,
> tiền xử lý, MongoDB, mô hình phân loại đến dashboard.

Kết thúc bằng hạn chế:

> Hạn chế hiện tại là dữ liệu còn mất cân bằng, đặc biệt lớp neutral và facility.
> Hướng phát triển là thu thập thêm phản hồi thực tế, cân bằng dữ liệu và thử
> thêm các mô hình tiếng Việt mạnh hơn.

## Phương án dự phòng

- Mất mạng: Bootstrap/Chart.js có thể cần cache; giữ sẵn ảnh chụp màn hình.
- Hugging Face lỗi: không tải lại; dùng `data/sample_feedbacks.csv`.
- Model lỗi: chạy `python -X utf8 scripts\train_models.py`.
- MongoDB chưa chạy: mở Services và Start dịch vụ MongoDB.
- Port 8000 bận: dùng `--port 8001` và đổi URL trình duyệt.
- Demo bị thiếu dữ liệu: chạy lại `python -X utf8 scripts\seed_sample_data.py`.
