# Xây Dựng Pipeline Big Data Phân Tán

| Môn học | Thực hành Big Data |
|---|---|
| Thời gian | 2 Tuần |
| Nền tảng | Local Hadoop/Spark Cluster (Docker / VMs / Linux) |

---

## 📋 Tổng Quan

**Kiến thức cần có:** Thuần thục xây dựng một cụm Big Data (HDFS + Spark).

**Nhiệm vụ:** Xây dựng một Pipeline Phân tán End-to-End:

> Nạp dữ liệu thô (Ingest) → Xử lý & Làm sạch (ETL bằng Spark) → Ứng dụng AI/ML (PyTorch / SparkML) → Trích xuất thông tin giá trị (Business Insight)

---

## ⚙️ Yêu Cầu Kỹ Thuật

Đồ án phải tuân thủ **nghiêm ngặt** các quy tắc sau:

| # | Quy tắc | Mô tả |
|---|---|---|
| 1 | **Bắt buộc dùng HDFS** | Dữ liệu đầu vào thô (Raw Data) phải được upload lên HDFS trước khi xử lý. Không được đọc trực tiếp từ ổ cứng máy thật (Local OS). |
| 2 | **Cấm dùng vòng lặp Local** | Tuyệt đối không sử dụng vòng lặp `for` của Python để duyệt qua từng file (ví dụ: `os.listdir`). Phải sử dụng Spark RDD hoặc DataFrame. |
| 3 | **AI Phân tán (Distributed Inference)** | Model AI/ML phải được chạy bên trong các Spark Workers (sử dụng UDFs hoặc Spark MLlib). |
| 4 | **Lưu trữ** | Kết quả trung gian và cuối cùng phải được ghi ngược lại HDFS dưới định dạng **Parquet**. |
| 5 | **Bằng chứng (History & Logs)** | Bắt buộc phải cấu hình và bật **Spark History Server**. Cấu hình để Spark ghi log sự kiện vào HDFS (ví dụ: `/spark-logs`). Báo cáo phải có ảnh chụp màn hình từ WebUI (port `18080`) chứng minh các Stage/Task đã chạy song song. |

---

## 🎯 Đề Tài

Sinh viên chọn **MỘT** trong hai đề tài dưới đây. Cả hai đề tài đều được thiết kế để gây khó khăn cho một máy tính cá nhân nếu chỉ dùng code Python thông thường mà không tận dụng sức mạnh phân tán.

---

### Option 1: The Deepfake Hunter 🕵️ (Xử lý Ảnh)

**Kịch bản:** Xây dựng một pipeline có khả năng mở rộng để quét hàng nghìn ảnh mỗi ngày nhằm phát hiện ảnh do AI tạo ra.

#### 1. Dữ liệu (Dataset)

- **Nguồn:** CIFAKE (Real vs. AI-Generated)
- **Khối lượng:** ~100.000 file ảnh nhỏ `.jpg` (32×32)

#### 2. Yêu Cầu Pipeline

| Bước | Tên | Mô tả |
|---|---|---|
| Step 1 | **Ingestion** | Nạp dữ liệu lên HDFS. |
| Step 2 | **Feature Extraction** | Sinh viên **KHÔNG** được dùng model "Deepfake Detector" có sẵn. Trích xuất đặc trưng từ ảnh bằng model train trên ImageNet như: `ResNet50`, `MobileNetV2`. |
| Step 3 | **Distributed Classification** | Chuyển đổi mảng đặc trưng thành kiểu dữ liệu `Vector` của Spark ML. Sử dụng Spark MLlib để huấn luyện bộ phân loại cổ điển (`LogisticRegression` hoặc `RandomForest`) để dự đoán **Real** vs **Fake**. |
| Step 4 | **Kiểm tra kết quả** | Đánh giá hiệu năng mô hình. |

#### 3. Kết Quả (Business Insight)

- Báo cáo các chỉ số **Accuracy**, **Precision**, và **Recall** của mô hình lai (Hybrid Model).
- Trả lời câu hỏi: *Liệu model được chọn có trích xuất đủ thông tin để phát hiện Deepfake không?*

---

### Option 2: The Open Source Influencer 🌐 (NLP & Graph)

**Kịch bản:** Phân tích "Mạng xã hội" của các lập trình viên trên GitHub để tìm ra các xu hướng công nghệ và những người có tầm ảnh hưởng thực sự.

#### 1. Dữ liệu (Dataset)

- **Nguồn:** [GitHub Archive](https://gharchive.org)
- **Phạm vi:** Chỉ giới hạn dữ liệu trong **Quý 3, 2025**
- **Khối lượng:** Lớn

#### 2. Yêu Cầu Pipeline

| Bước | Tên | Mô tả |
|---|---|---|
| Step 1 | **Ingestion & Cleaning** | Đọc dữ liệu JSON. Xử lý các dòng lỗi, nhiễu, trải phẳng dữ liệu. Xác định và loại bỏ các user có trên 100 commit/giờ. |
| Step 2 | **Graph Analysis** | Xây dựng đồ thị (Graph) với **Nodes** = Users/Repos và **Edges** = Tương tác (Star, Fork, Commit). |
| Step 3 | **NLP** | Phân tích ngôn ngữ tự nhiên trên dữ liệu văn bản từ GitHub. |

#### 3. Kết Quả (Business Insight)

- **Bảng Influencer:** Top 20 Repos dựa trên điểm PageRank.
- **Trend Cloud:** Danh sách Top 10 ngôn ngữ phổ biến nhất, Top 20 từ khóa đặc trưng cho từng ngôn ngữ.

---

## 📦 Sản Phẩm Nộp

| Hạng mục | Yêu cầu |
|---|---|
| **Source Code** | File nén chứa Jupyter Notebooks (`.ipynb`) hoặc Python scripts. Code phải có comment giải thích. |
| **Báo cáo (PDF)** | Bao gồm: Sơ đồ kiến trúc (Architecture Diagram), Kết quả phân tích (Biểu đồ / Bảng số liệu), Thách thức / lỗi đã gặp. |
