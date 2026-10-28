# Chương 1: Tổng quan và các kiểu học

Chương này vẽ bản đồ toàn cảnh. Đọc kỹ trước khi vào chi tiết từng thuật toán, vì mọi chương sau đều nằm đâu đó trên bản đồ này.

## Mục lục

1. [AI, Machine Learning, Deep Learning](#1-ai-machine-learning-deep-learning)
2. [Định nghĩa Machine Learning](#2-định-nghĩa-machine-learning)
3. [Khi nào nên và không nên dùng ML](#3-khi-nào-nên-và-không-nên-dùng-ml)
4. [Phân loại theo tín hiệu học: các kiểu học](#4-phân-loại-theo-tín-hiệu-học-các-kiểu-học)
5. [Học có giám sát](#5-học-có-giám-sát-supervised-learning)
6. [Học không giám sát](#6-học-không-giám-sát-unsupervised-learning)
7. [Học bán giám sát](#7-học-bán-giám-sát-semi-supervised-learning)
8. [Học tự giám sát](#8-học-tự-giám-sát-self-supervised-learning)
9. [Học tăng cường](#9-học-tăng-cường-reinforcement-learning)
10. [Các kiểu học khác](#10-các-kiểu-học-khác)
11. [Phân loại theo các trục khác](#11-phân-loại-theo-các-trục-khác)
12. [Các thành phần của một hệ thống ML](#12-các-thành-phần-của-một-hệ-thống-ml)
13. [Các loại dữ liệu](#13-các-loại-dữ-liệu)
14. [Quy trình một dự án ML](#14-quy-trình-một-dự-án-ml)
15. [Thuật ngữ cần thuộc](#15-thuật-ngữ-cần-thuộc)
16. [Chọn kiểu bài toán từ câu hỏi thực tế](#16-chọn-kiểu-bài-toán-từ-câu-hỏi-thực-tế)

---

## 1. AI, Machine Learning, Deep Learning

```
Trí tuệ nhân tạo (AI)
└── Machine Learning (ML): học quy luật từ dữ liệu
    └── Deep Learning (DL): ML dùng mạng neural nhiều tầng
        └── Mô hình ngôn ngữ lớn (LLM), mô hình sinh ảnh...
```

| Khái niệm | Là gì | Ví dụ |
|---|---|---|
| **AI** | Mọi kỹ thuật làm máy hành xử "thông minh", gồm cả luật viết tay và tìm kiếm | Hệ chuyên gia, thuật toán cờ vua cổ điển |
| **ML** | Máy tự học quy luật từ dữ liệu thay vì được lập trình từng luật | Lọc spam, dự đoán giá, gợi ý sản phẩm |
| **DL** | ML dựa trên mạng neural sâu, tự học đặc trưng từ dữ liệu thô | Nhận diện ảnh, dịch máy, ChatGPT |
| **Data Science** | Rút thông tin từ dữ liệu: thống kê, trực quan hóa, ML, truyền đạt kết quả | Báo cáo phân tích, thí nghiệm A/B |

Mối quan hệ: ML là tập con của AI. DL là tập con của ML. Data Science giao với cả ML lẫn thống kê, không nằm gọn trong chúng.

**Khác biệt về cách tiếp cận:**

```
Lập trình truyền thống:   Dữ liệu + Luật   →  Chương trình  →  Đáp án
Machine Learning:         Dữ liệu + Đáp án →  Học           →  Luật (mô hình)
```

---

## 2. Định nghĩa Machine Learning

Định nghĩa kinh điển của Tom Mitchell (1997):

> Một chương trình máy tính được cho là *học* từ kinh nghiệm **E** đối với tập tác vụ **T** và thước đo hiệu năng **P**, nếu hiệu năng của nó trên **T**, đo bằng **P**, tăng lên nhờ kinh nghiệm **E**.

Ví dụ lọc spam:
- **T** (tác vụ): phân loại thư là spam hay không.
- **E** (kinh nghiệm): tập thư đã được người dùng gán nhãn.
- **P** (thước đo): tỉ lệ thư phân loại đúng.

Cách phát biểu toán học thường dùng: có dữ liệu $\{(x_i, y_i)\}_{i=1}^n$ lấy từ một phân phối chưa biết. Tìm hàm $f$ trong một họ hàm cho trước sao cho $f(x) \approx y$ **trên dữ liệu mới** (tổng quát hóa), không chỉ trên dữ liệu đã thấy.

---

## 3. Khi nào nên và không nên dùng ML

**Nên dùng khi:**
- Không thể viết luật bằng tay, hoặc luật quá nhiều, quá phức tạp (nhận diện giọng nói, ảnh).
- Luật thay đổi theo thời gian, cần hệ thống tự thích nghi (gian lận, thị hiếu).
- Có đủ dữ liệu và dữ liệu đại diện cho tình huống thật.
- Chấp nhận được sai số (dự đoán không bao giờ đúng 100%).

**Không nên dùng khi:**
- Có luật rõ ràng và đơn giản. Một câu `if` rẻ và đáng tin hơn.
- Không có dữ liệu, hoặc dữ liệu quá ít hay quá nhiễu.
- Cần đảm bảo tuyệt đối và giải thích được từng quyết định theo luật cứng (một số bài toán pháp lý, an toàn).
- Chi phí xây dựng và vận hành vượt giá trị mang lại.

**Quy tắc thực tế:** thử giải pháp không dùng ML trước (luật đơn giản, trung bình, truy vấn SQL). Đó vừa là baseline, vừa cho biết ML có thực sự cần thiết.

---

## 4. Phân loại theo tín hiệu học: các kiểu học

Cách phân loại phổ biến nhất dựa vào **loại thông tin mô hình nhận được để học**.

| Kiểu học | Tín hiệu để học | Câu hỏi nó trả lời | Ví dụ |
|---|---|---|---|
| **Có giám sát** (supervised) | Cặp (đầu vào, nhãn đúng) | "Với đầu vào này, đầu ra là gì?" | Dự đoán giá nhà, phân loại spam |
| **Không giám sát** (unsupervised) | Chỉ có đầu vào, không nhãn | "Dữ liệu có cấu trúc gì?" | Phân nhóm khách hàng, phát hiện bất thường |
| **Bán giám sát** (semi-supervised) | Ít dữ liệu có nhãn, nhiều dữ liệu không nhãn | "Tận dụng dữ liệu không nhãn để học tốt hơn?" | Phân loại ảnh y tế khi chỉ vài ca được bác sĩ gán nhãn |
| **Tự giám sát** (self-supervised) | Nhãn được tạo tự động từ chính dữ liệu | "Học biểu diễn hữu ích từ lượng dữ liệu khổng lồ" | Dự đoán từ tiếp theo (GPT), che từ rồi đoán (BERT) |
| **Tăng cường** (reinforcement) | Phần thưởng sau khi hành động | "Hành động nào dẫn đến phần thưởng tích lũy lớn nhất?" | Chơi cờ, điều khiển robot, tối ưu quảng cáo |

Sơ đồ phân biệt nhanh:

```
Bạn có nhãn cho dữ liệu không?
├── Có, cho (gần) tất cả       → Học có giám sát
├── Có, cho một phần nhỏ       → Học bán giám sát
├── Không, nhưng tạo được nhãn
│   từ chính dữ liệu           → Học tự giám sát
├── Không có nhãn              → Học không giám sát
└── Không có nhãn, nhưng có
    phần thưởng sau hành động  → Học tăng cường
```

---

## 5. Học có giám sát (Supervised Learning)

Mô hình học từ dữ liệu đã gán nhãn, như học sinh làm bài có đáp án. Mục tiêu: học ánh xạ $f: X \to Y$.

### 5.1. Các dạng bài toán

| Dạng | Đầu ra $y$ | Ví dụ | Thuật toán thường dùng |
|---|---|---|---|
| **Hồi quy** (regression) | Số liên tục | Giá nhà, nhiệt độ, doanh thu | Linear/Ridge/Lasso, Random Forest, GBM |
| **Phân loại nhị phân** | 1 trong 2 lớp | Spam/không, gian lận/không | Logistic, SVM, cây, GBM |
| **Phân loại đa lớp** | 1 trong $K>2$ lớp | Chữ số 0–9, loại hoa | Softmax, Random Forest, SVM, KNN |
| **Phân loại đa nhãn** (multi-label) | Tập nhãn, nhiều nhãn cùng lúc | Gắn thẻ bài viết (chính trị + kinh tế) | Binary relevance, mạng neural |
| **Hồi quy thứ tự** (ordinal) | Lớp có thứ tự | Đánh giá 1–5 sao | Ordinal regression, GBM |
| **Xếp hạng** (ranking) | Thứ tự các mục | Kết quả tìm kiếm, gợi ý | LambdaMART, learning-to-rank |
| **Dự báo chuỗi thời gian** | Giá trị tương lai | Nhu cầu hàng tuần | ARIMA, GBM với đặc trưng trễ, mạng neural |
| **Dự đoán có cấu trúc** | Chuỗi, cây, đồ thị | Dịch máy, gán nhãn từ loại | Mạng neural, CRF |

**Phân biệt hồi quy và phân loại:** nhìn vào nhãn. Nếu nhãn là số mà "gần hơn thì tốt hơn" (dự đoán 305 thay vì 300 tốt hơn dự đoán 400) thì hồi quy. Nếu nhãn là danh mục thì phân loại. Một số bài toán chuyển đổi qua lại được (chia giá thành các khoảng), nhưng thường mất thông tin.

### 5.2. Cách học

1. Chọn họ mô hình và hàm mất mát.
2. Đưa dữ liệu có nhãn, mô hình dự đoán, so với nhãn thật, tính loss.
3. Điều chỉnh tham số để giảm loss (tối ưu hóa).
4. Đánh giá trên dữ liệu chưa từng thấy.

### 5.3. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Mục tiêu rõ ràng, đo được chất lượng trực tiếp | Cần dữ liệu có nhãn, thường tốn tiền và thời gian |
| Hiệu quả cao khi đủ nhãn tốt | Nhãn sai hoặc thiên lệch thì mô hình sai theo |
| Phần lớn ứng dụng thương mại thuộc nhóm này | Chỉ học được những gì nhãn thể hiện |

### 5.4. Chi phí gán nhãn

Nhãn là nút thắt lớn nhất. Các nguồn nhãn: gán tay (đắt, chất lượng cao), crowdsourcing (rẻ, nhiễu hơn), nhãn tự nhiên từ hệ thống (click, mua hàng, hủy dịch vụ), nhãn yếu từ luật (xem mục 10). Luôn kiểm tra chất lượng nhãn: nhờ hai người gán độc lập và đo mức đồng thuận (Cohen's kappa).

---

## 6. Học không giám sát (Unsupervised Learning)

Chỉ có đầu vào $x$, không có nhãn $y$. Mô hình tự tìm cấu trúc ẩn trong dữ liệu. Không có "đáp án đúng" để so sánh, nên việc đánh giá khó hơn và thường cần đánh giá bằng mục đích sử dụng.

### 6.1. Các dạng bài toán

| Dạng | Mục tiêu | Ví dụ | Thuật toán |
|---|---|---|---|
| **Gom cụm** (clustering) | Chia dữ liệu thành các nhóm tương tự | Phân khúc khách hàng, gom bài báo cùng chủ đề | K-Means, DBSCAN, phân cấp, GMM |
| **Giảm chiều** (dimensionality reduction) | Biểu diễn dữ liệu bằng ít chiều hơn, giữ thông tin chính | Trực quan hóa, nén, khử nhiễu | PCA, t-SNE, UMAP, autoencoder |
| **Phát hiện bất thường** (anomaly detection) | Tìm mẫu khác thường | Gian lận, hỏng hóc thiết bị, xâm nhập mạng | Isolation Forest, LOF, One-class SVM |
| **Ước lượng mật độ** (density estimation) | Mô hình hóa phân phối của dữ liệu | Phát hiện điểm hiếm, sinh dữ liệu | KDE, GMM |
| **Luật kết hợp** (association rules) | Tìm các mục hay đi cùng nhau | "Mua bia thường mua kèm đồ nhắm" | Apriori, FP-Growth |
| **Học biểu diễn** (representation learning) | Học vector đặc trưng hữu ích | Embedding từ, embedding ảnh | Autoencoder, Word2Vec |
| **Mô hình sinh** (generative) | Học để sinh dữ liệu giống thật | Sinh ảnh, sinh văn bản | VAE, GAN, diffusion |

### 6.2. Dùng để làm gì trong thực tế

- **Khám phá:** hiểu dữ liệu trước khi dựng mô hình có giám sát.
- **Tiền xử lý:** giảm chiều, khử nhiễu, tạo đặc trưng (số cụm, khoảng cách tới tâm cụm).
- **Sản phẩm độc lập:** phân khúc khách hàng, phát hiện bất thường, gợi ý.
- **Khi không có nhãn** và chưa thể có nhãn.

### 6.3. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Không cần nhãn | Không có thước đo đúng sai tuyệt đối |
| Tìm được cấu trúc bạn chưa nghĩ tới | Kết quả phụ thuộc mạnh vào lựa chọn thuật toán và siêu tham số |
| Dữ liệu không nhãn thì rất nhiều | Khó diễn giải: "cụm 3" nghĩa là gì? Cần con người đặt tên |

Chi tiết thuật toán ở [Chương 7](07-hoc-khong-giam-sat.md).

### 6.4. So sánh có giám sát và không giám sát

| | Có giám sát | Không giám sát |
|---|---|---|
| Dữ liệu | $(x, y)$ | Chỉ $x$ |
| Mục tiêu | Dự đoán $y$ | Tìm cấu trúc trong $x$ |
| Đánh giá | Metric so với nhãn thật (accuracy, RMSE) | Metric nội tại (silhouette), đánh giá theo mục đích |
| Độ khó đánh giá | Dễ | Khó |
| Ví dụ | Dự đoán khách có rời bỏ không | Chia khách hàng thành nhóm hành vi |

---

## 7. Học bán giám sát (Semi-supervised Learning)

Có **ít nhãn, nhiều dữ liệu không nhãn**. Ý tưởng: dữ liệu không nhãn cho biết hình dạng của phân phối, giúp mô hình đặt ranh giới quyết định tốt hơn so với chỉ dùng vài mẫu có nhãn.

Hai giả định thường đi kèm: **giả định cụm** (các điểm cùng cụm có khả năng cùng nhãn) và **giả định đa tạp** (dữ liệu nằm trên một cấu trúc ít chiều).

| Kỹ thuật | Cách làm |
|---|---|
| **Self-training / pseudo-labeling** | Huấn luyện trên dữ liệu có nhãn, dự đoán nhãn cho dữ liệu không nhãn, giữ lại dự đoán có độ tin cậy cao làm nhãn giả, huấn luyện lại, lặp |
| **Label propagation** | Dựng đồ thị các điểm gần nhau, lan truyền nhãn từ điểm có nhãn sang điểm lân cận |
| **Consistency regularization** | Ép mô hình cho dự đoán giống nhau khi đầu vào bị nhiễu nhẹ |
| **Co-training** | Hai mô hình nhìn hai "góc" đặc trưng khác nhau, dạy nhau |

Rủi ro: nhãn giả sai sẽ củng cố chính lỗi đó (confirmation bias). Dùng ngưỡng tin cậy cao và kiểm chứng trên tập validation có nhãn thật.

Code và ví dụ ở [Chương 10](10-cac-kieu-hoc-khac.md).

---

## 8. Học tự giám sát (Self-supervised Learning)

Nhãn được **tạo tự động từ chính dữ liệu**, nên không cần người gán. Mô hình giải một "bài toán giả" (pretext task) để buộc nó học biểu diễn hữu ích, sau đó biểu diễn này dùng cho bài toán thật.

| Bài toán giả | Cách tạo nhãn | Mô hình điển hình |
|---|---|---|
| **Dự đoán từ tiếp theo** | Nhãn là từ kế tiếp trong câu | GPT và hầu hết LLM |
| **Che từ rồi đoán** (masked LM) | Che một số từ, nhãn là từ bị che | BERT |
| **Học đối sánh** (contrastive) | Hai phiên bản biến đổi của cùng một ảnh là cặp "giống", ảnh khác là "khác" | SimCLR, CLIP |
| **Khôi phục** (autoencoding) | Nhãn là chính đầu vào đã bị làm nhiễu | Denoising autoencoder, MAE |

Đây là nền tảng của các mô hình lớn hiện nay: **tiền huấn luyện tự giám sát** trên lượng dữ liệu khổng lồ, sau đó **tinh chỉnh có giám sát** hoặc dùng trực tiếp. Bạn sẽ học sâu ở GĐ3 (NLP/LLM) và GĐ6 (fine-tuning). Ở giai đoạn ML cổ điển chỉ cần nắm khái niệm.

---

## 9. Học tăng cường (Reinforcement Learning)

Một **tác tử** (agent) tương tác với **môi trường** (environment) qua nhiều bước. Ở mỗi bước nó quan sát trạng thái, chọn hành động, nhận phần thưởng và chuyển sang trạng thái mới. Mục tiêu: học **chính sách** (policy) tối đa hóa **tổng phần thưởng tích lũy**.

```
         hành động a_t
   ┌─────────────────────┐
   │                     ▼
Tác tử (agent)       Môi trường
   ▲                     │
   └─────────────────────┘
   trạng thái s_{t+1}, phần thưởng r_{t+1}
```

| Khái niệm | Ý nghĩa |
|---|---|
| Trạng thái $s$ | Mô tả tình huống hiện tại |
| Hành động $a$ | Điều tác tử có thể làm |
| Phần thưởng $r$ | Tín hiệu đánh giá tức thời, có thể trễ và thưa |
| Chính sách $\pi(a\mid s)$ | Quy tắc chọn hành động |
| Hàm giá trị $V(s)$, $Q(s,a)$ | Tổng phần thưởng kỳ vọng trong tương lai |
| Hệ số chiết khấu $\gamma$ | Mức coi trọng phần thưởng xa so với gần |

**Điểm khác biệt so với học có giám sát:**
- Không có "đáp án đúng" cho từng bước, chỉ có phần thưởng, thường đến muộn (thắng/thua cuối ván).
- Dữ liệu do chính tác tử tạo ra qua hành động, nên dữ liệu phụ thuộc vào chính sách hiện tại.
- Phải cân bằng **khám phá** (thử cái mới) và **khai thác** (dùng cái đã biết là tốt).

**Ứng dụng:** trò chơi (cờ vây, game), robot, điều khiển, tối ưu quảng cáo/đề xuất, huấn luyện LLM theo phản hồi người dùng (RLHF).

Phần cơ bản (bandit, Q-learning) có ở [Chương 10](10-cac-kieu-hoc-khac.md).

---

## 10. Các kiểu học khác

| Kiểu | Ý tưởng | Ví dụ |
|---|---|---|
| **Học chuyển giao** (transfer learning) | Dùng mô hình đã học bài toán lớn làm điểm xuất phát cho bài toán mới, ít dữ liệu hơn | Dùng ResNet huấn luyện trên ImageNet để phân loại ảnh y tế |
| **Học đa tác vụ** (multi-task) | Học nhiều tác vụ liên quan cùng lúc, chia sẻ biểu diễn | Một mô hình vừa phát hiện vật thể vừa phân đoạn |
| **Học chủ động** (active learning) | Mô hình chọn mẫu nào nên được gán nhãn tiếp theo (mẫu nó ít chắc chắn nhất) | Tiết kiệm chi phí gán nhãn |
| **Học trực tuyến** (online learning) | Cập nhật mô hình liên tục khi dữ liệu đến từng mẫu hoặc từng lô | Dữ liệu luồng, hệ thống thay đổi nhanh |
| **Học vài mẫu / không mẫu** (few-shot / zero-shot) | Làm tác vụ mới với vài hoặc không có ví dụ | LLM làm theo hướng dẫn trong prompt |
| **Học từ nhãn yếu** (weak supervision) | Dùng nhãn nhiễu từ luật, heuristic, mô hình khác | Snorkel, gán nhãn bằng luật |
| **Học liên kết** (federated learning) | Huấn luyện trên nhiều thiết bị mà không gom dữ liệu về một chỗ | Bàn phím gợi ý trên điện thoại |
| **Học tương phản** (contrastive learning) | Kéo các mẫu giống nhau lại gần, đẩy mẫu khác xa trong không gian biểu diễn | CLIP |
| **Meta-learning** | "Học cách học", thích nghi nhanh với tác vụ mới | MAML |

Bạn không cần thành thạo hết. Điều cần là biết **tên và ý tưởng** để nhận ra khi một bài toán thật rơi vào nhóm nào.

---

## 11. Phân loại theo các trục khác

Ngoài kiểu học, mô hình còn được phân loại theo nhiều trục độc lập:

### 11.1. Tham số và phi tham số

| | Có tham số (parametric) | Phi tham số (non-parametric) |
|---|---|---|
| Số tham số | Cố định, không phụ thuộc kích thước dữ liệu | Tăng theo dữ liệu |
| Giả định về hình dạng hàm | Mạnh (ví dụ tuyến tính) | Yếu |
| Ví dụ | Linear/Logistic Regression, Naive Bayes, mạng neural | KNN, cây quyết định, kernel SVM |
| Ưu | Nhanh, cần ít dữ liệu, dễ diễn giải | Linh hoạt, bắt được quan hệ phức tạp |
| Nhược | Sai nếu giả định sai (bias cao) | Cần nhiều dữ liệu, dễ overfit |

### 11.2. Phân biệt (discriminative) và sinh (generative)

- **Phân biệt:** học trực tiếp $P(y \mid x)$ hoặc ranh giới giữa các lớp. Ví dụ: Logistic Regression, SVM, cây, mạng neural. Thường chính xác hơn cho tác vụ dự đoán.
- **Sinh:** học $P(x, y) = P(x \mid y)P(y)$ rồi dùng Bayes để suy ra $P(y \mid x)$. Ví dụ: Naive Bayes, LDA, GMM. Sinh ra được dữ liệu mới, xử lý dữ liệu thiếu tốt, nhưng thường cần giả định mạnh về phân phối.

### 11.3. Theo cách lưu tri thức

- **Dựa trên mô hình (model-based):** học tham số rồi bỏ dữ liệu (Linear Regression).
- **Dựa trên mẫu (instance-based):** giữ dữ liệu và so sánh khi dự đoán (KNN). Gọi là "học lười" (lazy), vì không có bước huấn luyện rõ ràng.

### 11.4. Theo cách nạp dữ liệu

- **Batch (offline):** huấn luyện một lần trên toàn bộ dữ liệu, định kỳ huấn luyện lại.
- **Online:** cập nhật dần theo dữ liệu mới, phù hợp dữ liệu lớn không nạp vừa bộ nhớ hoặc phân phối thay đổi.

### 11.5. Theo dạng đầu ra

- **Xác định (deterministic):** trả về một giá trị.
- **Xác suất (probabilistic):** trả về phân phối hoặc xác suất. Cần khi phải biết độ chắc chắn.

### 11.6. Tuyến tính và phi tuyến

Mô hình tuyến tính (Linear/Logistic Regression, SVM tuyến tính) dễ diễn giải, nhanh. Mô hình phi tuyến (cây, kernel SVM, mạng neural) bắt được quan hệ phức tạp hơn.

---

## 12. Các thành phần của một hệ thống ML

Mọi thuật toán ML đều có thể phân rã thành năm thành phần. Đây là cách nghĩ giúp bạn so sánh các thuật toán khác nhau một cách có hệ thống.

| Thành phần | Câu hỏi | Ví dụ với Logistic Regression |
|---|---|---|
| **Dữ liệu** | Học từ đâu? | Bảng đặc trưng và nhãn 0/1 |
| **Mô hình** (không gian giả thuyết) | Họ hàm nào? | $p = \sigma(\mathbf{w}^\top\mathbf{x}+b)$ |
| **Hàm mất mát** | Sai thế nào thì bị phạt bao nhiêu? | Cross-entropy |
| **Thuật toán tối ưu** | Tìm tham số tốt bằng cách nào? | Gradient descent |
| **Đánh giá** | Bao nhiêu là tốt? | F1, ROC-AUC trên tập test |

### Tối thiểu hóa rủi ro thực nghiệm (ERM)

Ta muốn tối thiểu hóa **rủi ro thật** (lỗi kỳ vọng trên phân phối thật), nhưng chỉ thấy dữ liệu mẫu, nên tối thiểu hóa **rủi ro thực nghiệm** (lỗi trung bình trên dữ liệu train):

$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n} L\big(f(x_i), y_i\big)$$

Khoảng cách giữa lỗi train và lỗi thật chính là vấn đề **tổng quát hóa** và gốc của overfitting (xem [Chương 8](08-danh-gia-mo-hinh.md)).

### Thiên kiến quy nạp và định lý "No Free Lunch"

Không có thuật toán nào tốt nhất cho mọi bài toán. Mỗi mô hình mang một **thiên kiến quy nạp** (inductive bias): giả định ngầm về dạng quy luật (tuyến tính, trơn, theo cây...). Mô hình tốt là mô hình có thiên kiến phù hợp với bài toán. Vì vậy cần thử nhiều mô hình và đo bằng dữ liệu thật.

### Nguyên lý Occam

Trong các mô hình có hiệu năng tương đương, chọn mô hình **đơn giản hơn**: ít overfit hơn, dễ diễn giải hơn, dễ triển khai hơn.

---

## 13. Các loại dữ liệu

| Loại | Ví dụ | Cách biểu diễn | Hướng tiếp cận chính |
|---|---|---|---|
| **Bảng (tabular)** | Hồ sơ khách hàng, giao dịch | Hàng là mẫu, cột là đặc trưng | ML cổ điển (cây, GBM) thường mạnh nhất |
| **Văn bản** | Email, đánh giá | Chuỗi token | TF-IDF + mô hình tuyến tính; embedding và Transformer |
| **Ảnh** | Ảnh y tế, ảnh sản phẩm | Ma trận điểm ảnh | CNN, Vision Transformer (DL) |
| **Âm thanh** | Giọng nói, nhạc | Dạng sóng, phổ | CNN trên phổ, Transformer |
| **Chuỗi thời gian** | Giá cổ phiếu, cảm biến | Chuỗi giá trị theo thời gian | Đặc trưng trễ + GBM, ARIMA, RNN |
| **Đồ thị** | Mạng xã hội, phân tử | Nút và cạnh | Graph Neural Network |
| **Đa phương thức** | Ảnh + mô tả | Kết hợp nhiều loại | Mô hình đa phương thức |

Giai đoạn này tập trung vào **dữ liệu dạng bảng**, nơi ML cổ điển và gradient boosting thường thắng cả mạng neural. Ảnh, văn bản, âm thanh sẽ gặp lại ở GĐ2 và GĐ3.

Về cấu trúc, dữ liệu còn chia thành **có cấu trúc** (bảng), **bán cấu trúc** (JSON, XML) và **phi cấu trúc** (văn bản thô, ảnh).

---

## 14. Quy trình một dự án ML

```
Hiểu bài toán → Thu thập dữ liệu → EDA → Chia dữ liệu → Tiền xử lý
   → Baseline → Thử mô hình → Đánh giá → Tuning → Đánh giá cuối trên test
   → Triển khai → Giám sát → (quay lại cải thiện)
```

Quy trình tiêu chuẩn của ngành gọi là **CRISP-DM** (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment), thực tế là một vòng lặp chứ không phải đường thẳng.

Nguyên tắc xuyên suốt:

1. **Bắt đầu bằng câu hỏi, không phải mô hình.** Dự đoán cái gì? Sai thì tốn kém kiểu nào? Metric nào phản ánh đúng điều đó?
2. **Luôn có baseline.** Hồi quy: dự đoán trung bình. Phân loại: dự đoán lớp phổ biến nhất. Mô hình phức tạp không thắng baseline thì vô nghĩa.
3. **Chia dữ liệu trước khi làm bất cứ điều gì học từ dữ liệu** (điền giá trị thiếu, chuẩn hóa, chọn đặc trưng).
4. **Tập test chỉ dùng một lần ở cuối.**
5. **Đơn giản trước, phức tạp sau.**
6. **Dữ liệu thường quan trọng hơn thuật toán.** Thêm dữ liệu tốt và đặc trưng tốt hiệu quả hơn đổi mô hình.

Thời gian thực tế của một dự án: khoảng 60–80% dành cho dữ liệu (thu thập, làm sạch, đặc trưng), phần còn lại cho mô hình và triển khai.

---

## 15. Thuật ngữ cần thuộc

| Thuật ngữ | Ý nghĩa |
|---|---|
| Mẫu / quan sát (sample, instance, example) | Một hàng dữ liệu |
| Đặc trưng (feature, attribute, predictor) | Một cột đầu vào, ký hiệu $x_j$ |
| Nhãn / mục tiêu (label, target, response) | Giá trị cần dự đoán, ký hiệu $y$ |
| Tham số (parameter) | Thứ mô hình tự học, ví dụ trọng số $w$ |
| Siêu tham số (hyperparameter) | Thứ người dùng đặt trước, ví dụ learning rate, độ sâu cây |
| Hàm mất mát (loss function) | Đo sai số trên **một** mẫu |
| Hàm chi phí (cost/objective function) | Loss trung bình trên cả tập, cộng thêm regularization nếu có |
| Huấn luyện (training / fitting) | Tìm tham số tối thiểu hóa chi phí |
| Suy luận (inference / prediction) | Dùng mô hình đã học để dự đoán |
| Tập huấn luyện / validation / kiểm tra | Train / validation / test set |
| Epoch | Một lượt duyệt qua toàn bộ tập train |
| Batch / mini-batch | Nhóm mẫu dùng cho một lần cập nhật tham số |
| Tổng quát hóa (generalization) | Chạy tốt trên dữ liệu chưa thấy |
| Overfitting / underfitting | Học thuộc nhiễu / chưa học đủ quy luật |
| Regularization | Kỹ thuật hạn chế độ phức tạp để giảm overfit |
| Bias / variance | Sai số do quá đơn giản / do quá nhạy với dữ liệu train |
| Rò rỉ dữ liệu (data leakage) | Thông tin ngoài tập train lọt vào lúc học |
| Baseline | Mô hình đơn giản làm mốc so sánh |
| Pipeline | Chuỗi các bước xử lý và mô hình đóng gói thành một khối |
| Cross-validation | Chia dữ liệu thành nhiều phần, luân phiên làm validation |

---

## 16. Chọn kiểu bài toán từ câu hỏi thực tế

| Câu hỏi của doanh nghiệp | Kiểu bài toán | Gợi ý bắt đầu |
|---|---|---|
| "Căn nhà này giá bao nhiêu?" | Hồi quy | Linear Regression, rồi Random Forest/LightGBM |
| "Khách này có rời bỏ dịch vụ không?" | Phân loại nhị phân | Logistic Regression, rồi LightGBM; chú ý mất cân bằng |
| "Ảnh này thuộc loại nào trong 10 loại?" | Phân loại đa lớp (DL) | CNN, transfer learning (GĐ2) |
| "Khách hàng của ta chia thành những nhóm nào?" | Gom cụm | K-Means sau khi chuẩn hóa |
| "Giao dịch nào đáng ngờ?" | Phát hiện bất thường (hoặc phân loại nếu có nhãn gian lận) | Isolation Forest; nếu có nhãn thì GBM |
| "Sản phẩm nào hay mua cùng nhau?" | Luật kết hợp | Apriori, FP-Growth |
| "Tuần sau bán được bao nhiêu?" | Dự báo chuỗi thời gian | Đặc trưng trễ + GBM, so với baseline "giống tuần trước" |
| "Nên hiển thị mẫu quảng cáo nào?" | Bandit / học tăng cường | Epsilon-greedy, Thompson sampling |
| "Có ít nhãn, nhiều dữ liệu thô" | Bán giám sát hoặc học chủ động | Self-training, chọn mẫu để gán nhãn |
| "Giảm 200 cột xuống còn vài cột để vẽ hình" | Giảm chiều | PCA, UMAP |

---

**Tiếp theo:** [Chương 2: Toán nền tảng và tối ưu hóa](02-toan-nen-tang-va-toi-uu.md)
