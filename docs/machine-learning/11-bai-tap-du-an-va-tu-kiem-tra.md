# Chương 11: Bài tập, dự án và tự kiểm tra

Đọc không đủ. Chương này cho bạn bài tập theo từng chương, dự án cuối giai đoạn, câu hỏi tự kiểm tra, lỗi thường gặp và bảng thuật ngữ Anh–Việt.

## Mục lục

1. [Bài tập theo chương](#1-bài-tập-theo-chương)
2. [Dự án cuối giai đoạn](#2-dự-án-cuối-giai-đoạn)
3. [Bộ dữ liệu gợi ý](#3-bộ-dữ-liệu-gợi-ý)
4. [Tự kiểm tra](#4-tự-kiểm-tra)
5. [Lỗi thường gặp](#5-lỗi-thường-gặp)
6. [Bảng thuật ngữ Anh–Việt](#6-bảng-thuật-ngữ-anhviệt)

---

## 1. Bài tập theo chương

Mức độ: ★ cơ bản, ★★ vừa, ★★★ thử thách.

### Chương 1: Tổng quan

- ★ Lấy 10 bài toán thực tế bất kỳ (lọc thư rác, gợi ý phim, dự báo thời tiết, nhận diện khuôn mặt...). Với mỗi bài, xác định: kiểu học, dạng bài toán, đầu vào/đầu ra, loại dữ liệu.
- ★ Cho ba tình huống: (a) 1 triệu ảnh chưa nhãn, (b) 500 ca bệnh có chẩn đoán, (c) một robot học đi. Chọn kiểu học phù hợp và giải thích.
- ★★ Chọn một bài toán và liệt kê rõ năm thành phần (dữ liệu, mô hình, loss, tối ưu, đánh giá).

### Chương 2: Dữ liệu

- ★ EDA một dataset: danh sách thiếu dữ liệu, phân phối nhãn, 5 phát hiện đáng chú ý, kèm hình.
- ★ So sánh ba cách điền giá trị thiếu (median, KNN, thêm cột cờ) bằng CV.
- ★★ **Gây rò rỉ có chủ đích:** fit scaler/imputer trên toàn bộ dữ liệu rồi mới chia; so điểm với cách làm đúng. Thử thêm một cột suy ra từ nhãn. Ghi lại mức "đẹp giả".
- ★★ So sánh One-hot, ordinal, target encoding trên bộ dữ liệu có biến phân loại nhiều giá trị.
- ★★ Dữ liệu mất cân bằng: so `class_weight`, SMOTE (trong pipeline) và chỉnh ngưỡng; báo cáo PR-AUC. Thử SMOTE **trước khi** chia và xem rò rỉ ra sao.
- ★★★ Dựng chuỗi tạo đặc trưng thời gian (trễ, trung bình trượt) với chia dữ liệu theo thời gian; chứng minh chia ngẫu nhiên cho điểm cao giả.

### Chương 3: Đánh giá

- ★ Tính tay precision, recall, F1 từ một ma trận nhầm lẫn; kiểm tra bằng `sklearn`.
- ★★ Trên dữ liệu lệch 1:99, chứng minh accuracy 99% của mô hình "luôn đoán âm"; chọn metric đúng.
- ★★ Vẽ learning curve, validation curve; chẩn đoán bias hay variance và đề xuất việc cần làm.
- ★★ Bootstrap khoảng tin cậy cho F1 trên tập test.
- ★★★ Cài nested CV và so với CV đơn lẻ có tuning; giải thích chênh lệch.
- ★★★ So hai mô hình bằng corrected resampled t-test hoặc bootstrap; kết luận có ý nghĩa không.

### Chương 4: Toán và tối ưu hóa

- ★ Tự suy ra gradient của MSE theo $\mathbf w$ và kiểm tra bằng đạo hàm số (chênh lệch hữu hạn).
- ★★ Cài mini-batch SGD cho hồi quy tuyến tính; thử nhiều learning rate và batch size, vẽ đường loss, rút ra nhận xét.
- ★★ Cài Momentum và Adam bằng NumPy, so với SGD thường trên cùng bài toán.
- ★★ Chứng minh (trên giấy) MSE là MLE khi nhiễu Gauss, và cross-entropy là MLE của Bernoulli.
- ★★★ Cài bootstrap để ước lượng khoảng tin cậy của hệ số hồi quy; so với kết quả của `statsmodels`.

### Chương 5: Hồi quy và phân loại tuyến tính

- ★★ Tự cài `LinearRegressionGD`, kiểm tra khớp `LinearRegression` (sai khác < $10^{-4}$).
- ★★ Tự cài `LogisticRegressionGD` có L2; so với `LogisticRegression`. Vẽ đường loss và ranh giới quyết định 2D.
- ★★ Hồi quy đa thức: vẽ lỗi train/validation theo bậc 1→15, chỉ ra điểm bắt đầu overfit; thêm Ridge và quan sát.
- ★★ Dùng Lasso trên dữ liệu 100 đặc trưng; vẽ đường đi trọng số theo $\alpha$.
- ★★ Chẩn đoán: vẽ phần dư, tính VIF, đề xuất sửa đổi.
- ★★★ Chọn ngưỡng quyết định tối thiểu chi phí khi FN tốn gấp 10 lần FP; vẽ chi phí theo ngưỡng.
- ★★★ Hiệu chỉnh xác suất của Random Forest bằng Platt/isotonic; so Brier score và đường hiệu chỉnh.

### Chương 6: Cây và ensemble

- ★★ Tự cài hàm tính Gini và tìm điểm chia tốt nhất cho một đặc trưng; mở rộng thành cây nông độ sâu 2–3.
- ★★ Chạy `SimpleGBR` trong chương 6 và so với `GradientBoostingRegressor`.
- ★★ So cây đơn, Bagging, Random Forest, LightGBM trên cùng CV; vẽ validation curve theo `max_depth` hoặc số cây.
- ★★ Dùng early stopping để chọn số cây LightGBM; vẽ loss train/validation.
- ★★★ Dựng stacking với 3 mô hình khác loại; so với mô hình đơn tốt nhất và voting. Kiểm tra bạn tránh được rò rỉ.

### Chương 7: SVM, KNN, Naive Bayes

- ★ Tự cài KNN bằng NumPy; so với `KNeighborsClassifier`; thử các $k$ và vẽ lỗi theo $k$.
- ★★ SVM: lưới `C × gamma`, vẽ heatmap điểm CV. Vẽ ranh giới quyết định với dữ liệu 2D "hai vầng trăng" cho kernel tuyến tính và RBF.
- ★★ Phân loại văn bản bằng TF-IDF + Naive Bayes, Logistic, Linear SVM; so sánh.
- ★★★ Tự cài Gaussian Naive Bayes từ đầu.

### Chương 8: Học không giám sát

- ★ K-Means trên dữ liệu khách hàng; chọn $K$ bằng elbow và silhouette; đặt tên cho cụm.
- ★★ Tự cài K-Means bằng NumPy; so với `KMeans` (inertia).
- ★★ So K-Means, DBSCAN, GMM trên dữ liệu "hai vầng trăng" và "vòng tròn đồng tâm"; giải thích vì sao khác nhau.
- ★★ PCA: tự cài bằng SVD, kiểm tra khớp `PCA`; vẽ scree plot; dựng lại ảnh chữ số từ k thành phần.
- ★★ Trực quan hóa MNIST bằng PCA, t-SNE, UMAP; so sánh.
- ★★ Phát hiện bất thường bằng Isolation Forest trên dữ liệu giao dịch; xem top 20 mẫu.
- ★★★ Luật kết hợp trên dữ liệu giỏ hàng (Online Retail); lọc theo lift và diễn giải.

### Chương 9: Pipeline, triển khai

- ★★ Dựng pipeline `ColumnTransformer` đầy đủ; chứng minh không rò rỉ.
- ★★ Tuning bằng Optuna 50 lần thử; vẽ lịch sử tối ưu và độ quan trọng tham số.
- ★★ Giải thích mô hình bằng permutation importance, PDP và SHAP; viết 5 nhận xét.
- ★★ Đóng gói mô hình thành API FastAPI, gọi bằng `curl` và `/docs`.
- ★★★ Viết hàm PSI, mô phỏng drift (đổi phân phối một đặc trưng) và xem điểm mô hình tụt ra sao.
- ★★★ Viết bộ test `pytest` cho pipeline (ít nhất 5 test).

### Chương 10: Các kiểu học khác

- ★★ Bán giám sát: giữ 5% nhãn của một dataset, so (a) chỉ phần có nhãn, (b) Self-training, (c) Label Spreading. Vẽ điểm theo tỉ lệ nhãn.
- ★★ Học chủ động: so uncertainty sampling với chọn ngẫu nhiên; vẽ điểm theo số nhãn đã dùng.
- ★★ Học trực tuyến: dùng `partial_fit` trên dữ liệu luồng giả lập có drift; so với mô hình tĩnh.
- ★★ Bandit: so $\varepsilon$-greedy, UCB và Thompson sampling; vẽ regret tích lũy.
- ★★★ Q-learning trên FrozenLake (có `is_slippery=True`); so tốc độ hội tụ khi đổi $\alpha$, $\varepsilon$.

---

---

## 2. Dự án cuối giai đoạn

Thư mục: `projects/01-house-price-ml/` (hoặc chọn bài toán riêng của bạn).

### 2.1. Mục tiêu

Dựng quy trình ML hoàn chỉnh, từ dữ liệu thô tới mô hình có thể triển khai, và chứng minh bằng số liệu mỗi quyết định.

### 2.2. Yêu cầu tối thiểu

**Dữ liệu và phân tích**
- [ ] Mô tả bài toán, nguồn dữ liệu, ý nghĩa cột
- [ ] EDA có biểu đồ và nhận xét (phân phối nhãn, thiếu dữ liệu, outlier, tương quan)
- [ ] Rà soát rò rỉ: ghi lý do giữ/loại từng cột đáng ngờ

**Mô hình hóa**
- [ ] Chia train/test đúng kiểu (và có lý do chọn kiểu đó); test chỉ mở một lần
- [ ] Baseline đơn giản + ít nhất 3 mô hình khác họ (tuyến tính, cây/ensemble, một mô hình khác)
- [ ] Pipeline dùng `ColumnTransformer`, không rò rỉ
- [ ] Feature engineering, có **ablation** chứng minh đặc trưng mới giúp ích
- [ ] Tuning bằng Optuna hoặc random search, có nhật ký các lần thử

**Đánh giá**
- [ ] CV với trung bình ± độ lệch chuẩn, cùng các fold cho mọi mô hình
- [ ] Learning curve và chẩn đoán bias/variance
- [ ] Đánh giá cuối trên test (một lần) kèm khoảng tin cậy bootstrap
- [ ] Phân tích lỗi: top mẫu sai nhiều nhất và đánh giá theo lát cắt (nhóm)
- [ ] Giải thích: permutation importance và SHAP

**Đóng gói**
- [ ] Code trong `src/`, notebook chỉ để khám phá
- [ ] Ít nhất 5 test `pytest`
- [ ] Lưu pipeline (joblib) với phiên bản thư viện
- [ ] API FastAPI đơn giản có kiểm tra đầu vào
- [ ] `README.md`: bài toán, cách làm, bảng kết quả, hạn chế, bài học

### 2.3. Cách chấm điểm bản thân

| Tiêu chí | Chưa đạt | Đạt | Tốt |
|---|---|---|---|
| Không rò rỉ | Có rò rỉ | Pipeline đúng | Có test chứng minh |
| So sánh mô hình | Một mô hình | Ba mô hình, có CV | Có ± std, ablation, kiểm định |
| Giải thích | Không có | Feature importance | SHAP + phân tích lát cắt + nêu hạn chế |
| Tái lập | Chạy lại ra khác | Seed cố định | Một lệnh chạy lại toàn bộ, có phiên bản dữ liệu |
| Nhận xét | Liệt kê số | Giải thích số | Giải thích vì sao + việc làm tiếp |

### 2.4. Hướng mở rộng

- Thêm chuỗi thời gian (giá theo thời điểm) với chia dữ liệu theo thời gian.
- Dự đoán **khoảng** (phân vị 10%–90%) bằng quantile regression.
- Gom cụm khu vực rồi dùng nhãn cụm làm đặc trưng.
- Mô phỏng drift và xây dashboard giám sát đơn giản.
- Làm một dự án thứ hai về **phân loại** với dữ liệu mất cân bằng (churn hoặc gian lận).

---

## 3. Bộ dữ liệu gợi ý

| Bộ dữ liệu | Loại | Gợi ý dùng cho |
|---|---|---|
| Titanic | Phân loại, bảng nhỏ | Làm quen quy trình, thiếu dữ liệu, mã hóa |
| California Housing (`sklearn.datasets`) | Hồi quy | Dự án nhập môn, có tọa độ |
| Ames Housing | Hồi quy, nhiều cột, nhiều biến phân loại | Feature engineering, dự án chính |
| Adult Income (UCI) | Phân loại | Công bằng/thiên lệch, biến phân loại |
| Telco Customer Churn | Phân loại, hơi lệch | Dự đoán rời bỏ |
| Credit Card Fraud | Phân loại, **cực lệch** | PR-AUC, lấy mẫu lại, ngưỡng |
| Wine Quality, Diabetes | Hồi quy/phân loại nhỏ | Thực hành thuật toán |
| MNIST / `load_digits` | Ảnh nhỏ | PCA, t-SNE, K-Means, SVM |
| Mall Customers, Online Retail (UCI) | Không giám sát | Gom cụm, luật kết hợp |
| 20 Newsgroups (`sklearn.datasets`) | Văn bản | TF-IDF + Naive Bayes/Logistic |
| Bike Sharing, Store Sales | Chuỗi thời gian dạng bảng | Chia theo thời gian, đặc trưng trễ |

Nguồn: Kaggle, UCI Machine Learning Repository, OpenML, `sklearn.datasets`, Hugging Face Datasets.

---

## 4. Tự kiểm tra

Bạn đã xong GĐ1 khi trả lời và làm được các việc dưới đây **không cần xem tài liệu**.

### 4.1. Khái niệm

- [ ] Phân biệt học có giám sát, không giám sát, bán giám sát, tự giám sát, tăng cường; mỗi loại cho một ví dụ thật.
- [ ] Hồi quy khác phân loại thế nào? Cho ví dụ chuyển bài toán từ loại này sang loại kia và mất gì.
- [ ] Tham số khác siêu tham số thế nào? Loss khác cost thế nào?
- [ ] Mô hình phân biệt khác mô hình sinh thế nào? Cho ví dụ mỗi loại.
- [ ] "No free lunch" và "inductive bias" nghĩa là gì?

### 4.2. Toán và tối ưu

- [ ] Viết gradient descent cho hồi quy tuyến tính và logistic; vì sao gradient của hai mô hình có dạng giống nhau?
- [ ] Điều gì xảy ra khi learning rate quá lớn/nhỏ? Cách nhận biết trên đường loss?
- [ ] Vì sao MSE là hợp lý khi nhiễu Gauss, và cross-entropy cho phân loại?
- [ ] L1 và L2 khác nhau thế nào về tác động lên trọng số, về hình học, về prior?
- [ ] Vì sao nên chuẩn hóa trước khi dùng gradient descent, KNN, SVM, PCA, Ridge?

### 4.3. Dữ liệu và đặc trưng

- [ ] Rò rỉ dữ liệu là gì? Nêu 4 dạng và cách phòng.
- [ ] Khi nào chia theo nhóm hoặc theo thời gian thay vì ngẫu nhiên?
- [ ] Ba cơ chế thiếu dữ liệu (MCAR, MAR, MNAR) khác nhau thế nào? Cách xử lý?
- [ ] Khi nào dùng one-hot, ordinal, target encoding? Rủi ro của target encoding?
- [ ] Mô hình nào cần và không cần chuẩn hóa? Giải thích.
- [ ] Cách xử lý dữ liệu mất cân bằng đúng, và cạm bẫy khi lấy mẫu lại.

### 4.4. Mô hình

- [ ] Vẽ cây quyết định; Gini và entropy tính thế nào; vì sao một cây dễ overfit?
- [ ] Random Forest giảm variance, boosting giảm bias: nghĩa là gì? Vì sao cần cây đa dạng?
- [ ] Gradient boosting học phần dư/gradient âm nghĩa là gì? Vai trò của learning rate và số cây?
- [ ] SVM: lề là gì, support vector là gì, $C$ và $\gamma$ điều khiển gì?
- [ ] Vì sao Naive Bayes "ngây thơ" mà vẫn hiệu quả? Nhược điểm về xác suất?
- [ ] KNN: $k$ nhỏ/lớn ảnh hưởng thế nào? Vì sao lời nguyền số chiều làm nó kém?
- [ ] So sánh K-Means, DBSCAN, GMM: giả định và lúc dùng.
- [ ] PCA làm gì, tính thế nào, khi nào không nên dùng?

### 4.5. Đánh giá

- [ ] Khi nào accuracy gây hiểu lầm? Chọn metric nào thay thế?
- [ ] Precision, recall, F1, ROC-AUC, PR-AUC: ý nghĩa và khi dùng mỗi cái.
- [ ] Vì sao ROC-AUC có thể cao dù mô hình tệ trên dữ liệu rất lệch?
- [ ] Overfitting là gì, phát hiện bằng cách nào, nêu ít nhất 5 cách xử lý?
- [ ] Đọc learning curve để biết nên thêm dữ liệu hay đổi mô hình.
- [ ] Vì sao tập test chỉ dùng một lần? Nested CV giải quyết điều gì?
- [ ] Làm sao biết chênh lệch giữa hai mô hình là thật?
- [ ] Cách chọn ngưỡng quyết định theo chi phí lỗi.

### 4.6. Thực hành

- [ ] Tự cài hồi quy tuyến tính và logistic bằng NumPy không xem code.
- [ ] Tự cài K-Means và KNN bằng NumPy.
- [ ] Dựng pipeline `ColumnTransformer` + mô hình + CV từ đầu trong 15 phút.
- [ ] Tuning bằng Optuna và giải thích kết quả.
- [ ] Viết một lời giải thích SHAP cho một dự đoán cụ thể.
- [ ] Đóng gói mô hình thành API và gọi thử.

---

## 5. Lỗi thường gặp

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| Fit scaler/imputer/chọn đặc trưng trên toàn bộ dữ liệu | Rò rỉ, điểm đẹp giả | `Pipeline`; chia trước, học sau |
| Chọn mô hình hoặc tuning theo điểm test | Test không còn khách quan | Chọn bằng CV; test dùng một lần |
| Dùng accuracy cho dữ liệu lệch | Tưởng tốt nhưng vô dụng | PR-AUC, F1, recall |
| Không có baseline | Không biết mô hình có giá trị | Luôn đặt baseline trước |
| Xáo trộn dữ liệu thời gian | Nhìn thấy tương lai | `TimeSeriesSplit`, chia theo thời gian |
| Chia ngẫu nhiên khi có nhóm | Rò rỉ nhóm | `GroupKFold` |
| Oversample/SMOTE trước khi chia | Rò rỉ | Lấy mẫu lại trong pipeline, từng fold |
| Quên chuẩn hóa cho KNN/SVM/PCA/regularization | Kết quả sai lệch | Đưa scaler vào pipeline |
| Tuning quá nhiều trên một tập validation | Overfit vào validation | CV, giới hạn số lần thử |
| Đặc trưng có thông tin không có lúc dự đoán thật | Mô hình vô dụng khi triển khai | Hỏi từng cột: "lúc dự đoán thật có không?" |
| Không đặt `random_state` | Không tái lập | Cố định seed |
| Nhìn một con số duy nhất | Bỏ sót vấn đề | Thêm learning curve, lát cắt, phân tích lỗi |
| Diễn giải tương quan như nhân quả | Kết luận sai | Phân biệt dự đoán và suy diễn nhân quả |
| Dùng xác suất chưa hiệu chỉnh để ra quyết định | Quyết định lệch | Calibration, kiểm tra Brier score |
| Hiểu `C` của SVM/Logistic ngược chiều | Regularization sai hướng | Nhớ: `C` là nghịch đảo regularization |
| Xóa outlier chỉ để điểm đẹp | Mô hình hỏng với dữ liệu thật | Hỏi: triển khai thật có gặp không? |
| Tin hình t-SNE/UMAP quá mức | Hiểu sai cấu trúc | Chỉ dùng để khám phá, thử nhiều tham số |
| Nạp file `pickle`/`joblib` không rõ nguồn | Thực thi mã độc | Chỉ nạp từ nguồn tin cậy |

**Thói quen tốt nhất sau khi có mô hình:** đọc những mẫu mô hình dự đoán sai nhiều nhất. Chúng thường chỉ ra lỗi dữ liệu hoặc đặc trưng còn thiếu.

---

## 6. Bảng thuật ngữ Anh–Việt

| English | Tiếng Việt |
|---|---|
| Supervised / unsupervised / semi-supervised / self-supervised learning | Học có giám sát / không giám sát / bán giám sát / tự giám sát |
| Reinforcement learning | Học tăng cường |
| Transfer learning | Học chuyển giao |
| Active / online learning | Học chủ động / trực tuyến |
| Feature / label (target) | Đặc trưng / nhãn (mục tiêu) |
| Sample (instance) | Mẫu (quan sát) |
| Training / validation / test set | Tập huấn luyện / kiểm định / kiểm tra |
| Parameter / hyperparameter | Tham số / siêu tham số |
| Loss function / cost function | Hàm mất mát / hàm chi phí |
| Gradient descent | Hạ gradient (giảm theo gradient) |
| Learning rate | Tốc độ học |
| Epoch / batch | Epoch (lượt duyệt toàn bộ dữ liệu) / lô |
| Overfitting / underfitting | Quá khớp / chưa khớp |
| Generalization | Tổng quát hóa |
| Bias / variance | Độ chệch / phương sai |
| Regularization | Điều chuẩn (chính quy hóa) |
| Cross-validation | Kiểm định chéo |
| Data leakage | Rò rỉ dữ liệu |
| Pipeline | Chuỗi xử lý (pipeline) |
| Baseline | Mô hình cơ sở (mốc so sánh) |
| Feature engineering / selection / extraction | Kỹ thuật đặc trưng / chọn / trích xuất đặc trưng |
| One-hot encoding | Mã hóa one-hot |
| Imputation | Điền giá trị thiếu |
| Outlier | Điểm ngoại lai |
| Class imbalance | Mất cân bằng lớp |
| Confusion matrix | Ma trận nhầm lẫn |
| Precision / recall | Độ chính xác dương / độ phủ (độ nhạy) |
| ROC curve / AUC | Đường cong ROC / diện tích dưới đường cong |
| Calibration | Hiệu chỉnh xác suất |
| Decision boundary | Ranh giới quyết định |
| Decision tree / random forest | Cây quyết định / rừng ngẫu nhiên |
| Bagging / boosting / stacking | Bagging / boosting / xếp chồng |
| Ensemble | Tập hợp mô hình |
| Margin / support vector | Lề / vector hỗ trợ |
| Kernel | Hàm nhân |
| Clustering | Gom cụm |
| Dimensionality reduction | Giảm chiều |
| Anomaly detection | Phát hiện bất thường |
| Association rule | Luật kết hợp |
| Explained variance | Phương sai giải thích được |
| Drift | Trôi dữ liệu (thay đổi phân phối) |
| Interpretability / explainability | Khả năng diễn giải / giải thích |

---

**Trước:** [Chương 10](10-cac-kieu-hoc-khac.md) | **Mục lục:** [README](README.md)
