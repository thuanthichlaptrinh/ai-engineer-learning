# Chương 6: SVM, KNN, Naive Bayes và các mô hình khác

Ba họ mô hình với ba cách tiếp cận rất khác nhau: **hình học** (SVM), **dựa trên mẫu** (KNN), **xác suất** (Naive Bayes, LDA). Chúng ít dùng hơn boosting cho dữ liệu bảng lớn, nhưng quan trọng để hiểu các ý tưởng nền và vẫn hiệu quả ở những bài toán nhỏ hoặc đặc thù.

## Mục lục

1. [Support Vector Machine (SVM)](#1-support-vector-machine-svm)
2. [K-Nearest Neighbors (KNN)](#2-k-nearest-neighbors-knn)
3. [Naive Bayes](#3-naive-bayes)
4. [LDA và QDA](#4-lda-và-qda)
5. [Mô hình phân biệt và mô hình sinh](#5-mô-hình-phân-biệt-và-mô-hình-sinh)
6. [Bảng so sánh tổng hợp](#6-bảng-so-sánh-tổng-hợp)
7. [Khi nào chọn mô hình nào](#7-khi-nào-chọn-mô-hình-nào)

---

## 1. Support Vector Machine (SVM)

### 1.1. Ý tưởng: lề lớn nhất

Giữa hai lớp có thể tách tuyến tính, có vô số siêu phẳng phân tách. SVM chọn siêu phẳng có **lề (margin) lớn nhất**: khoảng cách tới điểm gần nhất của mỗi lớp là xa nhất. Lề lớn nghĩa là ranh giới ít nhạy với nhiễu, tổng quát hóa tốt hơn.

Các điểm nằm sát biên gọi là **vector hỗ trợ (support vectors)**. Chỉ chúng quyết định ranh giới; xóa các điểm khác thì mô hình không đổi.

```
   Lớp -1   ○ ○            ┆ lề ┆            ● ●  Lớp +1
          ○   ○ ○  ←——  ————┼————  ——→  ● ●   ●
            ○              ┆    ┆              ●
                       ranh giới w·x + b = 0
```

### 1.2. Công thức

Với nhãn $y_i\in\{-1,+1\}$.

**Hard margin** (dữ liệu tách được hoàn toàn):

$$\min_{\mathbf w,b}\;\tfrac12\lVert\mathbf w\rVert^2\quad\text{sao cho}\quad y_i(\mathbf w^\top\mathbf x_i+b)\ge1\;\;\forall i$$

Độ rộng lề bằng $2/\lVert\mathbf w\rVert$, nên cực tiểu $\lVert\mathbf w\rVert$ là cực đại lề.

**Soft margin** (cho phép vài điểm vi phạm, thực tế hơn):

$$\min_{\mathbf w,b,\xi}\;\tfrac12\lVert\mathbf w\rVert^2+C\sum_i\xi_i\quad\text{sao cho}\quad y_i(\mathbf w^\top\mathbf x_i+b)\ge1-\xi_i,\;\;\xi_i\ge0$$

$\xi_i$ là mức vi phạm của mẫu $i$. Tương đương với việc cực tiểu **hinge loss** cộng regularization L2:

$$\sum_i\max\big(0,\,1-y_if(\mathbf x_i)\big)+\lambda\lVert\mathbf w\rVert^2$$

### 1.3. Siêu tham số $C$

| $C$ | Hành vi |
|---|---|
| Lớn | Phạt nặng vi phạm ⇒ lề hẹp, cố phân loại đúng từng điểm ⇒ dễ **overfit** |
| Nhỏ | Chấp nhận nhiều vi phạm ⇒ lề rộng, ranh giới mượt ⇒ dễ **underfit** |

### 1.4. Thủ thuật kernel

Với dữ liệu không tách được bằng siêu phẳng, ánh xạ lên không gian chiều cao hơn $\phi(\mathbf x)$ rồi tách tuyến tính ở đó. **Kernel trick:** dạng đối ngẫu của SVM chỉ cần tích vô hướng $\phi(\mathbf x)^\top\phi(\mathbf x')$, nên chỉ cần tính hàm kernel $K(\mathbf x,\mathbf x')$ mà không cần ánh xạ tường minh.

| Kernel | Công thức | Ghi chú |
|---|---|---|
| **Tuyến tính** | $\mathbf x^\top\mathbf x'$ | Nhanh, tốt khi nhiều đặc trưng (văn bản) |
| **Đa thức** | $(\gamma\,\mathbf x^\top\mathbf x'+r)^d$ | Bậc $d$ |
| **RBF (Gauss)** | $\exp(-\gamma\lVert\mathbf x-\mathbf x'\rVert^2)$ | Mặc định, bắt được ranh giới phức tạp |

Siêu tham số $\gamma$ của RBF: lớn ⇒ mỗi điểm ảnh hưởng hẹp, ranh giới uốn lượn ⇒ overfit; nhỏ ⇒ ranh giới mượt, gần tuyến tính.

### 1.5. Thực hành

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, SVR

svm = make_pipeline(StandardScaler(),
                    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
linear = make_pipeline(StandardScaler(), LinearSVC(C=1.0))     # nhanh hơn nhiều
svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
```

- **Bắt buộc chuẩn hóa đặc trưng.**
- Chọn $C$ và $\gamma$ bằng lưới theo thang log (`C` ∈ {0.1, 1, 10, 100}, `gamma` ∈ {0.001, 0.01, 0.1, 1}) kết hợp cross-validation.
- `probability=True` cho xác suất (bằng Platt scaling bên trong), nhưng chậm hơn đáng kể.
- **SVR** (hồi quy) dùng loss $\varepsilon$-insensitive: không phạt sai số nhỏ hơn $\varepsilon$.
- Độ phức tạp huấn luyện kernel SVM khoảng $O(n^2)$–$O(n^3)$: chậm khi $n$ vượt vài chục nghìn. Khi đó dùng `LinearSVC`, `SGDClassifier(loss="hinge")`, hoặc đổi sang boosting.

### 1.6. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Hiệu quả trong không gian nhiều chiều | Chậm với dữ liệu lớn |
| Tốt khi ít mẫu, nhiều đặc trưng | Nhạy với thang đo, cần chuẩn hóa |
| Linh hoạt nhờ kernel | Không cho xác suất tự nhiên |
| Có nền tảng lý thuyết chắc | Khó diễn giải, khó tune $C,\gamma$ |

---

## 2. K-Nearest Neighbors (KNN)

### 2.1. Ý tưởng

Dự đoán cho điểm mới dựa trên $k$ điểm **gần nhất** trong tập train:
- Phân loại: lớp phổ biến nhất trong $k$ láng giềng (có thể có trọng số theo khoảng cách).
- Hồi quy: trung bình (hoặc trung bình có trọng số) nhãn của $k$ láng giềng.

Không có bước huấn luyện thực sự (chỉ lưu dữ liệu), gọi là **học lười (lazy learning)** và **dựa trên mẫu (instance-based)**. Là mô hình phi tham số điển hình.

### 2.2. Các yếu tố quyết định

**Số láng giềng $k$:**
- $k$ nhỏ (1–3): ranh giới nhiễu, variance cao ⇒ overfit.
- $k$ lớn: ranh giới mượt, bias cao ⇒ underfit. Cực đoan: $k=n$ luôn dự đoán lớp đa số.
- Chọn bằng cross-validation. Phân loại nhị phân nên dùng $k$ lẻ để tránh hòa.

**Độ đo khoảng cách:**

| Khoảng cách | Công thức | Dùng khi |
|---|---|---|
| Euclid ($L_2$) | $\sqrt{\sum(x_j-x'_j)^2}$ | Mặc định, đặc trưng liên tục đã chuẩn hóa |
| Manhattan ($L_1$) | $\sum\lvert x_j-x'_j\rvert$ | Nhiều chiều, bền với outlier hơn |
| Cosine | $1-\frac{\mathbf x\cdot\mathbf x'}{\lVert\mathbf x\rVert\lVert\mathbf x'\rVert}$ | Văn bản, embedding (quan tâm hướng, không quan tâm độ dài) |
| Hamming | Số vị trí khác nhau | Đặc trưng nhị phân/phân loại |

### 2.3. Thực hành

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn = make_pipeline(
    StandardScaler(),                      # bắt buộc
    KNeighborsClassifier(n_neighbors=7, weights="distance"),
)
```

**Tự cài đặt bằng NumPy** (để hiểu):

```python
import numpy as np

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dist = np.linalg.norm(X_train - x, axis=1)    # khoảng cách tới mọi điểm train
        nearest = np.argsort(dist)[:k]                # chỉ số k điểm gần nhất
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        preds.append(labels[np.argmax(counts)])       # bỏ phiếu đa số
    return np.array(preds)
```

### 2.4. Điểm yếu

- **Bắt buộc chuẩn hóa**, vì đặc trưng có thang đo lớn sẽ lấn át khoảng cách.
- **Lời nguyền số chiều:** khi nhiều chiều, mọi điểm đều "xa như nhau", KNN suy giảm. Dùng chọn đặc trưng hoặc PCA trước.
- **Chậm khi dự đoán** ($O(n)$ mỗi truy vấn nếu duyệt tuần tự, tốn bộ nhớ vì giữ toàn bộ dữ liệu). Giảm bằng cấu trúc dữ liệu (KD-tree, Ball-tree) cho số chiều thấp, hoặc **tìm kiếm xấp xỉ (ANN)** như HNSW, FAISS cho số chiều cao: chính là kỹ thuật dùng trong vector database ở phần RAG.
- Nhạy với đặc trưng nhiễu/không liên quan.

### 2.5. Khi nào dùng

Bài toán nhỏ, số chiều thấp, ranh giới phức tạp; làm baseline phi tham số; hệ gợi ý dựa trên độ tương đồng; và là nền ý tưởng cho tìm kiếm vector.

---

## 3. Naive Bayes

### 3.1. Ý tưởng

Áp dụng định lý Bayes để tính xác suất mỗi lớp cho đặc trưng đã thấy:

$$P(c\mid x_1,\dots,x_d)=\frac{P(c)\,P(x_1,\dots,x_d\mid c)}{P(x_1,\dots,x_d)}\;\propto\;P(c)\prod_{j=1}^dP(x_j\mid c)$$

Bước then chốt: giả định **"ngây thơ"** rằng các đặc trưng **độc lập có điều kiện** theo lớp, nên xác suất đồng thời tách thành tích các xác suất đơn. Giả định này gần như luôn sai, nhưng mô hình vẫn thường hoạt động tốt vì ta chỉ cần xếp hạng đúng lớp, không cần xác suất chính xác.

Dự đoán: $\hat c=\arg\max_c\big[\log P(c)+\sum_j\log P(x_j\mid c)\big]$ (dùng log để tránh tràn số khi nhân nhiều xác suất nhỏ).

### 3.2. Các biến thể

| Biến thể | Mô hình $P(x_j\mid c)$ | Dùng cho |
|---|---|---|
| **Gaussian NB** | Phân phối chuẩn | Đặc trưng liên tục |
| **Multinomial NB** | Phân phối đa thức | Đếm từ, văn bản (bag-of-words, TF-IDF) |
| **Bernoulli NB** | Bernoulli | Đặc trưng nhị phân (từ có xuất hiện hay không) |
| **Categorical NB** | Categorical | Đặc trưng phân loại |

### 3.3. Làm mượt Laplace

Nếu một giá trị chưa từng xuất hiện trong lớp $c$ thì $P(x_j\mid c)=0$ và nhân vào tích làm cả xác suất bằng 0. Khắc phục: cộng thêm $\alpha$ (thường 1) vào mọi số đếm (`alpha` trong scikit-learn).

### 3.4. Ví dụ phân loại văn bản

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

spam_clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    MultinomialNB(alpha=0.5),
)
spam_clf.fit(texts_train, labels_train)
spam_clf.predict(["Bạn đã trúng thưởng, bấm vào link"])
```

### 3.5. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Rất nhanh, huấn luyện và dự đoán chỉ cần đếm | Giả định độc lập thường sai |
| Hoạt động tốt với ít dữ liệu | Xác suất đầu ra **không đáng tin** (quá tự tin); cần calibration |
| Mở rộng tốt với số chiều lớn (văn bản) | Kém khi các đặc trưng tương quan mạnh |
| Dễ cài đặt, ít siêu tham số | Hiếm khi đạt độ chính xác cao nhất |

Là baseline mạnh cho văn bản và dữ liệu rất nhiều chiều.

---

## 4. LDA và QDA

**Linear Discriminant Analysis** là mô hình **sinh**: giả định dữ liệu mỗi lớp có phân phối chuẩn nhiều chiều, tất cả lớp chia sẻ **cùng ma trận hiệp phương sai**. Khi đó ranh giới giữa các lớp là **tuyến tính**. **QDA** cho mỗi lớp một ma trận hiệp phương sai riêng nên ranh giới là **bậc hai** (linh hoạt hơn, cần nhiều dữ liệu hơn).

LDA còn dùng để **giảm chiều có giám sát**: chiếu dữ liệu xuống tối đa $C-1$ hướng làm các lớp tách nhau nhất (khác PCA vốn không nhìn nhãn).

```python
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

lda = LinearDiscriminantAnalysis()
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X_scaled, y)
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
```

Khi nào dùng: dữ liệu liên tục gần chuẩn, ít mẫu, cần mô hình nhanh và ổn định; hoặc giảm chiều có giám sát để trực quan hóa. LDA liên quan chặt với Logistic Regression nhưng thường tốt hơn khi các giả định đúng và dữ liệu rất ít.

---

## 5. Mô hình phân biệt và mô hình sinh

| | Phân biệt (discriminative) | Sinh (generative) |
|---|---|---|
| Học | $P(y\mid\mathbf x)$ hoặc ranh giới trực tiếp | $P(\mathbf x\mid y)$ và $P(y)$, rồi dùng Bayes |
| Ví dụ | Logistic, SVM, cây, mạng neural | Naive Bayes, LDA/QDA, GMM |
| Sinh được dữ liệu mới | Không | Có |
| Giả định về phân phối $\mathbf x$ | Không | Có (thường mạnh) |
| Ít dữ liệu | Cần nhiều hơn | Hội tụ nhanh hơn nếu giả định đúng |
| Nhiều dữ liệu | Thường chính xác hơn | Bị giới hạn bởi giả định sai |
| Thiếu giá trị | Khó xử lý | Xử lý tự nhiên bằng cách lấy tích phân |

Quy tắc kinh nghiệm: cho tác vụ **dự đoán thuần túy** với đủ dữ liệu, mô hình phân biệt thường thắng. Mô hình sinh có giá trị khi cần sinh dữ liệu, phát hiện bất thường, hoặc có rất ít nhãn. Các mô hình sinh sâu hiện đại (GAN, diffusion, LLM) cũng thuộc nhóm "sinh".

---

## 6. Bảng so sánh tổng hợp

| Mô hình | Loại | Cần chuẩn hóa? | Xử lý phi tuyến? | Tốc độ huấn luyện | Tốc độ dự đoán | Diễn giải | Dữ liệu lớn? |
|---|---|---|---|---|---|---|---|
| Linear / Logistic | Tham số | Có (nếu regularize) | Không (cần tự tạo đặc trưng) | Rất nhanh | Rất nhanh | Cao | Tốt |
| Cây đơn | Phi tham số | Không | Có | Nhanh | Rất nhanh | Cao | Tốt |
| Random Forest | Ensemble | Không | Có | Trung bình | Trung bình | Trung bình | Tốt |
| Gradient Boosting | Ensemble | Không | Có | Chậm hơn | Nhanh | Thấp (dùng SHAP) | Tốt |
| SVM (kernel) | Hình học | **Có** | Có (kernel) | Chậm | Chậm khi nhiều support vector | Thấp | Kém |
| KNN | Dựa trên mẫu | **Có** | Có | Không có | Chậm | Trung bình | Kém |
| Naive Bayes | Sinh | Không | Hạn chế | Cực nhanh | Cực nhanh | Cao | Tốt |
| LDA/QDA | Sinh | Nên có | LDA không, QDA có | Nhanh | Nhanh | Cao | Tốt |

---

## 7. Khi nào chọn mô hình nào

**Quy trình mặc định cho dữ liệu bảng:** baseline → Logistic/Ridge → Random Forest → LightGBM (xem [Chương 5](05-cay-va-ensemble.md)).

**Các tình huống đặc thù:**

| Tình huống | Gợi ý |
|---|---|
| Văn bản, nhiều chiều, nhanh | TF-IDF + Logistic Regression hoặc Linear SVM hoặc Naive Bayes |
| Rất ít mẫu, ít đặc trưng | Logistic có regularization, LDA, SVM kernel, Naive Bayes |
| Cần giải thích cho người ngoài ngành | Logistic Regression, cây nông |
| Cần xác suất đáng tin | Logistic Regression, hoặc calibrate mô hình khác |
| Dữ liệu cực lớn, cần nhanh | Mô hình tuyến tính với SGD, LightGBM |
| Tìm "giống nhau nhất" | KNN, tìm kiếm vector (ANN) |
| Chưa biết bắt đầu từ đâu | Thử vài mô hình trên cùng các fold CV rồi so sánh, đừng đoán |

Nhắc lại **No Free Lunch**: không mô hình nào luôn thắng. Quyết định cuối cùng phải dựa trên cross-validation chứ không phải trực giác.

---

**Trước:** [Chương 5](05-cay-va-ensemble.md) | **Tiếp theo:** [Chương 7: Học không giám sát](07-hoc-khong-giam-sat.md)
