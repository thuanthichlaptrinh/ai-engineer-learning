# Machine Learning: tài liệu học tập

Tài liệu này đi cùng **GĐ1** trong [roadmap](roadmap.md). Học xong bạn cần làm được: nhận một bài toán dữ liệu dạng bảng, dựng pipeline đầy đủ, chọn và đánh giá mô hình đúng cách, và giải thích được vì sao.

**Điều kiện đầu vào:** đã xong GĐ0 (Python, NumPy, Pandas, đại số tuyến tính, đạo hàm, xác suất cơ bản).

## Mục lục

1. [Machine Learning là gì](#1-machine-learning-là-gì)
2. [Quy trình một dự án ML](#2-quy-trình-một-dự-án-ml)
3. [Chuẩn bị dữ liệu](#3-chuẩn-bị-dữ-liệu)
4. [Hồi quy tuyến tính](#4-hồi-quy-tuyến-tính)
5. [Hồi quy logistic](#5-hồi-quy-logistic)
6. [Đánh giá mô hình](#6-đánh-giá-mô-hình)
7. [Overfitting, bias-variance, regularization](#7-overfitting-bias-variance-regularization)
8. [Cây quyết định và các mô hình tổ hợp](#8-cây-quyết-định-và-các-mô-hình-tổ-hợp)
9. [SVM, KNN, Naive Bayes](#9-svm-knn-naive-bayes)
10. [Học không giám sát: Clustering và PCA](#10-học-không-giám-sát-clustering-và-pca)
11. [Feature engineering và dữ liệu mất cân bằng](#11-feature-engineering-và-dữ-liệu-mất-cân-bằng)
12. [Pipeline và tuning siêu tham số](#12-pipeline-và-tuning-siêu-tham-số)
13. [Thiết kế thí nghiệm và giải thích mô hình](#13-thiết-kế-thí-nghiệm-và-giải-thích-mô-hình)
14. [Lỗi thường gặp](#14-lỗi-thường-gặp)
15. [Kế hoạch học, bài tập, dự án](#15-kế-hoạch-học-bài-tập-dự-án)
16. [Tài liệu tham khảo](#16-tài-liệu-tham-khảo)

## Cài đặt môi trường

```bash
uv add numpy pandas matplotlib seaborn scikit-learn scipy lightgbm xgboost optuna shap jupyter
```

---

## 1. Machine Learning là gì

Thay vì viết luật bằng tay, ta để máy **học luật từ dữ liệu**. Cho dữ liệu $(x_i, y_i)$, tìm hàm $f$ sao cho $f(x) \approx y$ trên dữ liệu **chưa từng thấy**. Chữ "chưa từng thấy" là điểm mấu chốt: mục tiêu là *tổng quát hóa*, không phải thuộc lòng dữ liệu huấn luyện.

### Các loại bài toán

| Loại | Có nhãn $y$? | Ví dụ | Mô hình tiêu biểu |
|---|---|---|---|
| **Hồi quy** (supervised) | Có, $y$ là số | Giá nhà, doanh thu | Linear Regression, Random Forest, XGBoost |
| **Phân loại** (supervised) | Có, $y$ là lớp | Spam, bệnh/không bệnh | Logistic Regression, SVM, cây |
| **Gom cụm** (unsupervised) | Không | Phân nhóm khách hàng | K-Means, DBSCAN |
| **Giảm chiều** (unsupervised) | Không | Nén đặc trưng, trực quan hóa | PCA |
| **Học tăng cường** | Phần thưởng | Chơi game, robot | Nằm ngoài GĐ1 |

### Thuật ngữ cần thuộc

- **Đặc trưng (feature)** $x$: đầu vào. **Nhãn (label/target)** $y$: đầu ra cần dự đoán.
- **Tham số (parameter)**: thứ mô hình tự học (trọng số $w$). **Siêu tham số (hyperparameter)**: thứ bạn đặt trước (learning rate, độ sâu cây).
- **Hàm mất mát (loss)**: đo mô hình sai bao nhiêu trên một mẫu. **Hàm chi phí (cost)**: trung bình loss trên cả tập.
- **Huấn luyện (train)**: tìm tham số làm cost nhỏ nhất. **Suy luận (inference)**: dùng mô hình đã học để dự đoán.

---

## 2. Quy trình một dự án ML

```
Hiểu bài toán → Thu thập dữ liệu → EDA → Chia dữ liệu → Tiền xử lý
   → Baseline → Thử mô hình → Đánh giá → Tuning → Đánh giá cuối trên test → Triển khai/giám sát
```

Nguyên tắc xuyên suốt:

1. **Bắt đầu bằng câu hỏi, không phải mô hình.** Dự đoán cái gì? Sai thì tốn kém kiểu nào (bỏ sót bệnh hay báo nhầm)? Metric nào phản ánh đúng điều đó?
2. **Luôn có baseline.** Hồi quy: dự đoán trung bình. Phân loại: dự đoán lớp phổ biến nhất. Mô hình phức tạp mà không thắng baseline thì vô nghĩa.
3. **Chia dữ liệu trước khi làm bất cứ điều gì học từ dữ liệu** (điền giá trị thiếu, chuẩn hóa, chọn đặc trưng). Nếu không sẽ bị rò rỉ dữ liệu.
4. **Tập test chỉ dùng một lần ở cuối.** Chọn mô hình dựa trên validation / cross-validation.
5. **Đơn giản trước, phức tạp sau.** Thêm độ phức tạp khi có bằng chứng nó giúp ích.

---

## 3. Chuẩn bị dữ liệu

### 3.1. Khám phá dữ liệu (EDA)

```python
import pandas as pd

df = pd.read_csv("data/housing.csv")
df.shape
df.info()                      # kiểu dữ liệu, số giá trị thiếu
df.describe()                  # thống kê số
df.isna().mean().sort_values(ascending=False)   # tỉ lệ thiếu theo cột
df["price"].hist(bins=50)      # phân phối nhãn
df.corr(numeric_only=True)["price"].sort_values()   # tương quan với nhãn
```

Cần trả lời được: có bao nhiêu mẫu và cột? Cột nào thiếu nhiều? Nhãn lệch hay cân bằng? Có giá trị bất thường (outlier) không? Có cột nào "biết trước tương lai" không (xem rò rỉ ở dưới)?

### 3.2. Chia dữ liệu

| Tập | Mục đích | Tỉ lệ thường dùng |
|---|---|---|
| Train | Học tham số | 60–80% |
| Validation | Chọn mô hình, tuning | 10–20% (hoặc dùng cross-validation) |
| Test | Đánh giá cuối cùng, dùng một lần | 10–20% |

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns="price")
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Phân loại: thêm stratify=y để giữ tỉ lệ lớp
```

Với **dữ liệu chuỗi thời gian**, không xáo trộn: tập train phải đứng trước tập test theo thời gian.

### 3.3. Giá trị thiếu

- Xóa hàng: chỉ khi thiếu ít và ngẫu nhiên.
- Điền số: median (bền với outlier hơn mean).
- Điền phân loại: giá trị phổ biến nhất hoặc một nhóm riêng `"unknown"`.
- Thêm cột cờ `is_missing` khi việc thiếu tự nó mang thông tin.
- Điền bằng thống kê **tính trên tập train**, rồi áp dụng sang validation/test.

### 3.4. Mã hóa biến phân loại

| Cách | Khi nào |
|---|---|
| One-hot | Ít giá trị, không có thứ tự (màu, thành phố) |
| Ordinal | Có thứ tự (nhỏ < vừa < lớn) |
| Target / frequency encoding | Rất nhiều giá trị (cẩn thận rò rỉ nhãn, phải tính trong cross-validation) |

Mô hình dạng cây (Random Forest, LightGBM) ít nhạy với cách mã hóa hơn mô hình tuyến tính.

### 3.5. Chuẩn hóa đặc trưng

- **StandardScaler:** $x' = (x - \mu)/\sigma$. Cần cho: hồi quy tuyến tính/logistic có regularization, SVM, KNN, PCA, K-Means, mạng neural.
- **MinMaxScaler:** đưa về $[0,1]$. Nhạy với outlier.
- **Không cần** cho mô hình dạng cây.

### 3.6. Rò rỉ dữ liệu (data leakage)

Rò rỉ là khi thông tin từ ngoài tập train lọt vào lúc huấn luyện, khiến điểm đánh giá đẹp giả tạo. Hai dạng hay gặp:

- **Rò rỉ nhãn:** có đặc trưng mà ở thời điểm dự đoán thật bạn không có. Ví dụ dự đoán "khách có hủy dịch vụ không" nhưng dùng cột "ngày hủy".
- **Rò rỉ qua tiền xử lý:** fit scaler/imputer trên toàn bộ dữ liệu rồi mới chia. Cách tránh: đóng gói tiền xử lý trong `Pipeline` (xem mục 12).

Dấu hiệu nghi ngờ: điểm gần hoàn hảo trên bài toán khó. Khi đó đừng ăn mừng, hãy đi tìm rò rỉ.

---

## 4. Hồi quy tuyến tính

### Mô hình

$$\hat{y} = w_1 x_1 + \dots + w_d x_d + b = \mathbf{w}^\top \mathbf{x} + b$$

### Hàm chi phí: MSE

$$J(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

### Gradient descent

Cập nhật ngược hướng gradient:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \,\frac{\partial J}{\partial \mathbf{w}}, \qquad \frac{\partial J}{\partial \mathbf{w}} = \frac{2}{n} X^\top(\hat{\mathbf{y}} - \mathbf{y})$$

$\eta$ là **learning rate**: quá lớn thì loss dao động hoặc phân kỳ, quá nhỏ thì học rất chậm. Luôn vẽ đường loss theo vòng lặp để kiểm tra.

### Tự cài đặt bằng NumPy

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []
        for _ in range(self.n_iters):
            err = X @ self.w + self.b - y
            self.w -= self.lr * (2 / n) * (X.T @ err)
            self.b -= self.lr * (2 / n) * err.sum()
            self.losses.append((err ** 2).mean())
        return self

    def predict(self, X):
        return X @ self.w + self.b
```

Đặc trưng phải được chuẩn hóa trước, nếu không gradient descent hội tụ rất chậm.

### Nghiệm đóng (normal equation)

Với dữ liệu nhỏ có thể giải thẳng, không cần lặp:

$$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$$

Trong code dùng `np.linalg.lstsq` thay vì tự nghịch đảo, vì ổn định số hơn.

```python
Xb = np.c_[np.ones(len(X)), X]                 # thêm cột 1 cho bias
theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)  # theta[0] là b
```

### Dùng scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression().fit(X_train, y_train)
rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
```

### Giả định và hạn chế

Quan hệ tuyến tính, sai số độc lập có phương sai không đổi, ít đa cộng tuyến giữa các đặc trưng. Nhạy với outlier. Muốn bắt quan hệ phi tuyến: thêm đặc trưng đa thức, hoặc chuyển sang mô hình cây.

---

## 5. Hồi quy logistic

Dù tên là "hồi quy", đây là mô hình **phân loại**. Nó dự đoán xác suất thuộc lớp 1.

### Mô hình

$$z = \mathbf{w}^\top \mathbf{x} + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Sigmoid ép $z$ về khoảng $(0,1)$. Quyết định: lớp 1 nếu $p \ge$ ngưỡng (mặc định 0.5, nhưng ngưỡng nên chỉnh theo bài toán).

### Hàm mất mát: cross-entropy

$$J = -\frac{1}{n}\sum_i \big[\, y_i \log p_i + (1-y_i)\log(1-p_i) \,\big]$$

Gradient có dạng gọn y như hồi quy tuyến tính:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n} X^\top (\mathbf{p} - \mathbf{y})$$

### Tự cài đặt

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip để tránh tràn số

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y)) / n
            self.b -= self.lr * (p - y).mean()
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### Nhiều lớp

Dùng **softmax** thay sigmoid, hoặc huấn luyện nhiều bộ phân loại one-vs-rest. scikit-learn xử lý sẵn.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

### Ưu điểm

Nhanh, ít tham số, xác suất đầu ra khá hiệu chỉnh, hệ số dễ diễn giải. Luôn là baseline phân loại tốt.

---

## 6. Đánh giá mô hình

### 6.1. Metric cho hồi quy

| Metric | Công thức | Ghi chú |
|---|---|---|
| MAE | $\frac1n\sum \lvert y-\hat y\rvert$ | Cùng đơn vị với $y$, bền với outlier |
| MSE / RMSE | $\frac1n\sum (y-\hat y)^2$, rồi căn bậc hai | Phạt nặng sai số lớn |
| R² | $1 - \frac{\sum (y-\hat y)^2}{\sum (y-\bar y)^2}$ | 1 là hoàn hảo, 0 bằng dự đoán trung bình, có thể âm |

### 6.2. Metric cho phân loại

Từ ma trận nhầm lẫn (TP, FP, FN, TN):

| Metric | Công thức | Trả lời câu hỏi |
|---|---|---|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | Đúng bao nhiêu phần trăm? (lừa người khi dữ liệu lệch) |
| Precision | $\frac{TP}{TP+FP}$ | Trong số dự đoán dương, bao nhiêu đúng? |
| Recall | $\frac{TP}{TP+FN}$ | Trong số thật sự dương, bắt được bao nhiêu? |
| F1 | $\frac{2PR}{P+R}$ | Trung bình điều hòa của precision và recall |
| ROC-AUC | Diện tích dưới đường ROC | Khả năng xếp mẫu dương cao hơn mẫu âm, không phụ thuộc ngưỡng |
| PR-AUC | Diện tích dưới đường precision-recall | Tốt hơn ROC-AUC khi lớp dương hiếm |

**Chọn metric theo hậu quả:**
- Bỏ sót ca bệnh rất tệ → ưu tiên **recall**.
- Báo nhầm spam làm mất thư quan trọng tệ → ưu tiên **precision**.
- Dữ liệu 99% âm, 1% dương → accuracy 99% khi đoán toàn "âm" là vô nghĩa. Dùng PR-AUC, F1, recall.

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)

y_pred = (proba >= 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC :", average_precision_score(y_test, proba))
```

### 6.3. Cross-validation

Một lần chia train/validation có thể may rủi. **K-fold CV** chia train thành $K$ phần, lần lượt giữ một phần làm validation, lấy trung bình điểm.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
print(scores.mean(), "±", scores.std())
```

Hãy báo cáo cả độ lệch chuẩn. Hai mô hình chênh nhau ít hơn độ lệch chuẩn thì chưa thể kết luận cái nào tốt hơn.

Biến thể: `StratifiedKFold` (giữ tỉ lệ lớp), `GroupKFold` (các mẫu cùng nhóm không bị tách đôi, ví dụ cùng một bệnh nhân), `TimeSeriesSplit` (dữ liệu thời gian).

---

## 7. Overfitting, bias-variance, regularization

### Hai kiểu thất bại

- **Underfitting:** mô hình quá đơn giản, sai cả trên train lẫn validation. Cách xử lý: mô hình mạnh hơn, thêm đặc trưng, giảm regularization.
- **Overfitting:** mô hình học thuộc cả nhiễu, tốt trên train nhưng tệ trên validation. Cách xử lý: thêm dữ liệu, mô hình đơn giản hơn, regularization, dừng sớm, loại đặc trưng nhiễu.

| | Lỗi train | Lỗi validation |
|---|---|---|
| Underfit | Cao | Cao |
| Vừa | Thấp | Thấp, sát lỗi train |
| Overfit | Rất thấp | Cao, cách xa lỗi train |

### Bias-variance

Lỗi kỳ vọng trên dữ liệu mới phân tách thành:

$$\text{Lỗi} = \underbrace{\text{Bias}^2}_{\text{sai do quá đơn giản}} + \underbrace{\text{Variance}}_{\text{nhạy với dữ liệu train}} + \underbrace{\text{Nhiễu}}_{\text{không giảm được}}$$

Mô hình phức tạp hơn thì bias giảm, variance tăng. Mục tiêu là điểm cân bằng.

### Chẩn đoán bằng learning curve

```python
from sklearn.model_selection import learning_curve
import numpy as np

sizes, train_sc, val_sc = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="neg_root_mean_squared_error",
)
# Vẽ -train_sc.mean(1) và -val_sc.mean(1) theo sizes
```

- Hai đường gần nhau và cùng cao → underfit, thêm dữ liệu không giúp.
- Khoảng cách lớn giữa hai đường → overfit, thêm dữ liệu sẽ giúp.

### Regularization

Thêm một khoảng phạt vào hàm chi phí để giữ trọng số nhỏ:

| Loại | Hàm chi phí | Tác dụng |
|---|---|---|
| **Ridge (L2)** | $J + \alpha \sum w_j^2$ | Co trọng số về gần 0, xử lý đa cộng tuyến |
| **Lasso (L1)** | $J + \alpha \sum \lvert w_j\rvert$ | Đẩy nhiều trọng số về đúng 0 (chọn đặc trưng tự động) |
| **Elastic Net** | Kết hợp L1 và L2 | Cân bằng hai tác dụng trên |

```python
from sklearn.linear_model import RidgeCV, LassoCV
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X_train_scaled, y_train)
```

$\alpha$ càng lớn thì regularization càng mạnh. Với `LogisticRegression` của scikit-learn, tham số là `C` và **nghịch đảo** của $\alpha$ (C nhỏ nghĩa là regularization mạnh).

---

## 8. Cây quyết định và các mô hình tổ hợp

Đây là nhóm mô hình quan trọng nhất cho dữ liệu dạng bảng. Trong thực tế, gradient boosting thường là lựa chọn mạnh nhất cho loại dữ liệu này.

### 8.1. Cây quyết định

Chia dữ liệu liên tiếp bằng các câu hỏi dạng `đặc trưng ≤ ngưỡng?`, chọn chỗ chia làm các nhóm con "thuần" nhất.

Độ không thuần của nút (phân loại):

$$\text{Gini} = 1 - \sum_k p_k^2, \qquad \text{Entropy} = -\sum_k p_k \log_2 p_k$$

Chia ở chỗ nào giảm độ không thuần nhiều nhất (information gain). Với hồi quy, tiêu chí là giảm phương sai.

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
```

- **Ưu:** dễ diễn giải, không cần chuẩn hóa, bắt được quan hệ phi tuyến và tương tác.
- **Nhược:** một cây không giới hạn sẽ overfit nặng, rất nhạy với thay đổi nhỏ của dữ liệu.
- Kiểm soát bằng `max_depth`, `min_samples_leaf`, `min_samples_split`.

### 8.2. Random Forest (bagging)

Huấn luyện nhiều cây, mỗi cây trên **một mẫu bootstrap** và chỉ xét **một tập con đặc trưng ngẫu nhiên** ở mỗi lần chia, rồi lấy trung bình (hồi quy) hoặc bỏ phiếu (phân loại). Việc lấy trung bình nhiều cây ít tương quan làm **giảm variance**.

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                           min_samples_leaf=3, n_jobs=-1, random_state=42)
```

Ít cần tuning, khó overfit khi tăng số cây, cho sẵn feature importance và ước lượng OOB. Nhược điểm: mô hình nặng, dự đoán chậm hơn một mô hình đơn.

### 8.3. Gradient Boosting

Huấn luyện các cây **tuần tự**. Mỗi cây mới học để sửa phần sai (residual) của tổng các cây trước, nói đúng hơn là học theo hướng gradient âm của loss:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Boosting chủ yếu **giảm bias**, và dễ overfit hơn Random Forest nếu không kiểm soát.

Thư viện thực tế: **XGBoost**, **LightGBM**, **CatBoost** (xử lý biến phân loại tốt).

```python
from lightgbm import LGBMRegressor

gbm = LGBMRegressor(
    n_estimators=2000, learning_rate=0.03,
    num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)
```

Siêu tham số quan trọng:

| Tham số | Ý nghĩa | Nếu overfit thì |
|---|---|---|
| `learning_rate` | Bước học của mỗi cây | Giảm (và tăng số cây) |
| `n_estimators` | Số cây | Dùng early stopping để tự chọn |
| `num_leaves` / `max_depth` | Độ phức tạp mỗi cây | Giảm |
| `min_child_samples` | Số mẫu tối thiểu mỗi lá | Tăng |
| `subsample`, `colsample_bytree` | Lấy mẫu hàng/cột mỗi cây | Giảm xuống 0.6–0.8 |
| `reg_alpha`, `reg_lambda` | Regularization L1/L2 | Tăng |

### 8.4. So sánh nhanh

| | Cây đơn | Random Forest | Gradient Boosting |
|---|---|---|---|
| Giảm | – | Variance | Bias |
| Tuning | Ít | Ít | Nhiều hơn |
| Rủi ro overfit | Cao | Thấp | Trung bình |
| Độ chính xác điển hình | Thấp | Cao | Thường cao nhất |

---

## 9. SVM, KNN, Naive Bayes

### SVM (Support Vector Machine)

Tìm siêu phẳng chia hai lớp với **lề (margin) lớn nhất**. Chỉ các điểm sát biên (support vector) quyết định biên. Với dữ liệu không tách tuyến tính, dùng **kernel** (RBF, đa thức) để ánh xạ ngầm lên không gian chiều cao hơn.

- `C`: càng lớn càng cố phân loại đúng từng điểm (dễ overfit); `C` nhỏ cho lề rộng hơn, chấp nhận vài lỗi.
- `gamma` (kernel RBF): càng lớn thì biên càng uốn lượn (dễ overfit).
- **Bắt buộc chuẩn hóa đặc trưng.** Chậm với dữ liệu lớn (hơn vài chục nghìn mẫu).

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
```

### KNN (K-Nearest Neighbors)

Dự đoán theo $k$ láng giềng gần nhất (lớp phổ biến nhất, hoặc trung bình giá trị). Không có bước huấn luyện, chi phí dồn vào lúc dự đoán.

- $k$ nhỏ → variance cao; $k$ lớn → bias cao. Chọn $k$ bằng cross-validation.
- **Bắt buộc chuẩn hóa**, vì dựa trên khoảng cách. Kém khi nhiều chiều ("lời nguyền số chiều").

### Naive Bayes

Dựa trên định lý Bayes với giả định "ngây thơ" rằng các đặc trưng độc lập có điều kiện theo lớp:

$$P(c \mid x_1,\dots,x_d) \propto P(c)\prod_j P(x_j \mid c)$$

Rất nhanh, hoạt động tốt với ít dữ liệu và văn bản (đếm từ). Giả định độc lập thường sai, nên xác suất đầu ra kém tin cậy dù xếp hạng lớp vẫn ổn. Biến thể: `GaussianNB`, `MultinomialNB`, `BernoulliNB`.

### Khi nào dùng cái gì (dữ liệu dạng bảng)

1. Logistic / Linear Regression làm baseline.
2. Random Forest để có kết quả mạnh nhanh.
3. LightGBM / XGBoost để tối đa độ chính xác.
4. SVM, KNN, Naive Bayes khi dữ liệu nhỏ hoặc có lý do đặc thù (văn bản ngắn, ranh giới hình học).

---

## 10. Học không giám sát: Clustering và PCA

### K-Means

1. Chọn $K$ tâm cụm ngẫu nhiên.
2. Gán mỗi điểm vào tâm gần nhất.
3. Cập nhật mỗi tâm thành trung bình các điểm của cụm.
4. Lặp 2–3 đến khi hội tụ.

Tối thiểu hóa **inertia** (tổng bình phương khoảng cách tới tâm). Nhược điểm: phải chọn $K$ trước, nhạy với khởi tạo (dùng `k-means++`, `n_init`), giả định cụm hình cầu và kích thước gần nhau, cần chuẩn hóa.

Chọn $K$: **elbow** (vẽ inertia theo $K$), **silhouette score** (cao hơn là tốt hơn, trong $[-1, 1]$).

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(k, km.inertia_, silhouette_score(X_scaled, km.labels_))
```

### DBSCAN

Gom các vùng dày đặc, tự đánh dấu điểm nhiễu (nhãn `-1`), không cần chọn số cụm, bắt được cụm hình dạng bất kỳ. Hai tham số: `eps` (bán kính láng giềng) và `min_samples`. Kém khi các cụm có mật độ rất khác nhau.

### PCA (giảm chiều)

Tìm các hướng (thành phần chính) mà dữ liệu biến thiên nhiều nhất. Về mặt toán là các vector riêng của ma trận hiệp phương sai, sắp theo giá trị riêng giảm dần.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)        # giữ 95% phương sai
X_reduced = pca.fit_transform(X_scaled)
print(pca.n_components_, pca.explained_variance_ratio_.cumsum())
```

Dùng để: trực quan hóa (2–3 chiều), nén, khử tương quan, tăng tốc mô hình. **Phải chuẩn hóa trước.** Các thành phần chính khó diễn giải hơn đặc trưng gốc. Đừng dùng PCA để "chọn đặc trưng quan trọng cho nhãn", vì nó không nhìn thấy nhãn.

---

## 11. Feature engineering và dữ liệu mất cân bằng

### Feature engineering

Đặc trưng tốt thường quan trọng hơn thuật toán. Một số kỹ thuật:

- **Biến đổi số:** log cho phân phối lệch phải (giá, thu nhập), phép chia/tỉ lệ (giá trên diện tích), đa thức.
- **Rời rạc hóa:** chia khoảng (nhóm tuổi) khi quan hệ không đơn điệu.
- **Tương tác:** tích hoặc hiệu của hai đặc trưng (mô hình tuyến tính cần, mô hình cây tự tìm được phần nào).
- **Ngày giờ:** tách năm, tháng, thứ trong tuần, giờ; dùng sin/cos cho tính tuần hoàn.
- **Tổng hợp theo nhóm:** trung bình giá theo khu vực, số lần mua theo khách (tính trong cross-validation để tránh rò rỉ).
- **Văn bản:** TF-IDF, đếm từ (nền tảng trước khi học embedding ở GĐ3).

Sau mỗi đặc trưng mới, kiểm chứng bằng cross-validation. Không cải thiện thì bỏ.

### Dữ liệu mất cân bằng

Khi một lớp hiếm (gian lận, bệnh hiếm):

1. **Đổi metric:** dùng PR-AUC, F1, recall thay cho accuracy.
2. **Class weight:** `class_weight="balanced"` làm lỗi trên lớp hiếm bị phạt nặng hơn. Cách đơn giản, nên thử đầu tiên.
3. **Chỉnh ngưỡng quyết định** thay vì cố định 0.5, dựa trên đường precision-recall.
4. **Lấy mẫu lại:** undersample lớp nhiều, oversample lớp hiếm (SMOTE tạo mẫu tổng hợp).

Lưu ý quan trọng: chỉ lấy mẫu lại **trên tập train trong từng fold**. Nếu lấy mẫu lại trước khi chia, bản sao của cùng một mẫu rơi vào cả train lẫn validation và gây rò rỉ. Dùng `imblearn.pipeline.Pipeline` để làm đúng.

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val, proba_val)
# Chọn ngưỡng đạt recall mong muốn với precision cao nhất
```

---

## 12. Pipeline và tuning siêu tham số

### Pipeline đầy đủ

`Pipeline` đảm bảo mọi bước tiền xử lý chỉ `fit` trên dữ liệu train của từng fold, nên loại bỏ rò rỉ.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

num_cols = X_train.select_dtypes("number").columns
cat_cols = X_train.select_dtypes(exclude="number").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)),
])

scores = cross_val_score(model, X_train, y_train, cv=5,
                         scoring="neg_root_mean_squared_error")
print("RMSE:", -scores.mean(), "±", scores.std())
```

### Tuning

Thứ tự ưu tiên: **tối ưu dữ liệu và đặc trưng trước, tuning sau.** Tuning thường chỉ cải thiện vài phần trăm.

| Cách | Ghi chú |
|---|---|
| Grid search | Thử mọi tổ hợp, tốn kém khi nhiều tham số |
| Random search | Thường hiệu quả hơn grid với cùng ngân sách |
| **Optuna** | Tìm kiếm thông minh (Bayesian), có pruning, nên dùng |

```python
import optuna
from lightgbm import LGBMRegressor

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMRegressor(**params, subsample_freq=1,
                                random_state=42, verbose=-1)),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5,
                             scoring="neg_root_mean_squared_error")
    return -scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print(study.best_params, study.best_value)
```

Sau khi chốt cấu hình, fit lại trên toàn bộ tập train và **đánh giá một lần** trên tập test.

---

## 13. Thiết kế thí nghiệm và giải thích mô hình

### Thí nghiệm có kỷ luật

1. **Baseline trước.** Ghi điểm của baseline đơn giản.
2. **Mỗi lần chỉ đổi một thứ,** để biết thay đổi nào gây ra khác biệt.
3. **Ablation:** bỏ từng thành phần (một nhóm đặc trưng, một bước tiền xử lý) rồi đo lại, xem nó thực sự đóng góp bao nhiêu.
4. **Cố định seed** và dùng cùng các fold khi so sánh.
5. **Báo cáo trung bình ± độ lệch chuẩn** qua các fold. Chênh nhỏ hơn độ lệch chuẩn thì xem như hòa.
6. **Ghi lại mọi thí nghiệm** (tham số, điểm, phiên bản dữ liệu) vào bảng hoặc công cụ như MLflow, để tái lập được.

Mẫu bảng kết quả:

| Mô hình | Đặc trưng | RMSE CV (±std) | Ghi chú |
|---|---|---|---|
| Baseline (trung bình) | – | 78 000 ± 2 100 | Mốc tham chiếu |
| Linear Regression | gốc | 52 000 ± 1 800 | |
| Random Forest | gốc | 31 500 ± 1 200 | |
| LightGBM | gốc + tỉ lệ | 28 900 ± 1 100 | Đặc trưng mới giúp ích |

### Giải thích mô hình

- **Hệ số** (mô hình tuyến tính): dấu và độ lớn cho biết ảnh hưởng, chỉ so sánh được khi đã chuẩn hóa.
- **Feature importance** của cây: nhanh nhưng thiên vị đặc trưng có nhiều giá trị khác nhau.
- **Permutation importance:** xáo trộn một cột và xem điểm giảm bao nhiêu, đáng tin hơn, dùng trên tập validation.
- **SHAP:** gán đóng góp của từng đặc trưng cho từng dự đoán, vừa giải thích cục bộ vừa tổng thể.

```python
import shap
from sklearn.inspection import permutation_importance

# Mô hình cây
fitted = model.fit(X_train, y_train)
Xt = fitted.named_steps["preprocess"].transform(X_val)
explainer = shap.TreeExplainer(fitted.named_steps["model"])
shap_values = explainer.shap_values(Xt)
shap.summary_plot(shap_values, Xt)

# Mô hình bất kỳ
r = permutation_importance(fitted, X_val, y_val, n_repeats=10, random_state=42)
```

### Thiên lệch và công bằng

Mô hình chỉ học những gì dữ liệu chứa. Nếu dữ liệu phản ánh định kiến hoặc thiếu đại diện một nhóm, mô hình sẽ lặp lại hoặc khuếch đại điều đó. Thói quen tối thiểu: **tách điểm đánh giá theo nhóm con** (giới tính, vùng miền...) để phát hiện chênh lệch, và tự hỏi đặc trưng nào là đại diện gián tiếp cho thuộc tính nhạy cảm.

---

## 14. Lỗi thường gặp

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| Fit scaler/imputer trên toàn bộ dữ liệu | Rò rỉ, điểm đẹp giả | Dùng `Pipeline` |
| Chọn mô hình dựa trên điểm test | Test không còn khách quan | Chọn bằng CV, test chỉ dùng cuối |
| Dùng accuracy cho dữ liệu lệch | Tưởng tốt nhưng vô dụng | PR-AUC, F1, recall |
| Không có baseline | Không biết mô hình có giá trị không | Luôn đặt baseline trước |
| Xáo trộn dữ liệu thời gian | Nhìn thấy tương lai | `TimeSeriesSplit` |
| Oversample trước khi chia | Rò rỉ | Lấy mẫu lại trong từng fold |
| Quên chuẩn hóa cho SVM/KNN/PCA/regularization | Kết quả sai lệch | Đưa scaler vào pipeline |
| Điều chỉnh quá nhiều trên một tập validation | Overfit vào validation | Dùng CV, giới hạn số lần thử |
| Đặc trưng dùng thông tin không có lúc dự đoán thật | Mô hình vô dụng khi triển khai | Kiểm tra từng cột: "lúc dự đoán thật mình có nó không?" |
| Không đặt `random_state` | Không tái lập được | Cố định seed |
| Chỉ nhìn một con số | Bỏ sót vấn đề | Xem thêm learning curve, phân tích sai sót |

Thói quen tốt nhất sau khi có mô hình: **đọc những mẫu mô hình dự đoán sai nhiều nhất.** Chúng thường chỉ ra lỗi dữ liệu hoặc đặc trưng còn thiếu.

---

## 15. Kế hoạch học, bài tập, dự án

### Kế hoạch 6–8 tuần

| Tuần | Nội dung | Làm gì |
|---|---|---|
| 1 | Mục 1–3: quy trình, dữ liệu | EDA một dataset (Titanic hoặc California Housing) |
| 2 | Mục 4: hồi quy tuyến tính | Tự cài `LinearRegressionGD`, so với `sklearn` |
| 3 | Mục 5–6: logistic, metric | Tự cài `LogisticRegressionGD`, vẽ ROC và PR |
| 4 | Mục 7: overfitting, regularization | Vẽ learning curve, thử Ridge/Lasso |
| 5 | Mục 8: cây, RF, boosting | So sánh cây đơn, RF, LightGBM trên cùng dữ liệu |
| 6 | Mục 9–10: SVM/KNN/NB, clustering, PCA | K-Means + PCA trên dữ liệu khách hàng |
| 7 | Mục 11–12: feature engineering, pipeline, Optuna | Dựng pipeline hoàn chỉnh cho dự án |
| 8 | Mục 13–14: thí nghiệm, giải thích | Hoàn thiện dự án, viết báo cáo, ôn lại |

Nếu thời gian eo hẹp, gộp tuần 6 vào tuần 5 và tuần 8 vào tuần 7.

### Bài tập

**Tự cài đặt (không dùng sklearn cho phần lõi):**
1. Hồi quy tuyến tính bằng gradient descent, kiểm tra kết quả khớp với `LinearRegression`.
2. Hồi quy logistic, tính loss và accuracy từng vòng lặp, vẽ đường loss.
3. K-Means bằng NumPy (khởi tạo, gán cụm, cập nhật tâm).
4. KNN bằng NumPy với khoảng cách Euclid.
5. Một cây quyết định đơn giản tính Gini để tìm điểm chia tốt nhất.

**Thực hành với thư viện:**
1. Trên cùng một dataset, so sánh Linear/Logistic, Random Forest, LightGBM bằng 5-fold CV.
2. Thử `class_weight`, SMOTE và chỉnh ngưỡng trên dữ liệu mất cân bằng (credit card fraud). So sánh PR-AUC.
3. Cố tình tạo rò rỉ (fit scaler trên toàn bộ dữ liệu, hoặc thêm cột nhãn) rồi xem điểm "đẹp" thế nào. Sau đó sửa lại.
4. Dùng SHAP giải thích mô hình tốt nhất và viết 5 câu nhận xét.

**Dataset gợi ý:** Titanic, California Housing, Adult Income, Ames Housing, Credit Card Fraud, Telco Customer Churn (Kaggle, UCI, `sklearn.datasets`).

### Dự án cuối giai đoạn: `projects/01-house-price-ml/`

Yêu cầu tối thiểu:
- [ ] EDA có biểu đồ và nhận xét
- [ ] Chia train/test đúng cách, không rò rỉ
- [ ] Baseline và ít nhất 3 mô hình so sánh bằng CV (có ± std)
- [ ] Pipeline dùng `ColumnTransformer`
- [ ] Feature engineering, có ablation chứng minh đặc trưng mới giúp ích
- [ ] Tuning bằng Optuna
- [ ] Đánh giá cuối trên tập test, **một lần**
- [ ] Giải thích mô hình (SHAP hoặc permutation importance) và phân tích các mẫu sai nhiều nhất
- [ ] README: bài toán, cách làm, kết quả, bài học

### Tự kiểm tra: bạn đã xong khi trả lời được

- [ ] Vì sao phải chuẩn hóa trước khi dùng KNN nhưng không cần cho Random Forest?
- [ ] Gradient descent hoạt động thế nào, và điều gì xảy ra khi learning rate quá lớn?
- [ ] Vì sao cross-entropy phù hợp hơn MSE cho phân loại?
- [ ] Khi nào accuracy gây hiểu lầm? Chọn metric nào thay thế?
- [ ] Overfitting là gì, phát hiện bằng cách nào, có ít nhất 4 cách xử lý?
- [ ] Random Forest giảm variance, boosting giảm bias: nghĩa là gì?
- [ ] Rò rỉ dữ liệu là gì? Cho 2 ví dụ và cách tránh.
- [ ] Vì sao tập test chỉ dùng một lần?
- [ ] L1 khác L2 thế nào về tác động lên trọng số?
- [ ] Tự cài đặt được hồi quy logistic từ đầu mà không xem code.

---

## 16. Tài liệu tham khảo

**Khóa học:**
- *Machine Learning Specialization* (Andrew Ng, Coursera): nền tảng trực quan, tốt cho người mới.
- *Introduction to Statistical Learning* (ISLP) và video đi kèm: lý thuyết rõ ràng, có bản Python.

**Sách:**
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron): thực hành, bám sát quy trình trong tài liệu này.
- *Pattern Recognition and Machine Learning* (Bishop): chiều sâu toán, đọc dần sau khi có nền.

**Thực hành:**
- Tài liệu và User Guide của scikit-learn (rất đáng đọc, giải thích kèm ví dụ).
- Kaggle Learn (Intro to ML, Intermediate ML, Feature Engineering).
- StatQuest (YouTube): giải thích khái niệm thống kê và ML bằng hình ảnh.

**Công cụ:** scikit-learn, LightGBM, XGBoost, Optuna, SHAP, imbalanced-learn, MLflow.
