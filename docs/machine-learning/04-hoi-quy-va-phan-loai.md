# Chương 4: Hồi quy và phân loại (mô hình tuyến tính)

Mô hình tuyến tính là nền tảng của ML: đơn giản, nhanh, dễ diễn giải, và là baseline tốt nhất. Hiểu kỹ chúng, bạn sẽ hiểu luôn nền của mạng neural.

## Mục lục

**Phần A: Hồi quy**
1. [Hồi quy tuyến tính](#1-hồi-quy-tuyến-tính)
2. [Chẩn đoán mô hình tuyến tính](#2-chẩn-đoán-mô-hình-tuyến-tính)
3. [Mở rộng sang quan hệ phi tuyến](#3-mở-rộng-sang-quan-hệ-phi-tuyến)
4. [Regularization: Ridge, Lasso, Elastic Net](#4-regularization-ridge-lasso-elastic-net)
5. [Hồi quy bền vững và các biến thể](#5-hồi-quy-bền-vững-và-các-biến-thể)

**Phần B: Phân loại**

6. [Hồi quy logistic](#6-hồi-quy-logistic)
7. [Phân loại đa lớp](#7-phân-loại-đa-lớp)
8. [Phân loại đa nhãn](#8-phân-loại-đa-nhãn)
9. [Ngưỡng quyết định và chi phí lỗi](#9-ngưỡng-quyết-định-và-chi-phí-lỗi)
10. [Hiệu chỉnh xác suất](#10-hiệu-chỉnh-xác-suất-calibration)

---

# Phần A: Hồi quy

## 1. Hồi quy tuyến tính

### 1.1. Mô hình

$$\hat y = w_1x_1 + \dots + w_dx_d + b = \mathbf{w}^\top\mathbf{x} + b$$

Dạng ma trận: thêm cột toàn 1 vào $X$ để gộp $b$ vào $\mathbf{w}$, khi đó $\hat{\mathbf y}=X\mathbf w$.

Ý nghĩa hệ số: $w_j$ là mức thay đổi của $\hat y$ khi $x_j$ tăng 1 đơn vị **với các đặc trưng khác giữ nguyên**.

### 1.2. Hàm chi phí

$$J(\mathbf w,b)=\frac1n\sum_{i=1}^n(\hat y_i-y_i)^2$$

Đây là MLE khi nhiễu có phân phối chuẩn (xem [Chương 2](02-toan-nen-tang-va-toi-uu.md)).

### 1.3. Hai cách giải

**a) Nghiệm đóng (normal equation).** Đặt gradient bằng 0:

$$\mathbf w=(X^\top X)^{-1}X^\top\mathbf y$$

Chính xác, không cần learning rate, nhưng tốn $O(d^3)$ và lỗi khi $X^\top X$ không khả nghịch (đặc trưng phụ thuộc tuyến tính). Trong code dùng `np.linalg.lstsq` thay vì tự nghịch đảo.

```python
import numpy as np

Xb = np.c_[np.ones(len(X)), X]                    # thêm cột 1 cho bias
theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)    # theta[0] là b
```

**b) Gradient descent.** Lặp lại cập nhật. Phù hợp khi $d$ hoặc $n$ rất lớn.

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

Đặc trưng phải được chuẩn hóa trước, nếu không hội tụ rất chậm. Kiểm tra bằng cách so kết quả với `sklearn.linear_model.LinearRegression`.

### 1.4. Dùng scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)
print(model.coef_, model.intercept_)
```

### 1.5. Giả định của hồi quy tuyến tính

| Giả định | Nếu vi phạm |
|---|---|
| Quan hệ tuyến tính giữa đặc trưng và nhãn | Dự đoán lệch hệ thống; thêm đặc trưng phi tuyến |
| Sai số độc lập với nhau | Ước lượng độ tin cậy sai (chuỗi thời gian có tự tương quan) |
| Phương sai sai số không đổi (homoscedasticity) | Sai số lớn ở một số vùng; thử log nhãn |
| Sai số xấp xỉ phân phối chuẩn | Ảnh hưởng khoảng tin cậy, ít ảnh hưởng dự đoán điểm |
| Không đa cộng tuyến mạnh | Hệ số mất ổn định, khó diễn giải |

Với mục tiêu **dự đoán**, mô hình vẫn dùng tốt khi các giả định chỉ vi phạm nhẹ. Với mục tiêu **suy diễn** (kết luận về ảnh hưởng của từng biến), giả định quan trọng hơn nhiều; hãy dùng `statsmodels` để có p-value và khoảng tin cậy.

---

## 2. Chẩn đoán mô hình tuyến tính

### 2.1. Biểu đồ phần dư

Phần dư $e_i=y_i-\hat y_i$. Vẽ phần dư theo giá trị dự đoán:

- Phân tán ngẫu nhiên quanh 0: tốt.
- Hình cong (chữ U): quan hệ phi tuyến bị bỏ sót.
- Hình phễu (rộng dần): phương sai không đổi bị vi phạm.
- Có điểm xa hẳn: outlier hoặc điểm có ảnh hưởng lớn.

```python
import matplotlib.pyplot as plt

resid = y_test - pred
plt.scatter(pred, resid, alpha=0.4)
plt.axhline(0, color="k")
plt.xlabel("Dự đoán"); plt.ylabel("Phần dư")
```

### 2.2. Đa cộng tuyến

Khi hai đặc trưng gần như là bản sao của nhau (diện tích m² và diện tích ft²), mô hình không phân biệt được ảnh hưởng của từng cái: hệ số lớn, trái dấu, thay đổi mạnh khi dữ liệu đổi nhẹ. Dự đoán vẫn ổn nhưng diễn giải sai.

Đo bằng **VIF** (variance inflation factor): VIF > 5–10 là đáng lo.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif = pd.Series(
    [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])],
    index=X.columns,
)
```

Cách xử lý: bỏ bớt đặc trưng trùng, gộp lại, dùng Ridge hoặc PCA.

### 2.3. Điểm có ảnh hưởng

Một vài điểm có thể kéo cả đường hồi quy về phía mình (đòn bẩy cao). Kiểm tra bằng khoảng cách Cook, hoặc đơn giản huấn luyện lại sau khi bỏ chúng xem hệ số thay đổi bao nhiêu.

---

## 3. Mở rộng sang quan hệ phi tuyến

"Tuyến tính" nghĩa là tuyến tính **theo tham số**, không nhất thiết theo $x$. Có thể biến đổi $x$ rồi vẫn dùng hồi quy tuyến tính.

### 3.1. Hồi quy đa thức

$$\hat y = w_0 + w_1x + w_2x^2 + \dots + w_px^p$$

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

poly = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0),
)
```

Bậc càng cao thì càng dễ overfit (đường uốn lượn theo từng điểm nhiễu). Kết hợp với regularization và chọn bậc bằng cross-validation. Số đặc trưng tăng rất nhanh theo bậc khi có nhiều biến.

### 3.2. Spline và hàm cơ sở

Thay vì một đa thức toàn cục, ghép nhiều đoạn đa thức nhỏ nối trơn tại các nút. Ổn định hơn đa thức bậc cao.

```python
from sklearn.preprocessing import SplineTransformer
spline = make_pipeline(SplineTransformer(degree=3, n_knots=6), Ridge(alpha=1.0))
```

### 3.3. Biến đổi nhãn

Nhãn lệch phải (giá) thường nên được biến đổi log, vì sai số trở thành sai số **tương đối**. Dùng `TransformedTargetRegressor` để tự biến đổi ngược khi dự đoán.

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(
    regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1
)
```

Khi quan hệ phức tạp và có nhiều tương tác, chuyển sang cây và boosting ([Chương 5](05-cay-va-ensemble.md)) thường đỡ công hơn.

---

## 4. Regularization: Ridge, Lasso, Elastic Net

### 4.1. Vì sao cần

Khi nhiều đặc trưng, ít dữ liệu, hoặc đặc trưng tương quan, hệ số có thể phình to và mô hình overfit. Regularization **phạt độ lớn của trọng số** để giữ mô hình đơn giản.

### 4.2. Ba biến thể

| | Hàm chi phí | Nghiệm | Tác dụng |
|---|---|---|---|
| **Ridge (L2)** | $\text{MSE}+\alpha\sum w_j^2$ | $\mathbf w=(X^\top X+\alpha I)^{-1}X^\top\mathbf y$ | Co đều các trọng số về gần 0, không về đúng 0. Ổn định khi đa cộng tuyến |
| **Lasso (L1)** | $\text{MSE}+\alpha\sum\lvert w_j\rvert$ | Không có dạng đóng | Đẩy nhiều trọng số về **đúng 0**: chọn đặc trưng tự động, mô hình thưa |
| **Elastic Net** | $\text{MSE}+\alpha\big[\rho\sum\lvert w_j\rvert+\tfrac{1-\rho}{2}\sum w_j^2\big]$ | Không có dạng đóng | Kết hợp: thưa nhưng ổn định với nhóm đặc trưng tương quan |

**Vì sao L1 cho nghiệm thưa còn L2 thì không?** Về hình học, miền ràng buộc của L1 là hình thoi có các đỉnh nằm trên trục, nên đường đồng mức của MSE dễ chạm đúng đỉnh (một số trọng số bằng 0). Miền của L2 là hình tròn, không có đỉnh nên trọng số nhỏ nhưng khác 0. Về xác suất: prior Laplace (L1) nhọn quanh 0, prior Gauss (L2) thì không.

### 4.3. Chọn $\alpha$

$\alpha$ lớn: mô hình đơn giản, bias cao (underfit). $\alpha=0$: không regularization (có thể overfit). Chọn bằng cross-validation.

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ridge = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42))
enet = make_pipeline(StandardScaler(),
                     ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, random_state=42))
```

**Bắt buộc chuẩn hóa đặc trưng trước**, vì khoản phạt nhạy với thang đo. Không phạt hệ số chặn $b$.

### 4.4. Khi nào dùng cái nào

- Mặc định: **Ridge**.
- Muốn chọn đặc trưng, tin rằng chỉ một ít đặc trưng thực sự quan trọng: **Lasso**.
- Nhiều đặc trưng tương quan theo nhóm: **Elastic Net** (Lasso thường chọn bừa một đại diện của nhóm).

Với `LogisticRegression` của scikit-learn, tham số là `C` và là **nghịch đảo** của $\alpha$ (`C` nhỏ nghĩa là regularization mạnh).

---

## 5. Hồi quy bền vững và các biến thể

| Phương pháp | Dùng khi | Công cụ |
|---|---|---|
| **Hồi quy Huber** | Có outlier ở nhãn; loss bình phương khi sai nhỏ, tuyến tính khi sai lớn | `HuberRegressor` |
| **RANSAC** | Nhiều outlier; fit trên tập con "sạch" ngẫu nhiên rồi chọn mô hình có nhiều điểm khớp nhất | `RANSACRegressor` |
| **Hồi quy phân vị** | Muốn dự đoán một phân vị (trung vị, 90%) hoặc khoảng dự đoán | `QuantileRegressor`, LightGBM `objective="quantile"` |
| **Hồi quy Poisson** | Nhãn là số đếm không âm (số cuộc gọi, số sự cố) | `PoissonRegressor` |
| **Hồi quy Gamma / Tweedie** | Nhãn dương lệch phải (số tiền bồi thường) | `GammaRegressor`, `TweedieRegressor` |
| **Hồi quy thứ tự** | Nhãn có thứ tự (1–5 sao) | `mord`, hoặc GBM với loss phù hợp |

Các mô hình Poisson, Gamma thuộc họ **GLM** (generalized linear model): mở rộng hồi quy tuyến tính cho nhãn không có phân phối chuẩn, bằng cách chọn phân phối cho nhãn và hàm liên kết.

---

# Phần B: Phân loại

## 6. Hồi quy logistic

Dù có chữ "hồi quy", đây là mô hình **phân loại**: dự đoán xác suất thuộc lớp dương.

### 6.1. Mô hình

$$z=\mathbf w^\top\mathbf x+b,\qquad p=P(y=1\mid\mathbf x)=\sigma(z)=\frac1{1+e^{-z}}$$

Sigmoid ép $z\in(-\infty,\infty)$ về $(0,1)$. Dự đoán lớp 1 nếu $p\ge$ ngưỡng (mặc định 0.5, ứng với $z\ge 0$). Ranh giới quyết định $\mathbf w^\top\mathbf x+b=0$ là một **siêu phẳng**, tức mô hình tuyến tính.

### 6.2. Diễn giải hệ số

Logistic mô hình hóa **log-odds** một cách tuyến tính:

$$\log\frac{p}{1-p}=\mathbf w^\top\mathbf x+b$$

Tăng $x_j$ một đơn vị thì log-odds tăng $w_j$, tức **odds nhân với $e^{w_j}$** (odds ratio). Ví dụ $w_j=0.7$ ⇒ $e^{0.7}\approx2$: odds tăng gấp đôi. Đây là lý do logistic được ưa chuộng trong y tế, tài chính, khi cần giải thích.

### 6.3. Hàm mất mát và gradient

Cross-entropy (negative log-likelihood của Bernoulli):

$$J=-\frac1n\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]$$

Gradient có dạng gọn, y như hồi quy tuyến tính:

$$\frac{\partial J}{\partial\mathbf w}=\frac1nX^\top(\mathbf p-\mathbf y),\qquad\frac{\partial J}{\partial b}=\frac1n\sum_i(p_i-y_i)$$

$J$ lồi nên gradient descent hội tụ tới cực tiểu toàn cục. Không có nghiệm đóng, nên phải lặp (gradient descent, L-BFGS, Newton).

**Vì sao không dùng MSE cho phân loại?** MSE kết hợp sigmoid cho hàm không lồi theo tham số và gradient bị triệt tiêu khi dự đoán sai mà rất tự tin (sigmoid bão hòa). Cross-entropy không gặp vấn đề này và là MLE đúng cho nhãn Bernoulli.

### 6.4. Tự cài đặt

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip tránh tràn số

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []
        eps = 1e-12
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            grad_w = X.T @ (p - y) / n + self.l2 * self.w
            grad_b = (p - y).mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.losses.append(
                -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            )
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 6.5. Dùng scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, penalty="l2", max_iter=1000),
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

### 6.6. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Nhanh, ít tham số, khó overfit | Ranh giới tuyến tính; cần tự tạo đặc trưng phi tuyến |
| Xác suất đầu ra khá hiệu chỉnh | Nhạy với đa cộng tuyến, cần chuẩn hóa |
| Hệ số diễn giải được (odds ratio) | Kém hơn cây/boosting khi có tương tác phức tạp |
| Baseline phân loại tốt nhất | |

---

## 7. Phân loại đa lớp

Với $K>2$ lớp có ba hướng.

### 7.1. Softmax (multinomial)

Mỗi lớp có một vector trọng số, xác suất lớp $k$:

$$P(y=k\mid\mathbf x)=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}},\qquad z_k=\mathbf w_k^\top\mathbf x+b_k$$

Loss là cross-entropy phân loại: $-\frac1n\sum_i\log P(y=y_i\mid\mathbf x_i)$. Đây là lớp đầu ra chuẩn của mạng neural phân loại. Các xác suất cộng lại bằng 1.

### 7.2. One-vs-Rest (OvR)

Huấn luyện $K$ bộ phân loại nhị phân, mỗi cái tách một lớp khỏi phần còn lại; dự đoán lớp có điểm cao nhất. Dùng được với mọi bộ phân loại nhị phân (SVM chẳng hạn).

### 7.3. One-vs-One (OvO)

Huấn luyện $K(K-1)/2$ bộ phân loại cho từng cặp lớp, bỏ phiếu. Mỗi bộ chỉ học trên dữ liệu của hai lớp nên nhanh khi mỗi lớp ít mẫu (SVM dùng cách này).

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC

OneVsRestClassifier(LinearSVC())
OneVsOneClassifier(LinearSVC())
LogisticRegression(max_iter=1000)       # mặc định đa lớp là softmax
```

Cây, Random Forest, KNN, Naive Bayes xử lý đa lớp trực tiếp.

---

## 8. Phân loại đa nhãn

Mỗi mẫu có thể thuộc **nhiều lớp cùng lúc** (một bài viết vừa về chính trị vừa về kinh tế). Khác đa lớp, nơi mỗi mẫu đúng một lớp.

| Cách | Mô tả | Nhược |
|---|---|---|
| **Binary relevance** | Một bộ phân loại nhị phân cho mỗi nhãn | Bỏ qua tương quan giữa các nhãn |
| **Classifier chain** | Xếp các bộ phân loại thành chuỗi, mỗi bộ dùng dự đoán của các bộ trước | Phụ thuộc thứ tự |
| **Label powerset** | Coi mỗi tổ hợp nhãn là một lớp | Bùng nổ số lớp |
| **Mạng neural, đầu ra sigmoid** | Mỗi nhãn một sigmoid, loss là tổng cross-entropy nhị phân | Cần DL |

```python
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
MultiOutputClassifier(LogisticRegression(max_iter=1000))
ClassifierChain(LogisticRegression(max_iter=1000), random_state=42)
```

Metric cho đa nhãn: Hamming loss, F1 trung bình micro/macro, subset accuracy (nghiêm ngặt).

---

## 9. Ngưỡng quyết định và chi phí lỗi

Mô hình xác suất cho $p$; **bạn** chọn ngưỡng để đổi thành quyết định. Ngưỡng 0.5 chỉ tối ưu khi hai loại lỗi tốn kém như nhau và dữ liệu cân bằng.

Nếu bỏ sót một ca dương (FN) tốn $C_{FN}$ và báo nhầm (FP) tốn $C_{FP}$, thì quyết định "dương" khi:

$$p>\frac{C_{FP}}{C_{FP}+C_{FN}}$$

Ví dụ bỏ sót ca bệnh tốn gấp 9 lần báo nhầm ⇒ ngưỡng $=1/(1+9)=0.1$.

Chọn ngưỡng thực hành: vẽ đường precision-recall trên **tập validation**, chọn ngưỡng đạt mục tiêu nghiệp vụ (ví dụ "recall ≥ 95%, precision cao nhất có thể").

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

prec, rec, thr = precision_recall_curve(y_val, proba_val)
ok = rec[:-1] >= 0.95                     # rec/prec dài hơn thr đúng 1 phần tử
best = thr[ok][np.argmax(prec[:-1][ok])]
y_pred = (proba_test >= best).astype(int)
```

---

## 10. Hiệu chỉnh xác suất (calibration)

Mô hình **được hiệu chỉnh tốt** khi trong số các mẫu nó dự đoán $p=0.8$, đúng khoảng 80% là lớp dương. Cần khi dùng xác suất để ra quyết định (tính chi phí kỳ vọng, xếp hạng rủi ro), không chỉ để phân lớp.

- Logistic Regression thường khá tốt.
- Random Forest, Naive Bayes, SVM, boosting thường **không** hiệu chỉnh (xác suất bị đẩy về giữa hoặc hai đầu).
- Lấy mẫu lại hoặc `class_weight` làm lệch xác suất.

**Kiểm tra:** biểu đồ độ tin cậy (reliability diagram) và Brier score (MSE giữa xác suất dự đoán và nhãn, càng nhỏ càng tốt).

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

frac_pos, mean_pred = calibration_curve(y_val, proba_val, n_bins=10)
# Vẽ mean_pred (x) với frac_pos (y); đường chéo y = x là hiệu chỉnh hoàn hảo
print("Brier:", brier_score_loss(y_val, proba_val))
```

**Sửa:**
- **Platt scaling** (`method="sigmoid"`): fit một logistic regression trên điểm của mô hình. Tốt khi dữ liệu ít.
- **Isotonic regression** (`method="isotonic"`): linh hoạt hơn, cần nhiều dữ liệu (hàng nghìn mẫu) để khỏi overfit.

```python
calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
```

Hiệu chỉnh phải fit trên dữ liệu **không dùng để huấn luyện mô hình gốc** (`CalibratedClassifierCV` tự xử lý bằng cross-validation).

---

**Trước:** [Chương 3](03-du-lieu-va-feature-engineering.md) | **Tiếp theo:** [Chương 5: Cây quyết định và ensemble](05-cay-va-ensemble.md)
