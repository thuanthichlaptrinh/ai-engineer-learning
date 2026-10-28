# Chương 2: Toán nền tảng và tối ưu hóa

Chương này nối toán ở GĐ0 với ML. Không cần chứng minh chặt chẽ, nhưng phải hiểu **vì sao** các công thức loss và regularization có dạng như vậy.

## Mục lục

1. [Hàm mất mát](#1-hàm-mất-mát)
2. [Ước lượng hợp lý cực đại (MLE) và MAP](#2-ước-lượng-hợp-lý-cực-đại-mle-và-map)
3. [Tối ưu hóa bằng gradient](#3-tối-ưu-hóa-bằng-gradient)
4. [Các biến thể của gradient descent](#4-các-biến-thể-của-gradient-descent)
5. [Lập lịch learning rate và chẩn đoán huấn luyện](#5-lập-lịch-learning-rate-và-chẩn-đoán-huấn-luyện)
6. [Đại số tuyến tính cho ML](#6-đại-số-tuyến-tính-cho-ml)
7. [Xác suất và thống kê cho ML](#7-xác-suất-và-thống-kê-cho-ml)
8. [Lý thuyết thông tin](#8-lý-thuyết-thông-tin)
9. [Lời nguyền số chiều](#9-lời-nguyền-số-chiều)

---

## 1. Hàm mất mát

Hàm mất mát $L(\hat{y}, y)$ đo cái giá phải trả khi dự đoán $\hat{y}$ mà đáp án là $y$. Chọn loss chính là chọn "thế nào là sai" và nó định hình cả hành vi của mô hình.

### 1.1. Loss cho hồi quy

| Loss | Công thức | Tính chất |
|---|---|---|
| **MSE** (bình phương) | $(y-\hat y)^2$ | Phạt rất nặng sai số lớn, nhạy với outlier, trơn nên dễ tối ưu. Mô hình ước lượng **trung bình** có điều kiện |
| **MAE** (tuyệt đối) | $\lvert y-\hat y\rvert$ | Bền với outlier, không trơn tại 0. Ước lượng **trung vị** có điều kiện |
| **Huber** | Bình phương khi sai số nhỏ, tuyến tính khi lớn | Kết hợp ưu điểm hai loại trên, có ngưỡng $\delta$ |
| **Log-cosh**, **Quantile (pinball)** | Dạng trơn của MAE / ước lượng phân vị | Quantile loss dùng để dự đoán khoảng (ví dụ phân vị 10% và 90%) |

### 1.2. Loss cho phân loại

| Loss | Công thức | Ghi chú |
|---|---|---|
| **0-1** | $\mathbb{1}[\hat y \ne y]$ | Là thứ ta thực sự muốn (đếm lỗi) nhưng không khả vi, không tối ưu trực tiếp được |
| **Cross-entropy / log loss** | $-[y\log p + (1-y)\log(1-p)]$ | Phạt rất nặng khi tự tin sai. Dùng với Logistic Regression, mạng neural |
| **Hinge** | $\max(0, 1 - y f(x))$ với $y\in\{-1,+1\}$ | Dùng trong SVM, chỉ phạt khi nằm trong lề hoặc sai phía |
| **Exponential** | $e^{-y f(x)}$ | Dùng trong AdaBoost |

Cross-entropy và hinge là các **hàm thay thế (surrogate)** trơn, lồi, khả vi, thay cho loss 0-1 khó tối ưu.

### 1.3. Hàm chi phí

Chi phí là trung bình loss trên tập train, thường cộng thêm một khoảng phạt regularization:

$$J(\theta) = \frac{1}{n}\sum_{i=1}^{n} L\big(f_\theta(x_i), y_i\big) + \lambda\, \Omega(\theta)$$

---

## 2. Ước lượng hợp lý cực đại (MLE) và MAP

Hầu hết loss "tự nhiên" xuất phát từ một giả định xác suất. Hiểu điều này cho bạn một nguyên tắc chung để tự suy ra loss cho bài toán mới.

### 2.1. MLE

Chọn tham số làm cho dữ liệu quan sát được **có khả năng xảy ra nhất**:

$$\hat\theta_{MLE} = \arg\max_\theta \prod_i p(y_i \mid x_i; \theta) = \arg\min_\theta \; -\sum_i \log p(y_i \mid x_i; \theta)$$

(Lấy log để biến tích thành tổng; cực đại log-likelihood tương đương cực tiểu negative log-likelihood.)

**Hồi quy tuyến tính ⇒ MSE.** Giả sử $y = \mathbf{w}^\top\mathbf{x} + \varepsilon$ với $\varepsilon \sim \mathcal{N}(0,\sigma^2)$. Khi đó

$$-\log p(y\mid x) = \frac{(y - \mathbf{w}^\top\mathbf{x})^2}{2\sigma^2} + \text{hằng số}$$

Cực tiểu tổng này chính là cực tiểu tổng bình phương sai số. Vậy **MSE là MLE khi nhiễu có phân phối chuẩn**. Nếu nhiễu có đuôi nặng (nhiều outlier), nhiễu Laplace dẫn đến MAE, hợp lý hơn.

**Hồi quy logistic ⇒ cross-entropy.** Giả sử $y\in\{0,1\}$ theo Bernoulli với $P(y=1\mid x) = p = \sigma(\mathbf{w}^\top\mathbf{x})$. Khi đó

$$-\log p(y\mid x) = -\big[y\log p + (1-y)\log(1-p)\big]$$

Chính là cross-entropy. Với nhiều lớp (phân phối categorical) ta được cross-entropy dạng softmax.

### 2.2. MAP và regularization

**MAP** (maximum a posteriori) thêm **niềm tin trước** (prior) về tham số, rồi cực đại xác suất hậu nghiệm theo Bayes:

$$\hat\theta_{MAP} = \arg\max_\theta \; p(\theta \mid \text{dữ liệu}) = \arg\max_\theta \big[\log p(\text{dữ liệu}\mid\theta) + \log p(\theta)\big]$$

| Prior cho trọng số | Khoảng phạt thêm vào chi phí | Tên |
|---|---|---|
| Chuẩn $\mathcal{N}(0,\tau^2)$ | $\propto \sum w_j^2$ | **L2 / Ridge** |
| Laplace (nhọn quanh 0) | $\propto \sum \lvert w_j\rvert$ | **L1 / Lasso** |

Nghĩa là **regularization chính là một niềm tin trước rằng trọng số nên nhỏ**. Prior chuẩn nói "trọng số có thể nhỏ vừa phải", prior Laplace nói "đa số trọng số gần như bằng đúng 0", nên L1 cho nghiệm thưa.

### 2.3. Ý nghĩa

- Khi chọn loss, hãy hỏi: nhiễu của dữ liệu trông thế nào? Có nhiều outlier không?
- Hiểu MLE giúp bạn đọc được các loss ở GĐ2, GĐ3 (cross-entropy của LLM là negative log-likelihood của token tiếp theo).

---

## 3. Tối ưu hóa bằng gradient

### 3.1. Ý tưởng

Cần tìm $\theta$ làm $J(\theta)$ nhỏ nhất. Gradient $\nabla J$ chỉ hướng tăng nhanh nhất, nên đi ngược hướng đó:

$$\theta \leftarrow \theta - \eta\, \nabla_\theta J(\theta)$$

$\eta$ là **learning rate** (bước học).

### 3.2. Ảnh hưởng của learning rate

| $\eta$ | Hiện tượng |
|---|---|
| Quá nhỏ | Loss giảm cực chậm, tốn thời gian |
| Vừa | Loss giảm đều rồi ổn định |
| Quá lớn | Loss dao động mạnh hoặc tăng vọt (phân kỳ) |

Luôn vẽ **đường loss theo vòng lặp** để kiểm tra.

### 3.3. Hàm lồi và không lồi

- **Lồi** (convex): chỉ có một cực tiểu toàn cục, gradient descent chắc chắn tới được nếu $\eta$ hợp lý. Linear Regression (MSE), Logistic Regression, SVM đều lồi.
- **Không lồi**: có nhiều cực tiểu địa phương và điểm yên ngựa. Mạng neural thuộc loại này, nhưng thực tế vẫn huấn luyện được tốt bằng biến thể của gradient descent.

### 3.4. Ảnh hưởng của thang đo đặc trưng

Khi các đặc trưng có thang đo rất khác nhau (diện tích 10–500, số phòng 1–5), mặt chi phí bị kéo dài thành hình elip hẹp, gradient descent đi zig-zag và hội tụ chậm. **Chuẩn hóa đặc trưng** làm mặt chi phí tròn hơn và hội tụ nhanh hơn nhiều.

---

## 4. Các biến thể của gradient descent

### 4.1. Theo lượng dữ liệu mỗi lần cập nhật

| Biến thể | Mỗi bước dùng | Ưu | Nhược |
|---|---|---|---|
| **Batch GD** | Toàn bộ dữ liệu | Gradient chính xác, quỹ đạo mượt | Chậm với dữ liệu lớn, tốn bộ nhớ |
| **Stochastic GD (SGD)** | 1 mẫu | Cập nhật rất nhanh, thoát được cực tiểu nông | Gradient nhiễu, dao động |
| **Mini-batch GD** | Một lô (32–512 mẫu) | Cân bằng, tận dụng được tính toán song song | Thêm siêu tham số batch size |

Mini-batch là lựa chọn phổ biến nhất, và thường được gọi tắt là "SGD".

```python
import numpy as np

def minibatch_sgd(X, y, lr=0.05, epochs=50, batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        idx = rng.permutation(n)                  # xáo trộn mỗi epoch
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            Xb, yb = X[batch], y[batch]
            err = Xb @ w + b - yb
            w -= lr * (2 / len(batch)) * (Xb.T @ err)
            b -= lr * (2 / len(batch)) * err.sum()
    return w, b
```

### 4.2. Cải tiến bước đi

**Momentum.** Tích lũy "vận tốc" từ các gradient trước, giúp đi nhanh theo hướng nhất quán và giảm dao động:

$$v \leftarrow \beta v + \nabla J, \qquad \theta \leftarrow \theta - \eta v \qquad (\beta \approx 0.9)$$

**RMSProp.** Chia gradient cho căn của trung bình bình phương gradient gần đây, nên mỗi tham số có bước học riêng.

**Adam.** Kết hợp momentum và RMSProp, hiện là lựa chọn mặc định cho mạng neural:

$$m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$$
$$\hat m = \frac{m}{1-\beta_1^t}, \quad \hat v = \frac{v}{1-\beta_2^t}, \qquad \theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

Giá trị mặc định thường dùng: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. $g$ là gradient hiện tại, $t$ là số bước.

### 4.3. Bảng chọn nhanh

| Tình huống | Nên dùng |
|---|---|
| Mô hình tuyến tính nhỏ, dữ liệu vừa | Nghiệm đóng hoặc L-BFGS (scikit-learn tự chọn) |
| Dữ liệu rất lớn, mô hình tuyến tính | SGD, mini-batch |
| Mạng neural (GĐ2) | Adam hoặc AdamW làm mặc định |
| Cây (Random Forest, GBM) | Không dùng gradient descent theo tham số, mà dùng thuật toán chia nút |

---

## 5. Lập lịch learning rate và chẩn đoán huấn luyện

### 5.1. Lịch learning rate

Bắt đầu với learning rate lớn để tiến nhanh, giảm dần để hội tụ chính xác:

- **Step decay:** giảm một nửa mỗi N epoch.
- **Exponential decay:** nhân với hệ số $<1$ mỗi epoch.
- **Cosine annealing:** giảm theo đường cos.
- **Warmup:** tăng dần từ giá trị nhỏ ở vài bước đầu (quan trọng khi huấn luyện Transformer).

### 5.2. Đọc đường loss

| Quan sát | Có thể do | Cách xử lý |
|---|---|---|
| Loss tăng hoặc thành `NaN` | Learning rate quá lớn, hoặc đặc trưng chưa chuẩn hóa | Giảm learning rate, chuẩn hóa |
| Loss giảm rất chậm | Learning rate quá nhỏ | Tăng learning rate |
| Loss train giảm, loss validation tăng | Overfitting | Regularization, dừng sớm, thêm dữ liệu |
| Loss dao động mạnh | Batch quá nhỏ hoặc learning rate lớn | Tăng batch, giảm learning rate |
| Loss phẳng ở mức cao | Underfit hoặc mô hình quá yếu | Mô hình mạnh hơn, đặc trưng tốt hơn |

### 5.3. Dừng sớm (early stopping)

Theo dõi loss validation, dừng khi nó không cải thiện sau $k$ lượt liên tiếp, rồi lấy tham số tại điểm tốt nhất. Cách regularization rẻ và hiệu quả, dùng cho cả boosting lẫn mạng neural.

---

## 6. Đại số tuyến tính cho ML

| Khái niệm | Vai trò trong ML |
|---|---|
| Vector, ma trận | Một mẫu là vector đặc trưng, tập dữ liệu là ma trận $X\in\mathbb{R}^{n\times d}$ |
| Nhân ma trận | Dự đoán của mô hình tuyến tính: $\hat{\mathbf y}=X\mathbf w$ |
| Chuyển vị, nghịch đảo | Nghiệm đóng $\mathbf w=(X^\top X)^{-1}X^\top\mathbf y$ |
| Chuẩn $\lVert\cdot\rVert_1$, $\lVert\cdot\rVert_2$ | Độ lớn trọng số trong regularization, khoảng cách trong KNN/K-Means |
| Tích vô hướng | Độ tương đồng (cosine similarity, SVM kernel) |
| Vector riêng, giá trị riêng | PCA: hướng có phương sai lớn nhất |
| SVD (phân rã giá trị suy biến) | PCA, giảm hạng, hệ gợi ý |
| Hạng ma trận, đa cộng tuyến | $X^\top X$ không khả nghịch khi các đặc trưng phụ thuộc tuyến tính |

**Dạng ma trận của MSE và gradient** (dùng được ngay trong NumPy, không cần vòng lặp):

$$J = \frac{1}{n}\lVert X\mathbf w - \mathbf y\rVert_2^2, \qquad \nabla_{\mathbf w} J = \frac{2}{n}X^\top(X\mathbf w-\mathbf y)$$

Thói quen quan trọng: **vector hóa** phép tính thay vì viết vòng lặp Python, nhanh hơn hàng chục đến hàng trăm lần. Luôn kiểm tra `shape` của từng phép nhân ma trận.

---

## 7. Xác suất và thống kê cho ML

### 7.1. Công thức cần nhớ

- **Xác suất có điều kiện và Bayes:** $P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B)}$.
- **Kỳ vọng:** $\mathbb{E}[X]$. **Phương sai:** $\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}X)^2]$.
- **Hiệp phương sai và tương quan:** đo mức hai biến cùng biến thiên. Tương quan Pearson chỉ đo quan hệ **tuyến tính**.
- **Độc lập:** $P(A,B)=P(A)P(B)$. Nền của giả định Naive Bayes.

### 7.2. Các phân phối thường gặp

| Phân phối | Dùng khi | Gặp trong |
|---|---|---|
| Bernoulli / Binomial | Kết quả 0/1, đếm số lần thành công | Phân loại nhị phân |
| Categorical / Multinomial | Chọn 1 trong $K$ | Phân loại đa lớp, đếm từ |
| Gauss (chuẩn) | Đại lượng liên tục, tổng của nhiều yếu tố nhỏ | Nhiễu hồi quy, GMM, Gaussian Naive Bayes |
| Poisson | Đếm sự kiện trong khoảng thời gian | Hồi quy Poisson (số cuộc gọi, số lỗi) |
| Đều (uniform) | Mọi giá trị như nhau | Khởi tạo ngẫu nhiên |

### 7.3. Định lý giới hạn trung tâm và khoảng tin cậy

Trung bình của nhiều mẫu độc lập xấp xỉ phân phối chuẩn, bất kể phân phối gốc. Nhờ vậy ta ước lượng được độ bất định của một thống kê (ví dụ điểm F1) và báo cáo dạng "trung bình ± độ lệch chuẩn" hoặc khoảng tin cậy.

### 7.4. Bootstrap

Ước lượng độ bất định mà không cần giả định phân phối: lấy mẫu **có hoàn lại** từ dữ liệu nhiều lần, tính thống kê trên từng mẫu, rồi nhìn phân phối của chúng.

```python
import numpy as np

def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = [stat(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return np.quantile(stats, [alpha / 2, 1 - alpha / 2])
```

Bootstrap cũng là nền của Random Forest (bagging).

### 7.5. Kiểm định giả thuyết

Dùng để hỏi "chênh lệch này có thật không hay do may rủi?". Giá trị $p$ nhỏ nghĩa là kết quả khó xảy ra nếu giả thuyết không (không có khác biệt) đúng. Lưu ý khi so sánh mô hình bằng cross-validation: các fold chia sẻ dữ liệu nên không độc lập, kiểm định t thông thường sẽ **đánh giá thấp độ bất định** (xem [Chương 8](08-danh-gia-mo-hinh.md)).

---

## 8. Lý thuyết thông tin

| Khái niệm | Công thức | Ý nghĩa |
|---|---|---|
| **Entropy** | $H(p)=-\sum_k p_k\log p_k$ | Mức bất định của một phân phối. Lớn nhất khi các lớp đều nhau |
| **Cross-entropy** | $H(p,q)=-\sum_k p_k\log q_k$ | Số "bit" trung bình khi dùng $q$ để mô tả dữ liệu thật phân phối $p$. Chính là loss phân loại |
| **KL divergence** | $D_{KL}(p\Vert q)=\sum_k p_k\log\frac{p_k}{q_k}$ | "Khoảng cách" (không đối xứng) giữa hai phân phối |
| **Information gain** | Entropy trước chia trừ entropy sau chia | Tiêu chí chọn chỗ chia của cây quyết định |
| **Mutual information** | $I(X;Y)=H(Y)-H(Y\mid X)$ | Mức thông tin $X$ cho biết về $Y$, bắt được cả quan hệ phi tuyến, dùng chọn đặc trưng |

Mối liên hệ: $H(p,q)=H(p)+D_{KL}(p\Vert q)$. Vì $H(p)$ cố định với dữ liệu, **cực tiểu cross-entropy chính là cực tiểu KL**, tức kéo phân phối dự đoán $q$ về gần phân phối thật $p$.

Perplexity của mô hình ngôn ngữ (GĐ3) là $e^{\text{cross-entropy}}$.

---

## 9. Lời nguyền số chiều

Khi số chiều $d$ tăng:
- Dữ liệu trở nên **thưa**: cần lượng mẫu tăng theo hàm mũ của $d$ để phủ kín không gian.
- **Khoảng cách mất ý nghĩa:** khoảng cách tới điểm gần nhất và xa nhất gần bằng nhau, nên KNN, K-Means suy giảm.
- Dễ overfit hơn vì có nhiều cách để "vừa khít" dữ liệu ngẫu nhiên.

Cách đối phó: chọn đặc trưng, giảm chiều (PCA), regularization, mô hình ít nhạy với số chiều (cây, boosting), và thêm dữ liệu.

---

**Trước:** [Chương 1](01-tong-quan-va-cac-kieu-hoc.md) | **Tiếp theo:** [Chương 3: Dữ liệu và feature engineering](03-du-lieu-va-feature-engineering.md)
