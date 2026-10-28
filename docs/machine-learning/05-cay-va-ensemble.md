# Chương 5: Cây quyết định và ensemble

Với dữ liệu dạng bảng, nhóm mô hình này (đặc biệt gradient boosting) thường cho kết quả tốt nhất trong thực tế, và là thứ bạn sẽ dùng nhiều nhất.

## Mục lục

1. [Cây quyết định](#1-cây-quyết-định)
2. [Vì sao ensemble hiệu quả](#2-vì-sao-ensemble-hiệu-quả)
3. [Bagging và Random Forest](#3-bagging-và-random-forest)
4. [Boosting](#4-boosting)
5. [Gradient Boosting](#5-gradient-boosting)
6. [XGBoost, LightGBM, CatBoost](#6-xgboost-lightgbm-catboost)
7. [Voting và Stacking](#7-voting-và-stacking)
8. [So sánh và hướng dẫn chọn](#8-so-sánh-và-hướng-dẫn-chọn)
9. [Mẹo thực hành](#9-mẹo-thực-hành)

---

## 1. Cây quyết định

### 1.1. Ý tưởng

Cây chia không gian đặc trưng thành các vùng chữ nhật bằng chuỗi câu hỏi dạng `đặc trưng ≤ ngưỡng?`. Mỗi lá chứa một dự đoán: lớp phổ biến nhất (phân loại) hoặc giá trị trung bình (hồi quy).

```
                 diện tích ≤ 80?
                 /            \
              có               không
        số phòng ≤ 2?        khu vực = trung tâm?
        /        \              /          \
    giá 1.5 tỷ  giá 2.2 tỷ   giá 5 tỷ    giá 3.4 tỷ
```

### 1.2. Thuật toán CART (xây cây tham lam)

Tại mỗi nút, thử mọi (đặc trưng, ngưỡng) và chọn chỗ chia **giảm độ không thuần nhiều nhất**, rồi đệ quy cho hai nút con, cho đến khi gặp điều kiện dừng.

**Độ không thuần (phân loại)** với $p_k$ là tỉ lệ lớp $k$ trong nút:

$$\text{Gini}=1-\sum_kp_k^2,\qquad\text{Entropy}=-\sum_kp_k\log_2p_k$$

Nút thuần (một lớp duy nhất) có Gini = 0. Chỗ chia tốt nhất làm **trung bình có trọng số** độ không thuần của hai nút con nhỏ nhất (tương đương information gain lớn nhất).

**Hồi quy:** tiêu chí là giảm phương sai (MSE) trong các nút con.

Vì tham lam, cây không đảm bảo cây tối ưu toàn cục, chỉ tối ưu từng bước.

### 1.3. Kiểm soát độ phức tạp

Cây không giới hạn sẽ tăng trưởng đến khi mỗi lá chỉ còn 1 mẫu: overfit hoàn toàn (lỗi train bằng 0). Điều chỉnh:

| Siêu tham số | Tác dụng khi **tăng** |
|---|---|
| `max_depth` | Cây sâu hơn, phức tạp hơn |
| `min_samples_split` | Cần nhiều mẫu hơn mới được chia, cây đơn giản hơn |
| `min_samples_leaf` | Lá phải có nhiều mẫu hơn, mượt hơn |
| `max_leaf_nodes` | Giới hạn số lá |
| `ccp_alpha` | Cắt tỉa mạnh hơn (cost-complexity pruning) |

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                              random_state=42).fit(X_train, y_train)
plot_tree(tree, feature_names=X_train.columns, filled=True, max_depth=3)
```

**Cắt tỉa (pruning):** xây cây lớn rồi cắt các nhánh ít giá trị, chọn mức cắt (`ccp_alpha`) bằng cross-validation.

### 1.4. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Dễ diễn giải, vẽ được | **Phương sai cao:** đổi một ít dữ liệu, cây đổi hẳn |
| Không cần chuẩn hóa đặc trưng | Một cây đơn thường kém chính xác |
| Xử lý được hỗn hợp số và phân loại | Ranh giới vuông góc trục, kém với quan hệ chéo |
| Tự bắt tương tác và quan hệ phi tuyến | Không ngoại suy: ngoài miền train chỉ trả về giá trị cố định |
| Bền với outlier ở đặc trưng | Thiên vị đặc trưng có nhiều giá trị khác nhau |

Điểm yếu lớn nhất (phương sai cao) chính là lý do ra đời các phương pháp ensemble.

---

## 2. Vì sao ensemble hiệu quả

**Ý tưởng:** kết hợp nhiều mô hình yếu hoặc không hoàn hảo để được mô hình mạnh hơn (trí tuệ đám đông).

Giả sử có $B$ mô hình, mỗi mô hình có phương sai $\sigma^2$ và tương quan từng cặp $\rho$. Phương sai của **trung bình** chúng là:

$$\text{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$

Hai hệ quả:
- Lấy trung bình nhiều mô hình giảm phần $\frac{1-\rho}{B}\sigma^2$, tiến về 0 khi $B$ lớn.
- Phần $\rho\sigma^2$ còn lại phụ thuộc vào **mức độ tương quan**: các mô hình càng **đa dạng** (ít tương quan) thì ensemble càng mạnh.

Hai họ chính:

| | Bagging (song song) | Boosting (tuần tự) |
|---|---|---|
| Cách làm | Huấn luyện độc lập, lấy trung bình | Huấn luyện nối tiếp, mô hình sau sửa lỗi mô hình trước |
| Giảm chủ yếu | **Variance** | **Bias** |
| Mô hình thành phần | Mô hình phức tạp (cây sâu) | Mô hình yếu (cây nông) |
| Rủi ro overfit | Thấp | Cao hơn, cần kiểm soát |
| Ví dụ | Random Forest | AdaBoost, Gradient Boosting, XGBoost, LightGBM |

---

## 3. Bagging và Random Forest

### 3.1. Bagging (Bootstrap Aggregating)

1. Tạo $B$ tập dữ liệu bằng **bootstrap** (lấy mẫu có hoàn lại, kích thước bằng tập gốc).
2. Huấn luyện một mô hình (thường cây sâu) trên mỗi tập.
3. Hồi quy: lấy trung bình. Phân loại: bỏ phiếu đa số (hoặc trung bình xác suất).

Mỗi bootstrap chứa khoảng 63% mẫu khác nhau; 37% còn lại là mẫu **out-of-bag (OOB)**, dùng làm tập validation miễn phí.

### 3.2. Random Forest

Bagging cây quyết định, **cộng thêm** một thủ thuật: ở mỗi lần chia, chỉ xét một **tập con ngẫu nhiên các đặc trưng** (`max_features`). Việc này làm các cây ít tương quan hơn (giảm $\rho$), nên ensemble tốt hơn bagging thuần.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,        # số cây; nhiều hơn thì ổn định hơn, hiếm khi tệ đi
    max_features="sqrt",     # số đặc trưng xét mỗi lần chia (phân loại); hồi quy hay dùng 1.0 hoặc 1/3
    min_samples_leaf=3,
    oob_score=True,          # ước lượng lỗi bằng mẫu out-of-bag
    n_jobs=-1,
    random_state=42,
).fit(X_train, y_train)

print("OOB score:", rf.oob_score_)
```

| Siêu tham số | Ghi chú |
|---|---|
| `n_estimators` | Tăng đến khi điểm phẳng; không gây overfit, chỉ tốn thời gian |
| `max_features` | Nhỏ hơn ⇒ cây đa dạng hơn nhưng yếu hơn từng cây. Tham số quan trọng nhất cần tune |
| `max_depth`, `min_samples_leaf` | Kiểm soát độ phức tạp từng cây |
| `max_samples` | Tỉ lệ mẫu mỗi bootstrap |
| `class_weight` | Cho dữ liệu lệch |

### 3.3. Độ quan trọng đặc trưng

- **Impurity-based** (`rf.feature_importances_`): nhanh, nhưng thiên vị biến liên tục/có nhiều giá trị, và tính trên tập train.
- **Permutation importance:** xáo trộn một cột trên tập validation và đo mức giảm điểm. Đáng tin hơn. Xem [Chương 9](09-pipeline-tuning-va-trien-khai.md).

### 3.4. Extra Trees

Giống Random Forest nhưng chọn **ngưỡng chia ngẫu nhiên** thay vì tìm ngưỡng tốt nhất. Nhanh hơn, phương sai thấp hơn một chút, bias cao hơn một chút. `ExtraTreesClassifier`.

### 3.5. Ưu và nhược

| Ưu | Nhược |
|---|---|
| Mạnh, ít cần tuning, khó overfit | Mô hình lớn, chậm khi dự đoán và tốn bộ nhớ |
| Không cần chuẩn hóa | Khó diễn giải hơn một cây |
| OOB score cho ước lượng miễn phí | Không ngoại suy |
| Xử lý tốt dữ liệu nhiều chiều | Thường thua boosting ở độ chính xác tối đa |

---

## 4. Boosting

Huấn luyện mô hình yếu **tuần tự**, mỗi mô hình mới tập trung vào chỗ các mô hình trước làm sai.

### 4.1. AdaBoost

1. Gán trọng số bằng nhau cho mọi mẫu.
2. Huấn luyện một mô hình yếu (thường **stump**: cây sâu 1 tầng) với trọng số mẫu hiện tại.
3. Tăng trọng số các mẫu bị phân loại sai, giảm trọng số mẫu đúng.
4. Gán cho mô hình một "tiếng nói" lớn hơn nếu nó chính xác.
5. Lặp, kết quả là tổng có trọng số của các mô hình.

Nhạy với nhiễu và outlier, vì các mẫu nhãn sai sẽ liên tục bị tăng trọng số.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200, learning_rate=0.5, random_state=42,
)
```

AdaBoost có ý nghĩa lịch sử và dễ hiểu, nhưng thực tế người ta dùng gradient boosting.

---

## 5. Gradient Boosting

### 5.1. Ý tưởng

Xây mô hình cộng dần: $F_m(x)=F_{m-1}(x)+\eta\,h_m(x)$, trong đó $h_m$ là một cây nhỏ được huấn luyện để **dự đoán phần sai còn lại**.

Nói chặt chẽ hơn: đây là gradient descent **trong không gian hàm**. Mỗi cây mới xấp xỉ **gradient âm** của loss tại dự đoán hiện tại:

$$r_i^{(m)}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}$$

Với loss bình phương ($L=\tfrac12(y-F)^2$), gradient âm đúng bằng **phần dư** $y_i-F_{m-1}(x_i)$, vì vậy có cách nói trực quan "mỗi cây học phần dư của các cây trước".

### 5.2. Thuật toán

1. Khởi tạo $F_0(x)$ là hằng số tối ưu (trung bình nhãn với MSE).
2. Với $m=1,\dots,M$:
   - Tính phần dư giả $r_i^{(m)}$ cho mỗi mẫu.
   - Huấn luyện cây nông $h_m$ để dự đoán $r_i^{(m)}$.
   - Cập nhật $F_m=F_{m-1}+\eta\,h_m$.
3. Dự đoán cuối: $F_M(x)$.

**Tự cài đặt phiên bản tối giản cho hồi quy** (để hiểu cơ chế):

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGBR:
    def __init__(self, n_estimators=200, lr=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth

    def fit(self, X, y):
        self.f0 = y.mean()
        self.trees = []
        pred = np.full(len(y), self.f0)
        for _ in range(self.n_estimators):
            residual = y - pred                        # gradient âm của MSE
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            pred += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        pred = np.full(len(X), self.f0)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
```

### 5.3. Ba núm điều chỉnh chính

- **`learning_rate` ($\eta$):** nhỏ hơn thì cần nhiều cây hơn nhưng tổng quát hóa tốt hơn. Quy tắc: đặt nhỏ (0.01–0.1), tăng số cây, dùng **early stopping**.
- **Số cây** ($M$): quá nhiều thì overfit. Chọn bằng early stopping trên validation.
- **Độ phức tạp mỗi cây** (`max_depth` thường 3–8, hoặc `num_leaves`): cây nông thì mô hình cộng tính hơn, ít tương tác.

### 5.4. Kỹ thuật chống overfit

| Kỹ thuật | Cách làm |
|---|---|
| **Shrinkage** | Learning rate nhỏ |
| **Subsampling** (stochastic GB) | Mỗi cây chỉ dùng một phần hàng (`subsample` 0.5–0.9) |
| **Column subsampling** | Mỗi cây chỉ dùng một phần cột (`colsample_bytree`) |
| **Regularization trên lá** | Phạt L1/L2 lên giá trị lá (`reg_alpha`, `reg_lambda`) |
| **Giới hạn mẫu tối thiểu** | `min_child_samples`, `min_child_weight` |
| **Early stopping** | Dừng khi validation không cải thiện sau N cây |

---

## 6. XGBoost, LightGBM, CatBoost

Ba thư viện gradient boosting phổ biến nhất, đều nhanh và đã tối ưu mạnh.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Cách mọc cây | Theo tầng (level-wise) | Theo lá (leaf-wise), chọn lá giảm loss nhiều nhất | Cây đối xứng (oblivious) |
| Điểm mạnh | Ổn định, regularization tốt, dùng khai thác đạo hàm bậc hai | **Rất nhanh**, ít bộ nhớ, dữ liệu lớn | **Xử lý biến phân loại** rất tốt, ít cần tuning, chống overfit tốt |
| Biến phân loại | Cần mã hóa (hỗ trợ thử nghiệm) | Hỗ trợ trực tiếp (`categorical_feature`) | Hỗ trợ trực tiếp, mạnh nhất |
| Giá trị thiếu | Tự xử lý | Tự xử lý | Tự xử lý |
| Điểm cần để ý | Chậm hơn LightGBM | Leaf-wise dễ overfit với dữ liệu nhỏ, cần `min_child_samples` | Chậm hơn khi huấn luyện trên CPU |

Gợi ý: bắt đầu với **LightGBM**; thử **CatBoost** khi nhiều biến phân loại; **XGBoost** là lựa chọn an toàn và phổ biến.

### 6.1. LightGBM với early stopping

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

gbm = lgb.LGBMRegressor(
    n_estimators=5000,         # đặt lớn, để early stopping tự dừng
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42, verbose=-1,
)
gbm.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)
print("Số cây tốt nhất:", gbm.best_iteration_)
```

### 6.2. Bảng siêu tham số quan trọng (LightGBM; tên tương tự ở XGBoost/CatBoost)

| Tham số | Ý nghĩa | Nếu overfit thì |
|---|---|---|
| `learning_rate` | Bước học | Giảm (và tăng `n_estimators`) |
| `n_estimators` | Số cây | Dùng early stopping |
| `num_leaves` / `max_depth` | Độ phức tạp mỗi cây | Giảm |
| `min_child_samples` | Số mẫu tối thiểu mỗi lá | Tăng |
| `subsample` + `subsample_freq` | Lấy mẫu hàng | Giảm xuống 0.6–0.8 |
| `colsample_bytree` | Lấy mẫu cột | Giảm xuống 0.6–0.8 |
| `reg_alpha`, `reg_lambda` | Regularization L1/L2 | Tăng |

### 6.3. HistGradientBoosting của scikit-learn

Nếu muốn chỉ dùng scikit-learn: `HistGradientBoostingClassifier/Regressor` có tốc độ gần LightGBM, tự xử lý giá trị thiếu và biến phân loại, có early stopping tích hợp.

---

## 7. Voting và Stacking

Kết hợp các mô hình **khác loại** để tận dụng điểm mạnh khác nhau.

### 7.1. Voting

- **Hard voting:** bỏ phiếu theo lớp dự đoán.
- **Soft voting:** trung bình xác suất (thường tốt hơn, cần các mô hình cho xác suất tốt).

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

voting = VotingClassifier(
    estimators=[("lr", LogisticRegression(max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
                ("gbm", lgb.LGBMClassifier(verbose=-1, random_state=42))],
    voting="soft",
)
```

### 7.2. Stacking

Dùng dự đoán của các mô hình cơ sở làm **đặc trưng** cho một mô hình cấp trên (meta-model, thường là mô hình đơn giản như Logistic/Ridge).

**Cẩn thận rò rỉ:** dự đoán của mô hình cơ sở trên dữ liệu train phải là dự đoán **out-of-fold** (từ cross-validation), không phải dự đoán trên chính dữ liệu nó đã học. `StackingClassifier` tự làm điều này qua tham số `cv`.

```python
from sklearn.ensemble import StackingClassifier

stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
                ("gbm", lgb.LGBMClassifier(verbose=-1, random_state=42))],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
```

Stacking thường cải thiện nhỏ nhưng chắc chắn và phổ biến trong thi đấu. Đổi lại phức tạp hơn nhiều khi triển khai; chỉ dùng khi lợi ích đáng kể.

**Điều kiện để ensemble có ích:** các mô hình thành phần phải **đa dạng** (khác thuật toán, khác đặc trưng, khác seed) và đều khá tốt. Trộn năm bản gần như giống nhau không giúp gì.

---

## 8. So sánh và hướng dẫn chọn

| | Cây đơn | Random Forest | Gradient Boosting |
|---|---|---|---|
| Giảm chủ yếu | – | Variance | Bias |
| Tuning | Ít | Ít | Nhiều hơn |
| Rủi ro overfit | Cao | Thấp | Trung bình |
| Độ chính xác điển hình | Thấp | Cao | Thường cao nhất |
| Tốc độ huấn luyện | Rất nhanh | Nhanh, song song | Chậm hơn (tuần tự) |
| Diễn giải | Tốt nhất | Trung bình | Cần SHAP |
| Nhạy với nhiễu nhãn | Cao | Thấp | Cao hơn |

**Quy trình gợi ý cho bài toán dạng bảng:**

1. Baseline: dự đoán trung bình / lớp phổ biến nhất.
2. Mô hình tuyến tính (Ridge/Logistic): mốc đơn giản, có thể giải thích.
3. Random Forest: mạnh nhanh, ít tuning.
4. LightGBM/XGBoost/CatBoost + early stopping: thường tốt nhất.
5. Tuning bằng Optuna ([Chương 9](09-pipeline-tuning-va-trien-khai.md)).
6. Chỉ khi cần vắt thêm: ensemble/stacking.

Khi nào **không** dùng cây: dữ liệu có cấu trúc đặc biệt (ảnh, âm thanh, văn bản dài) thì dùng mạng neural; cần ngoại suy xa ngoài miền train (cây không làm được); cần mô hình cực nhỏ và có thể giải thích bằng công thức.

---

## 9. Mẹo thực hành

- **Đừng chuẩn hóa** cho cây và boosting; nó không giúp gì.
- **Biến phân loại nhiều giá trị:** dùng hỗ trợ native của LightGBM/CatBoost thay vì one-hot (one-hot làm cây thưa và kém hiệu quả).
- **Giá trị thiếu:** để nguyên cho LightGBM/XGBoost/CatBoost/HistGradientBoosting; chúng học được hướng đi tốt nhất cho giá trị thiếu.
- **Số cây:** đặt lớn kèm early stopping, đừng đoán.
- **Learning curve và validation curve** cho biết đang overfit hay underfit ([Chương 8](08-danh-gia-mo-hinh.md)).
- **Đặc trưng quan trọng cao bất thường** thường là dấu hiệu rò rỉ.
- **Dữ liệu mất cân bằng:** `scale_pos_weight` (XGBoost/LightGBM) hoặc `class_weight`, và đánh giá bằng PR-AUC.
- **Tái lập:** cố định `random_state`, ghi lại phiên bản thư viện (kết quả giữa các phiên bản có thể khác).

---

**Trước:** [Chương 4](04-hoi-quy-va-phan-loai.md) | **Tiếp theo:** [Chương 6: SVM, KNN, Naive Bayes và các mô hình khác](06-svm-knn-naive-bayes.md)
