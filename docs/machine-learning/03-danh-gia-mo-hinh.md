# Chương 3: Đánh giá mô hình

Đánh giá đúng quan trọng ngang với dựng mô hình. Một mô hình đánh giá sai có thể trông xuất sắc mà vô dụng khi triển khai.

> **Phạm vi:** chương này viết cho **học có giám sát** (có nhãn để so sánh). Cách đánh giá học không giám sát (silhouette, ARI, đánh giá theo mục đích) nằm ở [chương 8, mục 3](08-hoc-khong-giam-sat.md), và đánh giá phát hiện bất thường ở mục 5.2 của chương đó.

## Mục lục

1. [Nguyên tắc đánh giá](#1-nguyên-tắc-đánh-giá)
2. [Metric cho hồi quy](#2-metric-cho-hồi-quy)
3. [Metric cho phân loại](#3-metric-cho-phân-loại)
4. [Đường cong ROC và precision-recall](#4-đường-cong-roc-và-precision-recall)
5. [Phân loại đa lớp, đa nhãn và xếp hạng](#5-phân-loại-đa-lớp-đa-nhãn-và-xếp-hạng)
6. [Chiến lược kiểm định chéo](#6-chiến-lược-kiểm-định-chéo)
7. [Overfitting, underfitting, bias-variance](#7-overfitting-underfitting-bias-variance)
8. [Learning curve và validation curve](#8-learning-curve-và-validation-curve)
9. [So sánh mô hình có ý nghĩa thống kê](#9-so-sánh-mô-hình-có-ý-nghĩa-thống-kê)
10. [Phân tích lỗi](#10-phân-tích-lỗi)
11. [Từ metric kỹ thuật đến giá trị kinh doanh](#11-từ-metric-kỹ-thuật-đến-giá-trị-kinh-doanh)

---

## 1. Nguyên tắc đánh giá

1. **Đánh giá trên dữ liệu mô hình chưa từng thấy.** Điểm trên dữ liệu train không nói gì về tổng quát hóa.
2. **Metric phải phản ánh mục tiêu nghiệp vụ**, không phải metric dễ tính nhất.
3. **Luôn so với baseline.** Con số 85% chỉ có nghĩa khi biết baseline đạt bao nhiêu.
4. **Ba tập tách biệt:** train để học, validation để chọn, test để báo cáo cuối (dùng một lần).
5. **Môi trường đánh giá phải giống triển khai:** chia theo thời gian nếu dự đoán tương lai, chia theo nhóm nếu gặp nhóm mới.
6. **Báo cáo độ bất định**, không chỉ một con số: trung bình ± độ lệch chuẩn qua các fold.

---

## 2. Metric cho hồi quy

Ký hiệu: $y_i$ thật, $\hat y_i$ dự đoán, $\bar y$ trung bình nhãn.

| Metric | Công thức | Đặc điểm |
|---|---|---|
| **MAE** | $\frac1n\sum\lvert y_i-\hat y_i\rvert$ | Cùng đơn vị với $y$; bền với outlier; diễn giải: sai trung bình bao nhiêu |
| **MSE** | $\frac1n\sum(y_i-\hat y_i)^2$ | Phạt nặng sai số lớn; khác đơn vị |
| **RMSE** | $\sqrt{\text{MSE}}$ | Cùng đơn vị với $y$, phạt nặng sai số lớn; **phổ biến nhất** |
| **R²** | $1-\dfrac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ | Tỉ lệ phương sai được giải thích. 1 hoàn hảo, 0 bằng dự đoán trung bình, **có thể âm** nếu tệ hơn baseline |
| **R² hiệu chỉnh** | $1-(1-R^2)\dfrac{n-1}{n-p-1}$ | Phạt số đặc trưng $p$; dùng khi so mô hình có số biến khác nhau |
| **MAPE** | $\frac1n\sum\left\lvert\frac{y_i-\hat y_i}{y_i}\right\rvert$ | Sai số %; **hỏng** khi $y$ gần 0, và thiên vị dự đoán thấp |
| **RMSLE** | RMSE trên $\log(1+y)$ | Quan tâm sai số tương đối; dùng khi nhãn trải nhiều bậc độ lớn (giá) |
| **Median AE** | Trung vị của $\lvert y-\hat y\rvert$ | Bền nhất với outlier |

**Chọn metric:**
- Sai số lớn đặc biệt tệ (vượt tồn kho gây thiếu hàng) ⇒ RMSE.
- Mọi sai số đáng kể như nhau, có outlier ⇒ MAE.
- Quan tâm sai số theo tỉ lệ (nhà 1 tỷ sai 100 triệu khác nhà 10 tỷ sai 100 triệu) ⇒ RMSLE hoặc MAPE (cẩn thận).
- R² tiện để giao tiếp nhưng tránh dùng một mình; so sánh nó giữa các bộ dữ liệu khác nhau là vô nghĩa.

```python
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error, mean_squared_log_error)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2 = r2_score(y_true, y_pred)
rmsle = mean_squared_log_error(y_true, y_pred) ** 0.5     # chỉ khi y, y_pred >= 0
```

**Luôn so với baseline** (`DummyRegressor(strategy="mean")` hoặc `"median"`), và phân tích phần dư (xem [Chương 5](05-hoi-quy-va-phan-loai.md)).

---

## 3. Metric cho phân loại

### 3.1. Ma trận nhầm lẫn (nhị phân)

|  | Dự đoán dương | Dự đoán âm |
|---|---|---|
| **Thật dương** | TP (đúng dương) | FN (bỏ sót) |
| **Thật âm** | FP (báo nhầm) | TN (đúng âm) |

### 3.2. Các metric từ ma trận nhầm lẫn

| Metric | Công thức | Trả lời |
|---|---|---|
| **Accuracy** | $\dfrac{TP+TN}{TP+TN+FP+FN}$ | Đúng bao nhiêu %? (**lừa người khi dữ liệu lệch**) |
| **Precision** | $\dfrac{TP}{TP+FP}$ | Trong số dự đoán dương, bao nhiêu đúng? |
| **Recall** (sensitivity, TPR) | $\dfrac{TP}{TP+FN}$ | Trong số thật sự dương, bắt được bao nhiêu? |
| **Specificity** (TNR) | $\dfrac{TN}{TN+FP}$ | Trong số thật sự âm, nhận đúng bao nhiêu? |
| **FPR** | $\dfrac{FP}{FP+TN}$ | Tỉ lệ báo động nhầm |
| **F1** | $\dfrac{2PR}{P+R}$ | Trung bình điều hòa của precision và recall |
| **F-beta** | $(1+\beta^2)\dfrac{PR}{\beta^2P+R}$ | $\beta>1$ ưu tiên recall, $\beta<1$ ưu tiên precision |
| **Balanced accuracy** | $\frac12(\text{Recall}+\text{Specificity})$ | Accuracy không bị lệch bởi tỉ lệ lớp |
| **MCC** | $\dfrac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Một con số cân bằng cho cả bốn ô, từ −1 đến +1; tốt cho dữ liệu lệch |
| **Cohen's kappa** | Mức đồng thuận vượt mức ngẫu nhiên | Đồng thuận giữa mô hình và nhãn (hoặc giữa người gán nhãn) |

**Đánh đổi precision–recall:** hạ ngưỡng thì bắt được nhiều ca dương hơn (recall tăng) nhưng báo nhầm nhiều hơn (precision giảm), và ngược lại.

### 3.3. Chọn metric theo hậu quả

| Tình huống | Lỗi nào tệ hơn | Ưu tiên |
|---|---|---|
| Sàng lọc bệnh nặng | Bỏ sót ca bệnh (FN) | **Recall** cao |
| Lọc spam | Chặn nhầm thư quan trọng (FP) | **Precision** cao |
| Phát hiện gian lận | Bỏ sót, nhưng báo nhầm quá nhiều cũng tốn nhân lực | F-beta, PR-AUC, precision ở recall cố định |
| Dữ liệu 99% âm, 1% dương | Accuracy vô nghĩa | PR-AUC, F1, MCC, recall |
| Cân bằng, hai lỗi tốn như nhau | – | Accuracy, F1, ROC-AUC |

```python
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

y_pred = (proba >= 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print("MCC:", matthews_corrcoef(y_test, y_pred))
```

### 3.4. Metric xác suất

Khi quan tâm chất lượng của **xác suất**, không chỉ nhãn:

| Metric | Ý nghĩa |
|---|---|
| **Log loss** (cross-entropy) | Phạt nặng khi tự tin sai; chính là loss khi huấn luyện |
| **Brier score** | MSE giữa xác suất và nhãn 0/1 |
| **Đường cong hiệu chỉnh** | Xác suất dự đoán có khớp tần suất thật không ([Chương 5](05-hoi-quy-va-phan-loai.md)) |

---

## 4. Đường cong ROC và precision-recall

Hai đường cong cho thấy hiệu năng **ở mọi ngưỡng**.

### 4.1. ROC

Vẽ **TPR (recall)** theo **FPR** khi quét ngưỡng. Đường chéo là đoán ngẫu nhiên. **ROC-AUC** là diện tích dưới đường cong, có cách hiểu xác suất: *xác suất mô hình cho mẫu dương một điểm cao hơn một mẫu âm ngẫu nhiên*. 0.5 là ngẫu nhiên, 1.0 là hoàn hảo.

### 4.2. Precision-Recall

Vẽ **precision** theo **recall**. **PR-AUC** (còn gọi average precision). Mốc ngẫu nhiên là tỉ lệ lớp dương (không phải 0.5).

### 4.3. Chọn ROC hay PR

- Dữ liệu cân bằng ⇒ hai đường tương tự nhau.
- **Dữ liệu rất lệch** ⇒ **PR phản ánh trung thực hơn**. ROC-AUC có thể vẫn cao (0.95) dù precision thực tế rất thấp, vì FPR được chia cho lượng âm khổng lồ nên nhỏ.

```python
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             RocCurveDisplay, PrecisionRecallDisplay)

print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC :", average_precision_score(y_test, proba))
print("Baseline PR-AUC:", y_test.mean())

RocCurveDisplay.from_predictions(y_test, proba)
PrecisionRecallDisplay.from_predictions(y_test, proba)
```

AUC **không phụ thuộc ngưỡng**, nhưng khi triển khai bạn vẫn phải chọn một ngưỡng cụ thể ([Chương 5](05-hoi-quy-va-phan-loai.md)).

---

## 5. Phân loại đa lớp, đa nhãn và xếp hạng

### 5.1. Đa lớp: cách lấy trung bình

Tính metric cho từng lớp rồi gộp:

| Cách | Ý nghĩa | Dùng khi |
|---|---|---|
| **Macro** | Trung bình đơn giản các lớp | Mọi lớp quan trọng như nhau, kể cả lớp hiếm |
| **Weighted** | Trung bình có trọng số theo số mẫu mỗi lớp | Muốn phản ánh phân phối thật |
| **Micro** | Gộp TP/FP/FN của mọi lớp rồi tính | Với đa lớp đơn nhãn, micro-F1 = accuracy |

```python
f1_score(y_true, y_pred, average="macro")
f1_score(y_true, y_pred, average="weighted")
confusion_matrix(y_true, y_pred)           # ma trận K x K cho thấy lớp nào hay bị nhầm với lớp nào
```

Với đa lớp, **nhìn ma trận nhầm lẫn** để biết cặp lớp nào bị lẫn.

### 5.2. Đa nhãn

Hamming loss (tỉ lệ nhãn dự đoán sai), F1 micro/macro theo nhãn, subset accuracy (đúng toàn bộ tập nhãn, rất nghiêm ngặt).

### 5.3. Xếp hạng và gợi ý

| Metric | Ý nghĩa |
|---|---|
| **Precision@k, Recall@k** | Trong $k$ kết quả đầu, bao nhiêu đúng / bắt được bao nhiêu |
| **MAP** | Trung bình của precision tại mỗi vị trí đúng |
| **NDCG** | Thưởng nhiều hơn khi kết quả liên quan nằm ở trên cao; hỗ trợ mức liên quan nhiều bậc |
| **MRR** | Nghịch đảo thứ hạng của kết quả đúng đầu tiên |

Các metric này sẽ gặp lại khi đánh giá truy xuất trong RAG.

---

## 6. Chiến lược kiểm định chéo

### 6.1. Hold-out

Chia một lần train/validation. Đơn giản, nhanh, nhưng kết quả phụ thuộc vào lần chia cụ thể và lãng phí dữ liệu. Phù hợp khi dữ liệu rất lớn.

### 6.2. K-fold cross-validation

Chia train thành $K$ phần (fold); lần lượt dùng mỗi phần làm validation, huấn luyện trên $K-1$ phần còn lại; lấy trung bình $K$ điểm. $K=5$ hoặc $10$ là phổ biến.

```python
from sklearn.model_selection import cross_val_score, cross_validate, KFold

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
print(f"RMSE: {-scores.mean():.3f} ± {scores.std():.3f}")

res = cross_validate(model, X_train, y_train, cv=5,
                     scoring=["r2", "neg_mean_absolute_error"], return_train_score=True)
```

### 6.3. Các biến thể

| Biến thể | Khi nào | Công cụ |
|---|---|---|
| **Stratified K-fold** | Phân loại, đặc biệt lệch: giữ tỉ lệ lớp mỗi fold | `StratifiedKFold` |
| **Repeated K-fold** | Dữ liệu ít, cần ước lượng ổn định hơn | `RepeatedKFold`, `RepeatedStratifiedKFold` |
| **Group K-fold** | Nhiều mẫu cùng một nhóm (bệnh nhân, khách hàng): nhóm không bị tách đôi | `GroupKFold`, `StratifiedGroupKFold` |
| **Time series split** | Dữ liệu thời gian: train luôn đứng trước validation | `TimeSeriesSplit` |
| **Leave-one-out (LOO)** | Dữ liệu cực ít; tốn kém, variance cao | `LeaveOneOut` |

```python
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = GroupKFold(n_splits=5)                    # dùng cùng tham số groups=...
cv = TimeSeriesSplit(n_splits=5)
```

### 6.4. Nested cross-validation

Nếu bạn vừa **tuning siêu tham số bằng CV** vừa **dùng chính điểm CV đó báo cáo**, điểm sẽ lạc quan (đã chọn cấu hình tốt nhất trên chính các fold đó). **Nested CV**: vòng ngoài đánh giá, vòng trong tuning.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

inner = GridSearchCV(model, param_grid, cv=3)        # vòng trong: chọn siêu tham số
outer_scores = cross_val_score(inner, X_train, y_train, cv=5)   # vòng ngoài: đánh giá
```

Trong thực tế, cách đơn giản hơn là tuning bằng CV trên train, rồi báo cáo **một lần** trên tập test độc lập.

### 6.5. Quy tắc quan trọng

- Mọi bước học từ dữ liệu (impute, scale, chọn đặc trưng, lấy mẫu lại) phải nằm **trong pipeline** để chỉ `fit` trên train của từng fold.
- Dùng cùng các fold khi so sánh nhiều mô hình (cố định `random_state`).

---

## 7. Overfitting, underfitting, bias-variance

### 7.1. Ba trạng thái

| | Lỗi train | Lỗi validation | Triệu chứng |
|---|---|---|---|
| **Underfit** | Cao | Cao | Mô hình quá đơn giản, chưa nắm được quy luật |
| **Vừa** | Thấp | Thấp, sát lỗi train | Mong muốn |
| **Overfit** | Rất thấp | Cao, cách xa lỗi train | Học thuộc nhiễu |

### 7.2. Phân rã bias-variance

Lỗi kỳ vọng trên dữ liệu mới:

$$\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\text{Bias}^2}_{\text{quá đơn giản}}+\underbrace{\text{Variance}}_{\text{nhạy với mẫu train}}+\underbrace{\sigma^2}_{\text{nhiễu không giảm được}}$$

- **Bias cao:** mô hình không đủ linh hoạt để biểu diễn quy luật (đường thẳng cho dữ liệu cong).
- **Variance cao:** mô hình thay đổi mạnh khi dữ liệu train đổi nhẹ (cây sâu, KNN với $k=1$).
- **Nhiễu $\sigma^2$:** giới hạn dưới không thể vượt.

Mô hình phức tạp hơn thì bias giảm, variance tăng. Mục tiêu: điểm cân bằng có tổng lỗi nhỏ nhất.

### 7.3. Cách xử lý

| Vấn đề | Cách khắc phục |
|---|---|
| **Underfit (bias cao)** | Mô hình mạnh hơn; thêm/tạo đặc trưng; giảm regularization; huấn luyện lâu hơn; bỏ giới hạn quá chặt (độ sâu cây) |
| **Overfit (variance cao)** | **Thêm dữ liệu**; mô hình đơn giản hơn; tăng regularization (L1/L2, dropout); bỏ đặc trưng nhiễu; early stopping; ensemble kiểu bagging; tăng cường dữ liệu (augmentation) |

Lưu ý: thêm dữ liệu giúp giảm variance nhưng **không giúp** khi vấn đề là bias.

---

## 8. Learning curve và validation curve

### 8.1. Learning curve (điểm theo kích thước tập train)

Cho biết thêm dữ liệu có giúp không, và đang bias hay variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

sizes, train_sc, val_sc = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="neg_root_mean_squared_error", n_jobs=-1,
)
plt.plot(sizes, -train_sc.mean(1), label="train")
plt.plot(sizes, -val_sc.mean(1), label="validation")
plt.xlabel("Kích thước tập train"); plt.ylabel("RMSE"); plt.legend()
```

| Hình dạng | Chẩn đoán | Hướng đi |
|---|---|---|
| Hai đường **gần nhau và cùng cao** | Bias cao (underfit) | Mô hình mạnh hơn, đặc trưng tốt hơn. Thêm dữ liệu không giúp |
| **Khoảng cách lớn** giữa hai đường, validation còn đang cải thiện | Variance cao (overfit) | **Thêm dữ liệu**, regularization |
| Hai đường hội tụ ở mức tốt | Đã ổn | Dừng tuning mô hình |

### 8.2. Validation curve (điểm theo một siêu tham số)

Thấy rõ vùng underfit, vùng tốt nhất, vùng overfit khi thay đổi độ phức tạp.

```python
from sklearn.model_selection import validation_curve

depths = np.arange(1, 21)
train_sc, val_sc = validation_curve(
    tree, X_train, y_train, param_name="max_depth", param_range=depths,
    cv=5, scoring="neg_root_mean_squared_error",
)
# Vẽ -train_sc.mean(1) và -val_sc.mean(1) theo depths: validation có đáy chữ U
```

---

## 9. So sánh mô hình có ý nghĩa thống kê

Mô hình A có RMSE 28.9, mô hình B có 29.3: A thật sự tốt hơn, hay do may rủi?

### 9.1. Nguyên tắc thực dụng

1. Báo cáo **trung bình ± độ lệch chuẩn** qua các fold với **cùng các fold**.
2. Nếu chênh lệch **nhỏ hơn** độ lệch chuẩn thì chưa thể kết luận.
3. Ưu tiên mô hình **đơn giản hơn** nếu hiệu năng tương đương (Occam).
4. Dùng **repeated CV** hoặc nhiều seed để giảm nhiễu khi hai mô hình sát nhau.

### 9.2. Các phép kiểm định

| Phép kiểm định | Dùng cho | Lưu ý |
|---|---|---|
| **Paired t-test trên điểm các fold** | So hai mô hình trên cùng fold | Các fold **không độc lập** (chia sẻ dữ liệu train) nên t-test thường **đánh giá thấp độ bất định**, cho kết luận quá dễ dãi |
| **Corrected resampled t-test** (Nadeau–Bengio) | Hiệu chỉnh cho việc chồng lấn dữ liệu | An toàn hơn t-test thường |
| **McNemar** | So hai bộ phân loại trên **cùng một tập test** (đếm các mẫu hai mô hình bất đồng) | Cần tập test đủ lớn |
| **Wilcoxon signed-rank** | Phi tham số, so điểm theo cặp | Ít giả định hơn t-test |
| **Bootstrap** | Khoảng tin cậy cho metric hoặc cho **hiệu** giữa hai mô hình | Linh hoạt, không cần giả định phân phối |

### 9.3. Khoảng tin cậy bằng bootstrap trên tập test

```python
import numpy as np
from sklearn.metrics import f1_score

def bootstrap_metric(y_true, y_pred, metric=f1_score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                  # lấy mẫu có hoàn lại
        stats.append(metric(y_true[idx], y_pred[idx]))
    return np.mean(stats), np.quantile(stats, [0.025, 0.975])

mean_f1, (lo, hi) = bootstrap_metric(y_test, y_pred)
print(f"F1 = {mean_f1:.3f}, khoảng tin cậy 95% [{lo:.3f}, {hi:.3f}]")
```

Khoảng tin cậy rộng cho thấy tập test nhỏ hoặc mô hình không ổn định; khi hai mô hình có khoảng tin cậy chồng lấn nhiều thì chưa đủ cơ sở nói cái nào hơn.

---

## 10. Phân tích lỗi

Con số tổng hợp che giấu cấu trúc lỗi. Sau khi có mô hình, **dành thời gian đọc các mẫu bị sai**.

### 10.1. Quy trình

1. Lấy các mẫu mô hình sai nhiều nhất (sai số lớn nhất, hoặc sai với độ tự tin cao).
2. Xem bằng mắt: có nhóm lỗi chung không? Nhãn sai? Đặc trưng thiếu? Dữ liệu bất thường?
3. Phân loại nguyên nhân, ước lượng tần suất mỗi nhóm, sửa nhóm đáng giá nhất trước.

```python
errors = X_val.copy()
errors["y"] = y_val
errors["pred"] = pred
errors["abs_err"] = (errors["y"] - errors["pred"]).abs()
errors.sort_values("abs_err", ascending=False).head(20)
```

### 10.2. Đánh giá theo lát cắt (slicing)

Điểm tổng thể tốt có thể che giấu một nhóm con rất tệ. Tách điểm theo nhóm có ý nghĩa:

```python
errors.groupby("district")["abs_err"].agg(["mean", "median", "count"])
```

Tách theo: vùng địa lý, nhóm khách hàng, khoảng giá trị nhãn, khoảng thời gian, thuộc tính nhạy cảm (kiểm tra công bằng). Nếu một nhóm có sai số cao hơn hẳn, cần thêm dữ liệu/đặc trưng cho nhóm đó, hoặc nhận ra giới hạn của mô hình.

### 10.3. Kiểm tra nhãn

Các mẫu mô hình dự đoán sai nhưng rất tự tin thường là **nhãn sai**. Sửa nhãn đôi khi hiệu quả hơn bất kỳ thay đổi mô hình nào.

---

## 11. Từ metric kỹ thuật đến giá trị kinh doanh

Metric tốt chưa chắc tạo ra giá trị. Nối chúng lại:

- **Quy đổi lỗi ra tiền:** FN tốn bao nhiêu? FP tốn bao nhiêu? Tính chi phí kỳ vọng tại từng ngưỡng và chọn ngưỡng tối thiểu hóa chi phí.
- **So với quy trình hiện tại** (luật cũ, con người), không chỉ so với 0.
- **Kiểm thử thực tế:** A/B test sau khi triển khai. Metric offline tốt không đảm bảo kết quả online tốt (phân phối dữ liệu đổi, người dùng phản ứng với mô hình).
- **Độ trễ, chi phí, độ ổn định, khả năng giải thích** cũng là tiêu chí, không chỉ độ chính xác.
- **Theo dõi sau triển khai:** hiệu năng giảm dần khi dữ liệu thay đổi (drift), xem [Chương 9](09-pipeline-tuning-va-trien-khai.md).

Bảng tóm tắt nhanh "chọn metric":

| Bài toán | Metric chính | Metric phụ |
|---|---|---|
| Hồi quy, sai số lớn rất tệ | RMSE | MAE, R² |
| Hồi quy, có outlier | MAE | Median AE |
| Hồi quy giá, quan tâm tỉ lệ | RMSLE | MAPE |
| Phân loại cân bằng | Accuracy, F1 | ROC-AUC |
| Phân loại lệch | PR-AUC, F1, recall@precision | MCC |
| Cần xác suất tốt | Log loss, Brier | Đường cong hiệu chỉnh |
| Xếp hạng/tìm kiếm | NDCG, MAP, precision@k | MRR |
| Gom cụm | Silhouette + đánh giá theo mục đích | ARI (nếu có nhãn tham chiếu) |
| Phát hiện bất thường | PR-AUC trên tập có nhãn, precision@k | – |

---

**Trước:** [Chương 2](02-du-lieu-va-feature-engineering.md) | **Tiếp theo:** [Chương 4: Toán nền tảng và tối ưu hóa](04-toan-nen-tang-va-toi-uu.md)
