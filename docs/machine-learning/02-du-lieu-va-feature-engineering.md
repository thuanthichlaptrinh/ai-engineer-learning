# Chương 2: Dữ liệu và feature engineering

"Garbage in, garbage out." Phần lớn chất lượng của một hệ thống ML đến từ dữ liệu và đặc trưng, không phải thuật toán.

> **Phạm vi:** chương này dùng cho **mọi kiểu học**. Các phần về nhãn, chia train/validation/test, rò rỉ nhãn và mất cân bằng lớp chỉ áp dụng khi có nhãn (học có giám sát). Học không giám sát vẫn cần xử lý giá trị thiếu, outlier, mã hóa và đặc biệt là chuẩn hóa (xem [chương 8](08-hoc-khong-giam-sat.md)).

## Mục lục

1. [Khám phá dữ liệu (EDA)](#1-khám-phá-dữ-liệu-eda)
2. [Các kiểu biến](#2-các-kiểu-biến)
3. [Chia dữ liệu](#3-chia-dữ-liệu)
4. [Giá trị thiếu](#4-giá-trị-thiếu)
5. [Outlier](#5-outlier)
6. [Mã hóa biến phân loại](#6-mã-hóa-biến-phân-loại)
7. [Chuẩn hóa và biến đổi đặc trưng](#7-chuẩn-hóa-và-biến-đổi-đặc-trưng)
8. [Tạo đặc trưng](#8-tạo-đặc-trưng)
9. [Chọn đặc trưng](#9-chọn-đặc-trưng)
10. [Rò rỉ dữ liệu](#10-rò-rỉ-dữ-liệu-data-leakage)
11. [Dữ liệu mất cân bằng](#11-dữ-liệu-mất-cân-bằng)
12. [Chất lượng nhãn và thiên lệch dữ liệu](#12-chất-lượng-nhãn-và-thiên-lệch-dữ-liệu)
13. [Cần bao nhiêu dữ liệu](#13-cần-bao-nhiêu-dữ-liệu)
14. [Checklist chất lượng dữ liệu](#14-checklist-chất-lượng-dữ-liệu)

---

## 1. Khám phá dữ liệu (EDA)

Mục tiêu: hiểu dữ liệu **trước khi** dựng mô hình.

```python
import pandas as pd

df = pd.read_csv("data/housing.csv")

df.shape                                   # số hàng, số cột
df.info()                                  # kiểu dữ liệu, số giá trị khác null
df.head()
df.describe(include="all")                 # thống kê số và phân loại
df.isna().mean().sort_values(ascending=False)   # tỉ lệ thiếu theo cột
df.duplicated().sum()                      # số hàng trùng
df.nunique().sort_values()                 # số giá trị khác nhau mỗi cột
```

### 1.1. Phân tích từng biến (univariate)

- Biến số: histogram, boxplot, độ lệch (skewness). Có lệch phải nặng không? Có giá trị bất thường?
- Biến phân loại: đếm tần suất. Có lớp hiếm không? Có giá trị viết sai chính tả (`"HN"`, `"Hà Nội"`, `"ha noi"`)?
- **Nhãn:** phân phối của nhãn là việc đầu tiên cần xem. Nhãn hồi quy lệch thì cân nhắc log. Nhãn phân loại lệch thì xem mục 11.

### 1.2. Phân tích quan hệ (bivariate, multivariate)

```python
import seaborn as sns
import matplotlib.pyplot as plt

df.corr(numeric_only=True)["price"].sort_values()      # tương quan với nhãn
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
sns.scatterplot(data=df, x="area", y="price", hue="district")
df.groupby("district")["price"].agg(["mean", "median", "count"])
```

Lưu ý: tương quan Pearson chỉ bắt quan hệ tuyến tính. Quan hệ hình chữ U có tương quan gần 0. Hãy luôn **vẽ hình**, đừng chỉ nhìn con số. Và **tương quan không phải nhân quả**.

### 1.3. Những điều cần trả lời sau EDA

- Có bao nhiêu mẫu, bao nhiêu đặc trưng? Dữ liệu có vừa với bộ nhớ không?
- Cột nào thiếu nhiều? Thiếu có theo quy luật không?
- Có hàng trùng, giá trị vô lý (tuổi âm, diện tích 0) không?
- Biến nào lệch mạnh, có outlier thật sự hay lỗi nhập liệu?
- Có cột nào có thể làm rò rỉ nhãn không? (mục 10)
- Dữ liệu có phản ánh đúng tình huống khi triển khai thật không? (thời gian, khu vực, nhóm người dùng)

---

## 2. Các kiểu biến

| Kiểu | Mô tả | Ví dụ | Xử lý thường gặp |
|---|---|---|---|
| **Số liên tục** | Giá trị thực trong khoảng | Giá, diện tích | Chuẩn hóa, log nếu lệch |
| **Số rời rạc** (đếm) | Số nguyên | Số phòng, số lần mua | Giữ nguyên hoặc coi như phân loại nếu ít giá trị |
| **Phân loại danh nghĩa** (nominal) | Danh mục không thứ tự | Màu, thành phố | One-hot, target encoding |
| **Phân loại thứ tự** (ordinal) | Có thứ tự | Nhỏ/Vừa/Lớn, 1–5 sao | Ordinal encoding |
| **Nhị phân** | Hai giá trị | Có/không | 0/1 |
| **Ngày giờ** | Mốc thời gian | Ngày đăng tin | Tách năm, tháng, thứ, giờ; tính khoảng thời gian |
| **Văn bản** | Chuỗi tự do | Mô tả sản phẩm | TF-IDF, embedding |
| **ID** | Định danh | Mã khách hàng | Thường bỏ (nguy cơ gây overfit), hoặc dùng để nhóm |

---

## 3. Chia dữ liệu

### 3.1. Vai trò ba tập

| Tập | Mục đích | Tỉ lệ gợi ý |
|---|---|---|
| **Train** | Học tham số | 60–80% |
| **Validation** | Chọn mô hình, tuning siêu tham số | 10–20% (hoặc thay bằng cross-validation) |
| **Test** | Đánh giá cuối cùng, **dùng một lần** | 10–20% |

Nếu dùng test để chọn mô hình hoặc tuning, test không còn là ước lượng khách quan.

### 3.2. Các cách chia

```python
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# Ngẫu nhiên (hồi quy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Phân tầng: giữ tỉ lệ lớp (phân loại, đặc biệt dữ liệu lệch)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Theo nhóm: các mẫu cùng nhóm không bị tách đôi
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["patient_id"]))

# Theo thời gian: train trước, test sau
cutoff = "2024-01-01"
train = df[df["date"] < cutoff]
test = df[df["date"] >= cutoff]
```

| Tình huống | Cách chia |
|---|---|
| Mẫu độc lập, nhãn cân bằng | Ngẫu nhiên |
| Nhãn lệch | Phân tầng (`stratify`) |
| Nhiều mẫu của cùng một bệnh nhân, khách hàng, thiết bị | Theo nhóm (`GroupKFold`, `GroupShuffleSplit`) |
| Dữ liệu thời gian, dự báo tương lai | Theo thời gian (`TimeSeriesSplit`), **không xáo trộn** |

Chia sai là nguồn rò rỉ phổ biến nhất: nếu hai ảnh của cùng một bệnh nhân rơi vào cả train và test, mô hình "nhớ" bệnh nhân và điểm test đẹp giả.

---

## 4. Giá trị thiếu

### 4.1. Ba cơ chế thiếu

| Cơ chế | Ý nghĩa | Ví dụ |
|---|---|---|
| **MCAR** (thiếu hoàn toàn ngẫu nhiên) | Thiếu không liên quan gì đến dữ liệu | Cảm biến hỏng ngẫu nhiên |
| **MAR** (thiếu ngẫu nhiên có điều kiện) | Thiếu phụ thuộc vào các biến **đã quan sát** | Người trẻ hay bỏ trống mục thu nhập |
| **MNAR** (thiếu không ngẫu nhiên) | Thiếu phụ thuộc vào chính giá trị bị thiếu | Người thu nhập cao né không khai |

MNAR nguy hiểm nhất: việc "thiếu" tự nó mang thông tin, và điền bừa sẽ làm sai lệch.

### 4.2. Cách xử lý

| Cách | Khi nào | Lưu ý |
|---|---|---|
| Xóa hàng | Thiếu ít (<5%) và ngẫu nhiên | Mất dữ liệu, có thể sinh thiên lệch |
| Xóa cột | Thiếu quá nhiều (>50–60%) và ít giá trị | Cân nhắc cột đó có quan trọng không |
| Điền median / mean | Biến số | Median bền với outlier hơn |
| Điền mode / giá trị `"unknown"` | Biến phân loại | Nhóm `"unknown"` riêng thường tốt hơn mode |
| Điền theo nhóm | Thiếu phụ thuộc nhóm | Ví dụ điền theo median của từng quận |
| KNN imputer / Iterative imputer | Quan hệ giữa các biến mạnh | Chậm, dễ rò rỉ nếu không nằm trong pipeline |
| **Thêm cột cờ** `is_missing` | Việc thiếu có thể mang thông tin | Dùng kèm các cách trên |
| Để nguyên | Mô hình tự xử lý (LightGBM, XGBoost, HistGradientBoosting) | Không dùng được với mô hình tuyến tính, SVM, KNN |

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

SimpleImputer(strategy="median", add_indicator=True)   # add_indicator thêm cột cờ
KNNImputer(n_neighbors=5)
IterativeImputer(random_state=42)
```

**Quy tắc vàng:** thống kê điền (median, mean) phải tính **chỉ trên tập train**, rồi áp sang validation/test. Dùng `Pipeline` để tự động đảm bảo.

---

## 5. Outlier

Outlier là điểm khác biệt rõ rệt với phần còn lại. Có hai loại: **lỗi** (nhập sai, cảm biến hỏng) nên sửa hoặc bỏ, và **thật nhưng hiếm** (căn biệt thự giữa khu nhà phố) nên giữ vì chứa thông tin.

### 5.1. Phát hiện

| Cách | Quy tắc |
|---|---|
| **IQR** | Ngoài $[Q_1 - 1.5\,\mathrm{IQR},\; Q_3 + 1.5\,\mathrm{IQR}]$ |
| **Z-score** | $\lvert z\rvert > 3$ (chỉ hợp lý khi phân phối gần chuẩn) |
| **Trực quan** | Boxplot, scatter plot |
| **Mô hình** | Isolation Forest, LOF (xem [Chương 8](08-hoc-khong-giam-sat.md)) |

```python
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
mask = df["price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
df_clean = df[mask]
```

### 5.2. Xử lý

- **Xóa** nếu là lỗi rõ ràng.
- **Cắt (winsorize/clip):** gán giá trị vượt ngưỡng về ngưỡng (ví dụ phân vị 1% và 99%).
- **Biến đổi:** log làm giảm ảnh hưởng của giá trị cực lớn.
- **Đổi mô hình hoặc loss:** mô hình cây ít nhạy outlier, Huber/MAE bền hơn MSE.
- **Giữ nguyên** nếu outlier là tín hiệu (phát hiện gian lận thì outlier chính là mục tiêu).

Tránh xóa outlier chỉ để điểm đẹp. Hãy hỏi: ở thời điểm triển khai thật, có gặp những mẫu như vậy không? Nếu có thì mô hình phải xử lý được chúng.

---

## 6. Mã hóa biến phân loại

Hầu hết mô hình cần đầu vào số.

| Cách | Mô tả | Khi nào | Nhược điểm |
|---|---|---|---|
| **One-hot** | Mỗi giá trị thành một cột 0/1 | Ít giá trị (<~20), không thứ tự | Bùng nổ số cột khi nhiều giá trị |
| **Ordinal** | Gán số theo thứ tự | Biến có thứ tự thật | Gán thứ tự bừa cho biến danh nghĩa sẽ gây sai |
| **Label encoding** | Gán số tùy ý | Chỉ cho nhãn, hoặc mô hình cây | Mô hình tuyến tính sẽ hiểu sai thành quan hệ thứ tự |
| **Frequency encoding** | Thay bằng tần suất xuất hiện | Nhiều giá trị | Hai giá trị cùng tần suất bị lẫn |
| **Target encoding** | Thay bằng trung bình nhãn của từng giá trị | Rất nhiều giá trị (mã bưu điện) | **Dễ rò rỉ nhãn**, phải tính trong cross-validation và làm mượt |
| **Hashing** | Băm giá trị vào số cột cố định | Cực nhiều giá trị, dữ liệu luồng | Va chạm, mất khả năng diễn giải |
| **Embedding** | Vector học được | Mạng neural, nhiều giá trị | Cần DL |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OrdinalEncoder(categories=[["small", "medium", "large"]])
TargetEncoder(smooth="auto")        # scikit-learn >= 1.3, có cross-fitting nội bộ
```

Hai chi tiết thực tế:
- Giá trị **chưa từng thấy** lúc train: dùng `handle_unknown="ignore"` để không lỗi khi triển khai.
- **One-hot với mô hình tuyến tính** có thể bỏ một cột (`drop="first"`) để tránh đa cộng tuyến hoàn hảo, nhưng nếu có regularization thì không bắt buộc.

LightGBM và CatBoost xử lý được biến phân loại trực tiếp, thường tốt hơn one-hot khi nhiều giá trị.

---

## 7. Chuẩn hóa và biến đổi đặc trưng

### 7.1. Chuẩn hóa thang đo

| Bộ chuẩn hóa | Công thức | Dùng khi |
|---|---|---|
| **StandardScaler** | $(x-\mu)/\sigma$ | Mặc định cho mô hình tuyến tính, SVM, KNN, PCA, mạng neural |
| **MinMaxScaler** | $(x-\min)/(\max-\min)$ | Cần miền $[0,1]$ (ảnh, mạng neural); nhạy với outlier |
| **RobustScaler** | $(x-\text{median})/\mathrm{IQR}$ | Dữ liệu có nhiều outlier |
| **MaxAbsScaler** | $x/\max\lvert x\rvert$ | Dữ liệu thưa, giữ nguyên số 0 |

**Mô hình nào cần chuẩn hóa?** Cần: mô hình dựa trên khoảng cách hoặc gradient/regularization (KNN, K-Means, SVM, PCA, Linear/Logistic có regularization, mạng neural). Không cần: cây, Random Forest, gradient boosting.

### 7.2. Biến đổi phân phối

Phân phối lệch phải (giá, thu nhập, số lượt xem) khiến mô hình tuyến tính kém. Biến đổi để gần phân phối chuẩn hơn:

| Cách | Ghi chú |
|---|---|
| `np.log1p(x)` | Cho giá trị $\ge 0$, hay dùng nhất |
| Căn bậc hai | Nhẹ hơn log |
| **Box-Cox** | Chỉ cho giá trị $>0$ |
| **Yeo-Johnson** | Cho cả giá trị âm |
| **QuantileTransformer** | Ép về phân phối đều hoặc chuẩn |

```python
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
PowerTransformer(method="yeo-johnson")
QuantileTransformer(output_distribution="normal")
```

Nếu biến đổi **nhãn** (ví dụ dự đoán `log(price)`), nhớ biến đổi ngược khi báo cáo. Dùng `TransformedTargetRegressor` để tự động.

### 7.3. Rời rạc hóa (binning)

Chia biến số thành các khoảng (nhóm tuổi). Hữu ích khi quan hệ với nhãn không đơn điệu và dùng mô hình tuyến tính. Tránh khi dùng mô hình cây, vì cây tự tìm được ngưỡng.

---

## 8. Tạo đặc trưng

Đặc trưng tốt đưa tri thức miền vào dữ liệu. Thường cải thiện nhiều hơn việc đổi thuật toán.

### 8.1. Các kỹ thuật

| Kỹ thuật | Ví dụ |
|---|---|
| **Tỉ lệ, hiệu** | Giá trên mét vuông, thu nhập trừ chi tiêu |
| **Tương tác** | `diện tích × số tầng`; mô hình tuyến tính cần, cây tự tìm phần nào |
| **Đa thức** | $x^2$, $x^3$ qua `PolynomialFeatures` |
| **Ngày giờ** | Năm, tháng, thứ trong tuần, giờ, cuối tuần?, ngày lễ?, số ngày kể từ sự kiện |
| **Tính tuần hoàn** | Giờ trong ngày: dùng $\sin(2\pi h/24)$ và $\cos(2\pi h/24)$ để 23h gần 0h |
| **Tổng hợp theo nhóm** | Giá trung bình của quận, số lần mua trong 30 ngày qua |
| **Đếm / tần suất** | Số lần xuất hiện của giá trị |
| **Khoảng cách** | Khoảng cách tới trung tâm, từ tọa độ |
| **Văn bản** | Độ dài, số từ, TF-IDF |
| **Từ mô hình khác** | Nhãn cụm K-Means, thành phần PCA |

```python
import numpy as np

df["price_per_m2"] = df["price"] / df["area"]          # cẩn thận: dùng nhãn là rò rỉ!
df["age"] = df["sold_year"] - df["built_year"]
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["district_mean_area"] = df.groupby("district")["area"].transform("mean")
```

Dòng đầu là ví dụ **sai**: `price_per_m2` dùng chính nhãn `price`, nên không có sẵn lúc dự đoán thật.

### 8.2. Quy trình thêm đặc trưng

1. Đặt giả thuyết dựa trên hiểu biết miền ("nhà gần trung tâm đắt hơn").
2. Tạo đặc trưng.
3. Đo bằng cross-validation, so với trước khi thêm.
4. Giữ nếu cải thiện có ý nghĩa, bỏ nếu không.

Với đặc trưng tổng hợp theo nhóm hoặc dùng nhãn (target encoding), phải tính **bên trong từng fold** để không rò rỉ.

---

## 9. Chọn đặc trưng

Nhiều đặc trưng không có nghĩa là tốt hơn: tăng nhiễu, tăng overfit, chậm hơn, khó diễn giải.

| Nhóm | Cách | Ví dụ |
|---|---|---|
| **Filter** (độc lập với mô hình) | Chấm điểm từng đặc trưng bằng thống kê | Loại phương sai thấp, tương quan cao với đặc trưng khác, mutual information, chi-bình phương, ANOVA F |
| **Wrapper** (dùng mô hình để đánh giá) | Thử các tập con, chọn tập tốt nhất | RFE (loại dần đặc trưng yếu nhất), chọn tiến/lùi |
| **Embedded** (nằm trong quá trình học) | Mô hình tự chọn | Lasso (đưa trọng số về 0), độ quan trọng của cây |

```python
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif, RFE)
from sklearn.linear_model import LogisticRegression

VarianceThreshold(threshold=0.0)                       # bỏ cột hằng
SelectKBest(mutual_info_classif, k=20)                 # giữ 20 cột có MI cao nhất
RFE(LogisticRegression(max_iter=1000), n_features_to_select=15)
```

Quy tắc: chọn đặc trưng cũng là bước "học từ dữ liệu", nên phải nằm **trong pipeline và cross-validation**, nếu không sẽ rò rỉ.

Hai đặc trưng tương quan rất cao (>0.95) mang thông tin gần như trùng nhau: có thể bỏ một. Với mô hình tuyến tính, đa cộng tuyến làm hệ số mất ổn định (kiểm tra bằng VIF, xem [Chương 5](05-hoi-quy-va-phan-loai.md)).

---

## 10. Rò rỉ dữ liệu (data leakage)

Rò rỉ là khi thông tin không có sẵn lúc dự đoán thật lọt vào lúc huấn luyện hoặc đánh giá. Hậu quả: điểm đánh giá đẹp giả tạo, nhưng mô hình thất bại khi triển khai.

### 10.1. Các dạng

| Dạng | Mô tả | Ví dụ |
|---|---|---|
| **Rò rỉ mục tiêu** (target leakage) | Đặc trưng chứa hoặc suy ra từ nhãn | Dự đoán khách có hủy dịch vụ nhưng dùng cột "lý do hủy" hoặc "ngày hủy" |
| **Rò rỉ thời gian** | Dùng thông tin từ tương lai | Dự đoán giá cổ phiếu hôm nay bằng giá ngày mai; chia ngẫu nhiên dữ liệu chuỗi thời gian |
| **Ô nhiễm train-test** | Xử lý bằng thống kê của cả tập | Fit scaler/imputer/PCA/chọn đặc trưng trên toàn bộ dữ liệu rồi mới chia |
| **Rò rỉ nhóm** | Mẫu cùng nhóm nằm ở cả train và test | Nhiều ảnh của cùng một bệnh nhân |
| **Trùng lặp** | Hàng giống nhau ở cả train và test | Dữ liệu chưa khử trùng |
| **Lấy mẫu lại sai chỗ** | Oversample/SMOTE trước khi chia | Bản sao của cùng một mẫu rơi vào cả hai tập |
| **Tuning trên test** | Chọn siêu tham số theo điểm test | Lặp đi lặp lại "thử xem test thế nào" |

### 10.2. Dấu hiệu nghi ngờ

- Điểm gần hoàn hảo trên bài toán vốn khó.
- Một đặc trưng có độ quan trọng áp đảo, cao bất thường.
- Điểm CV rất cao nhưng điểm trên dữ liệu mới thật thấp hơn hẳn.

### 10.3. Cách phòng

1. Với mỗi cột, hỏi: **"Lúc dự đoán thật, mình có giá trị này không?"**
2. Chia dữ liệu **trước** mọi bước học từ dữ liệu.
3. Dùng `Pipeline` để mọi bước chỉ `fit` trên train của từng fold.
4. Chia theo nhóm hoặc theo thời gian khi dữ liệu có cấu trúc đó.
5. Khử trùng lặp trước khi chia.
6. Giữ test ở một nơi riêng, chỉ mở một lần.

---

## 11. Dữ liệu mất cân bằng

Khi một lớp hiếm (gian lận 0.1%, bệnh hiếm 1%).

**Vấn đề:** mô hình đoán toàn lớp đa số đã đạt accuracy 99%, nhưng vô dụng.

### 11.1. Các bước xử lý, theo thứ tự nên thử

1. **Đổi metric:** PR-AUC, F1, recall/precision theo hậu quả; không dùng accuracy.
2. **Phân tầng khi chia:** `stratify=y`, `StratifiedKFold`.
3. **Class weight:** phạt nặng hơn khi sai lớp hiếm. Đơn giản và thường hiệu quả.
4. **Chỉnh ngưỡng quyết định:** thay vì cố định 0.5, chọn ngưỡng dựa trên đường precision-recall và chi phí lỗi.
5. **Lấy mẫu lại:** undersample lớp đa số, oversample lớp hiếm, hoặc SMOTE (tạo mẫu tổng hợp giữa các láng giềng của lớp hiếm).
6. **Mô hình/hàm loss chuyên biệt:** focal loss, ensemble cân bằng (BalancedRandomForest).
7. **Phát hiện bất thường** nếu lớp hiếm quá hiếm hoặc không có nhãn đáng tin.

```python
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Class weight
clf = LogisticRegression(class_weight="balanced", max_iter=1000)

# SMOTE đặt TRONG pipeline: chỉ áp dụng lên tập train của mỗi fold
pipe = ImbPipeline([
    ("pre", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000)),
])
```

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val, proba_val)
# Chọn ngưỡng đạt recall mong muốn với precision cao nhất có thể
```

**Cảnh báo:** chỉ lấy mẫu lại trên **tập train của từng fold**. Tập validation/test giữ nguyên phân phối thật, vì đó mới là điều bạn sẽ gặp khi triển khai.

Lưu ý: oversample hay đổi trọng số làm **xác suất đầu ra bị lệch** (không còn là xác suất thật). Nếu cần xác suất đáng tin, hiệu chỉnh lại (calibration, xem [Chương 5](05-hoi-quy-va-phan-loai.md)).

---

## 12. Chất lượng nhãn và thiên lệch dữ liệu

### 12.1. Nhãn nhiễu

Nhãn sai (do gán nhầm, nhãn tự động sai) giới hạn mức độ tốt tối đa của mô hình. Cách xử lý:
- Gán nhãn bởi nhiều người và đo mức đồng thuận (Cohen's kappa).
- Xem lại các mẫu mô hình dự đoán sai với độ tự tin cao: thường là lỗi nhãn.
- Dùng mô hình/loss bền với nhiễu nhãn (ví dụ Huber), hoặc công cụ như `cleanlab`.

### 12.2. Thiên lệch dữ liệu

| Loại | Mô tả |
|---|---|
| **Thiên lệch chọn mẫu** | Dữ liệu không đại diện cho quần thể thật (khảo sát chỉ qua mạng) |
| **Thiên lệch người sống sót** | Chỉ thấy những trường hợp "sống sót" (chỉ phân tích công ty còn tồn tại) |
| **Thiên lệch đo lường** | Công cụ đo sai lệch theo hệ thống |
| **Thiên lệch lịch sử** | Dữ liệu phản ánh định kiến quá khứ (tuyển dụng, cho vay) |
| **Thiên lệch phản hồi** | Mô hình ảnh hưởng dữ liệu tương lai (chỉ gợi ý món A thì chỉ thu được dữ liệu về A) |

Mô hình học y những gì dữ liệu chứa. Kiểm tra bằng cách tách điểm đánh giá theo nhóm con (giới tính, vùng, độ tuổi...) và để ý chênh lệch.

---

## 13. Cần bao nhiêu dữ liệu

Không có con số cố định, nhưng có hướng dẫn thực tế:

- Mô hình càng phức tạp, càng nhiều tham số thì càng cần nhiều dữ liệu.
- Mô hình tuyến tính: vài trăm đến vài nghìn mẫu thường đã dùng được cho dữ liệu bảng.
- Gradient boosting: từ vài nghìn mẫu trở lên cho kết quả tốt.
- Mạng neural từ đầu: hàng chục nghìn đến hàng triệu (hoặc dùng transfer learning).
- Quy tắc kinh nghiệm cũ: ít nhất 10 mẫu cho mỗi đặc trưng cho mô hình tuyến tính, và 10–20 mẫu của lớp hiếm cho mỗi đặc trưng trong phân loại.

**Cách kiểm chứng bằng thực nghiệm:** vẽ **learning curve** (điểm theo kích thước tập train). Nếu điểm validation còn đang tăng khi thêm dữ liệu thì thêm dữ liệu sẽ giúp; nếu đã phẳng thì thêm dữ liệu không giúp, cần đặc trưng hoặc mô hình tốt hơn (xem [Chương 3](03-danh-gia-mo-hinh.md)).

---

## 14. Checklist chất lượng dữ liệu

Trước khi dựng mô hình, đánh dấu:

- [ ] Hiểu ý nghĩa từng cột và đơn vị (có từ điển dữ liệu)
- [ ] Đã xem phân phối nhãn
- [ ] Đã kiểm tra hàng trùng, giá trị vô lý, kiểu dữ liệu sai
- [ ] Biết cơ chế thiếu dữ liệu và có chiến lược xử lý
- [ ] Đã quyết định cách xử lý outlier (và có lý do)
- [ ] Chia dữ liệu đúng kiểu (ngẫu nhiên, phân tầng, nhóm, thời gian) **trước** mọi bước học từ dữ liệu
- [ ] Mọi bước tiền xử lý nằm trong `Pipeline`
- [ ] Đã rà từng đặc trưng về rò rỉ: "lúc dự đoán thật có giá trị này không?"
- [ ] Dữ liệu train đại diện cho tình huống triển khai
- [ ] Có baseline để so sánh
- [ ] Đã ghi lại phiên bản dữ liệu và nguồn gốc

---

**Trước:** [Chương 1](01-tong-quan-va-cac-kieu-hoc.md) | **Tiếp theo:** [Chương 3: Đánh giá mô hình](03-danh-gia-mo-hinh.md)
