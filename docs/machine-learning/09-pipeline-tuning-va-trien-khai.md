# Chương 9: Pipeline, tuning, giải thích và triển khai

Từ notebook thử nghiệm đến hệ thống đáng tin: đóng gói đúng, tuning có kỷ luật, giải thích được, và đưa ra chạy thật.

## Mục lục

1. [Pipeline trong scikit-learn](#1-pipeline-trong-scikit-learn)
2. [Tuning siêu tham số](#2-tuning-siêu-tham-số)
3. [Thiết kế thí nghiệm](#3-thiết-kế-thí-nghiệm)
4. [Theo dõi thí nghiệm và tái lập](#4-theo-dõi-thí-nghiệm-và-tái-lập)
5. [Giải thích mô hình](#5-giải-thích-mô-hình)
6. [Công bằng và thiên lệch](#6-công-bằng-và-thiên-lệch)
7. [Lưu và nạp mô hình](#7-lưu-và-nạp-mô-hình)
8. [Triển khai thành API](#8-triển-khai-thành-api)
9. [Giám sát sau triển khai](#9-giám-sát-sau-triển-khai)
10. [Cấu trúc dự án và kiểm thử](#10-cấu-trúc-dự-án-và-kiểm-thử)

---

## 1. Pipeline trong scikit-learn

`Pipeline` nối các bước tiền xử lý và mô hình thành **một đối tượng**. Lợi ích:

- **Chống rò rỉ:** mọi bước `fit` chỉ trên train của từng fold khi cross-validation.
- **Một lệnh** `fit`/`predict` cho cả quy trình, dễ lưu và triển khai.
- **Tuning** được cả tham số tiền xử lý lẫn mô hình trong cùng một tìm kiếm.

### 1.1. Pipeline đầy đủ cho dữ liệu hỗn hợp

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
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
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
print(f"RMSE: {-scores.mean():.1f} ± {scores.std():.1f}")
```

### 1.2. Bộ biến đổi tự viết

Khi cần bước riêng (tạo đặc trưng theo nghiệp vụ), viết một transformer để vẫn nằm trong pipeline:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class HouseFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age"] = X["sold_year"] - X["built_year"]
        X["rooms_per_area"] = X["rooms"] / X["area"]
        X["log_area"] = np.log1p(X["area"])
        return X

model = Pipeline([("features", HouseFeatures()), ("preprocess", preprocess),
                  ("model", RandomForestRegressor(random_state=42))])
```

Hoặc dùng `FunctionTransformer` cho hàm đơn giản không có trạng thái.

### 1.3. Xem kết quả từng bước

```python
model.fit(X_train, y_train)
feature_names = model.named_steps["preprocess"].get_feature_names_out()
model.set_output(transform="pandas")         # đầu ra các bước thành DataFrame (sklearn >= 1.2)
```

---

## 2. Tuning siêu tham số

Thứ tự ưu tiên: **dữ liệu và đặc trưng trước, mô hình sau, tuning sau cùng.** Tuning thường chỉ thêm vài phần trăm; làm sạch dữ liệu và thêm đặc trưng tốt thường hơn hẳn.

### 2.1. Các phương pháp

| Phương pháp | Cách làm | Khi nào |
|---|---|---|
| **Grid search** | Thử mọi tổ hợp trong lưới | Ít tham số, ít giá trị |
| **Random search** | Lấy ngẫu nhiên tổ hợp từ phân phối | Mặc định tốt: hiệu quả hơn grid với cùng ngân sách, vì không phí công thử đều các tham số ít quan trọng |
| **Successive halving** | Thử nhiều cấu hình với ít tài nguyên, giữ lại cấu hình tốt rồi tăng tài nguyên | Muốn nhanh (`HalvingRandomSearchCV`) |
| **Bayesian optimization (Optuna)** | Dùng kết quả các lần thử trước để chọn lần thử hứa hẹn tiếp theo; có pruning dừng sớm thử nghiệm tệ | **Khuyến nghị**, nhất là boosting và mạng neural |

### 2.2. Random search với scikit-learn

```python
from scipy.stats import randint, loguniform
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "model__n_estimators": randint(200, 800),
    "model__max_depth": randint(4, 30),
    "model__min_samples_leaf": randint(1, 20),
    "model__max_features": [0.3, 0.5, 0.7, 1.0],
}
search = RandomizedSearchCV(
    model, param_dist, n_iter=40, cv=5,
    scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42,
)
search.fit(X_train, y_train)
print(search.best_params_, -search.best_score_)
```

(Tên tham số trong pipeline có dạng `tên_bước__tham_số`.)

### 2.3. Optuna

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

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=60)
print(study.best_params, study.best_value)
```

### 2.4. Nguyên tắc tuning

- **Tune trên validation/CV, không bao giờ trên test.**
- Dùng thang **log** cho tham số trải nhiều bậc (learning rate, `C`, `alpha`).
- Bắt đầu với **khoảng rộng, ít thử**, rồi thu hẹp quanh vùng tốt.
- Nhớ rằng nhiều lần thử trên cùng validation làm điểm **lạc quan** (overfit vào validation); giữ test độc lập để xác nhận.
- Đặt **ngân sách** (số thử, thời gian) trước; dừng khi hiệu quả giảm dần.
- Sau khi chốt, **huấn luyện lại trên toàn bộ train** với cấu hình tốt nhất, rồi đánh giá **một lần** trên test.

---

## 3. Thiết kế thí nghiệm

1. **Baseline trước:** ghi điểm của mô hình đơn giản nhất làm mốc.
2. **Mỗi lần đổi một thứ,** để biết điều gì tạo ra khác biệt.
3. **Ablation study:** bỏ từng thành phần (một nhóm đặc trưng, một bước tiền xử lý) và đo lại để biết nó đóng góp bao nhiêu.
4. **Cố định seed và các fold** khi so sánh.
5. **Báo cáo trung bình ± độ lệch chuẩn**; chênh nhỏ hơn độ lệch chuẩn xem như hòa.
6. **Ghi chép mọi thí nghiệm:** dữ liệu (phiên bản), đặc trưng, tham số, điểm, ngày, nhận xét.

Mẫu bảng kết quả:

| # | Mô hình | Đặc trưng | RMSE CV (± std) | Ghi chú |
|---|---|---|---|---|
| 0 | Baseline (trung bình) | – | 78 000 ± 2 100 | Mốc |
| 1 | Ridge | gốc | 52 000 ± 1 800 | |
| 2 | Random Forest | gốc | 31 500 ± 1 200 | Phi tuyến giúp nhiều |
| 3 | LightGBM | gốc | 29 800 ± 1 100 | |
| 4 | LightGBM | gốc + tỉ lệ + tuổi nhà | 28 900 ± 1 100 | Đặc trưng mới giúp ích |
| 5 | LightGBM đã tuning | như trên | 28 400 ± 1 000 | Tuning cải thiện nhỏ |

Giữ nhật ký các **giả thuyết đã thử và thất bại**: ngăn bạn lặp lại, và là thứ có giá trị khi viết báo cáo.

---

## 4. Theo dõi thí nghiệm và tái lập

### 4.1. MLflow

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("house-price")
with mlflow.start_run(run_name="lgbm-tuned"):
    mlflow.log_params(study.best_params)
    mlflow.log_metric("rmse_cv", study.best_value)
    mlflow.log_metric("rmse_test", rmse_test)
    mlflow.sklearn.log_model(final_model, "model")
    mlflow.log_artifact("reports/error_analysis.csv")
```

Xem lại bằng `mlflow ui`. Lựa chọn khác: Weights & Biases, DVC.

### 4.2. Checklist tái lập

- [ ] Cố định `random_state` ở mọi nơi (chia dữ liệu, mô hình, CV, NumPy).
- [ ] Ghi phiên bản thư viện (`uv.lock` hoặc `requirements.txt` ghim phiên bản).
- [ ] Ghi phiên bản dữ liệu (băm nội dung hoặc DVC); không sửa dữ liệu thô tại chỗ.
- [ ] Tách **cấu hình** (tham số) khỏi **code**, lưu vào file YAML/JSON hoặc dataclass.
- [ ] Ghi commit hash của code cho mỗi lần chạy.
- [ ] Dùng script/pipeline chạy lại được toàn bộ từ dữ liệu thô, không phụ thuộc thứ tự chạy ô notebook.

---

## 5. Giải thích mô hình

Hai mục đích khác nhau: **tin tưởng và gỡ lỗi** mô hình, và **giải thích cho người dùng/cơ quan quản lý**.

### 5.1. Giải thích toàn cục (cả mô hình)

| Phương pháp | Ý nghĩa | Lưu ý |
|---|---|---|
| **Hệ số** (mô hình tuyến tính) | Ảnh hưởng của mỗi đặc trưng | Chỉ so sánh được độ lớn nếu đã chuẩn hóa |
| **Feature importance của cây** | Mức giảm impurity | Nhanh nhưng thiên vị biến nhiều giá trị, tính trên train |
| **Permutation importance** | Xáo trộn một cột trên validation, đo mức tụt điểm | Đáng tin hơn; sai lệch khi các đặc trưng tương quan mạnh |
| **Partial dependence (PDP)** | Dự đoán trung bình theo một đặc trưng, giữ trung bình các đặc trưng khác | Che dấu khác biệt giữa các mẫu |
| **ICE** | Một đường cho mỗi mẫu | Thấy được sự không đồng nhất |
| **SHAP (tóm tắt)** | Đóng góp trung bình của mỗi đặc trưng | Xem bên dưới |

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

model.fit(X_train, y_train)
r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42,
                           scoring="neg_root_mean_squared_error")
imp = pd.Series(r.importances_mean, index=X_val.columns).sort_values(ascending=False)

PartialDependenceDisplay.from_estimator(model, X_val, ["area", "age"], kind="both")
```

### 5.2. Giải thích cục bộ (một dự đoán)

**SHAP** (SHapley Additive exPlanations) chia dự đoán của một mẫu thành đóng góp của từng đặc trưng, dựa trên lý thuyết trò chơi, với tính chất cộng tính: dự đoán = giá trị nền + tổng đóng góp.

```python
import shap

fitted = model.fit(X_train, y_train)
Xt = fitted.named_steps["preprocess"].transform(X_val)
names = fitted.named_steps["preprocess"].get_feature_names_out()

explainer = shap.TreeExplainer(fitted.named_steps["model"])
sv = explainer.shap_values(Xt)

shap.summary_plot(sv, Xt, feature_names=names)             # toàn cục
shap.force_plot(explainer.expected_value, sv[0], Xt[0],    # một mẫu
                feature_names=names, matplotlib=True)
```

**LIME** là lựa chọn khác: xấp xỉ mô hình quanh một điểm bằng mô hình tuyến tính đơn giản.

### 5.3. Cảnh báo khi giải thích

- Giải thích cho thấy mô hình **dùng** đặc trưng nào, **không** chứng minh quan hệ nhân quả.
- Đặc trưng tương quan chia sẻ "công" một cách khó đoán; đừng diễn giải quá chi tiết.
- Một đặc trưng quan trọng bất thường thường là **dấu hiệu rò rỉ**: kiểm tra ngay.
- Giải thích có thể phản ánh lỗi dữ liệu hơn là quy luật thật.

---

## 6. Công bằng và thiên lệch

Mô hình học y những gì dữ liệu chứa, kể cả định kiến lịch sử. Thói quen tối thiểu:

1. **Xác định nhóm bảo vệ** liên quan (giới tính, tuổi, vùng miền...), kể cả khi không dùng làm đặc trưng, vì đặc trưng khác có thể là đại diện gián tiếp (proxy), ví dụ mã bưu điện.
2. **Tách điểm đánh giá theo nhóm** và so sánh: tỉ lệ dương dự đoán, recall, precision, tỉ lệ báo nhầm.
3. Nhận biết các **định nghĩa công bằng** khác nhau và **xung đột lẫn nhau** (không thể thỏa mãn mọi định nghĩa cùng lúc), nên cần chọn theo bối cảnh:
   - Ngang bằng nhân khẩu học (demographic parity): tỉ lệ dương như nhau giữa các nhóm.
   - Cơ hội bình đẳng (equal opportunity): recall (TPR) như nhau giữa các nhóm.
   - Hiệu chỉnh theo nhóm (calibration): xác suất đúng nghĩa như nhau ở mọi nhóm.
4. Cách giảm: thu thập dữ liệu cân đối, đổi trọng số, chỉnh ngưỡng theo nhóm (cân nhắc pháp lý), loại đặc trưng proxy, ràng buộc công bằng khi huấn luyện (`fairlearn`).
5. Ghi nhận giới hạn của mô hình (model card): dùng cho mục đích nào, không nên dùng khi nào, đã kiểm tra những nhóm nào.

---

## 7. Lưu và nạp mô hình

```python
import joblib

joblib.dump(model, "artifacts/house_price_v1.joblib")
model = joblib.load("artifacts/house_price_v1.joblib")
```

Lưu **cả pipeline** (gồm tiền xử lý), không chỉ mô hình thô, để dự đoán đúng với dữ liệu đầu vào thô.

**Lưu ý quan trọng:**
- File `joblib`/`pickle` có thể **thực thi mã tùy ý khi nạp**. **Không nạp file từ nguồn không tin cậy.**
- Mô hình lưu bằng một phiên bản scikit-learn có thể lỗi hoặc cho kết quả khác ở phiên bản khác. **Ghim phiên bản** thư viện cùng với file mô hình.
- Gắn phiên bản vào tên file hoặc dùng model registry (MLflow) và lưu kèm metadata (ngày huấn luyện, dữ liệu, điểm, commit).
- Định dạng thay thế: ONNX (độc lập ngôn ngữ, nhanh), định dạng riêng của LightGBM/XGBoost (`save_model`).

---

## 8. Triển khai thành API

Đóng gói pipeline đã lưu thành dịch vụ HTTP với FastAPI:

```python
# src/serve.py
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load("artifacts/house_price_v1.joblib")
    yield

app = FastAPI(lifespan=lifespan)

class House(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    district: str
    built_year: int
    sold_year: int

@app.post("/predict")
def predict(h: House):
    df = pd.DataFrame([h.model_dump()])
    price = float(state["model"].predict(df)[0])
    return {"price": price, "model_version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Chạy: `uvicorn src.serve:app --reload`. Kiểm tra tại `http://localhost:8000/docs`.

**Các điểm cần chú ý:**
- **Kiểm tra đầu vào** bằng schema (Pydantic): kiểu, miền giá trị, trường bắt buộc.
- Nạp mô hình **một lần** khi khởi động, không nạp mỗi yêu cầu.
- Xử lý **giá trị lạ/thiếu** y như lúc huấn luyện (nhờ pipeline).
- Trả về **phiên bản mô hình** để truy vết.
- Đóng gói bằng **Docker**, ghim phiên bản Python và thư viện.
- Với tải lớn: dự đoán theo lô (batch), hàng đợi, hoặc lập lịch chạy offline thay vì API thời gian thực. Chọn **batch hay online** theo nhu cầu độ trễ.

Phần Docker, CI/CD, serving nâng cao nằm ở GĐ7 (MLOps).

---

## 9. Giám sát sau triển khai

Mô hình **xuống cấp theo thời gian** vì thế giới thay đổi. Cần giám sát:

| Hiện tượng | Ý nghĩa | Cách phát hiện |
|---|---|---|
| **Data drift** (covariate shift) | Phân phối đầu vào $P(x)$ đổi | So phân phối đặc trưng hiện tại với lúc train: PSI, KS-test, khoảng cách phân phối |
| **Concept drift** | Quan hệ $P(y\mid x)$ đổi (hành vi khách hàng đổi) | Theo dõi metric khi có nhãn trễ; so dự đoán với kết quả thật |
| **Label/prior shift** | Tỉ lệ lớp đổi | Theo dõi tỉ lệ dự đoán dương theo thời gian |
| **Lỗi dữ liệu** | Cột thiếu, đổi đơn vị, giá trị vô lý | Kiểm tra schema, tỉ lệ thiếu, miền giá trị |
| **Sức khỏe hệ thống** | Độ trễ, lỗi, thông lượng | Giám sát hạ tầng thông thường |

**Population Stability Index (PSI)** đo mức chênh giữa phân phối kỳ vọng $e_i$ và thực tế $a_i$ theo từng ô:

$$\text{PSI}=\sum_i(a_i-e_i)\ln\frac{a_i}{e_i}$$

Quy tắc kinh nghiệm: PSI < 0.1 ổn định, 0.1–0.25 thay đổi vừa, > 0.25 thay đổi lớn cần xem xét.

```python
import numpy as np

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))
```

**Vòng đời sau triển khai:**
1. Ghi log đầu vào, dự đoán, phiên bản mô hình (và nhãn thật khi có).
2. Dashboard + cảnh báo khi drift hoặc metric tụt.
3. Huấn luyện lại định kỳ hoặc theo kích hoạt; kiểm tra mô hình mới trước khi thay.
4. **Triển khai an toàn:** shadow mode (chạy song song, chưa dùng kết quả), canary (một phần nhỏ lưu lượng), A/B test, và luôn có đường **quay về** phiên bản cũ.

---

## 10. Cấu trúc dự án và kiểm thử

### 10.1. Cấu trúc thư mục gợi ý

```
projects/01-house-price-ml/
├── README.md              # bài toán, cách làm, kết quả, bài học
├── pyproject.toml
├── data/
│   ├── raw/               # dữ liệu gốc, không sửa (gitignore, ghi nguồn)
│   └── processed/
├── notebooks/             # khám phá: 01_eda.ipynb, 02_baseline.ipynb ...
├── src/
│   ├── data.py            # nạp, làm sạch, chia dữ liệu
│   ├── features.py        # biến đổi, tạo đặc trưng
│   ├── train.py           # huấn luyện, tuning, lưu mô hình
│   ├── evaluate.py        # metric, phân tích lỗi
│   └── serve.py           # API
├── tests/
├── artifacts/             # mô hình đã lưu (gitignore hoặc DVC)
├── reports/               # hình, bảng kết quả
└── configs/               # tham số thí nghiệm
```

Nguyên tắc: **notebook để khám phá, `src/` để lưu code đã chốt.** Khi một đoạn code trong notebook đã ổn thì đưa thành hàm trong `src/` và gọi lại từ notebook.

### 10.2. Kiểm thử code ML

Không thể kiểm thử "mô hình đúng", nhưng kiểm thử được rất nhiều thứ:

```python
# tests/test_pipeline.py
import numpy as np
import pandas as pd
from src.train import build_pipeline

def tiny_df():
    return pd.DataFrame({
        "area": [50, 80, 120, np.nan],
        "district": ["a", "b", "a", "c"],
    }), np.array([1.0, 2.0, 3.0, 2.5])

def test_pipeline_fits_and_predicts_with_missing():
    X, y = tiny_df()
    pipe = build_pipeline().fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(X),)
    assert not np.isnan(pred).any()

def test_pipeline_handles_unseen_category():
    X, y = tiny_df()
    pipe = build_pipeline().fit(X, y)
    new = pd.DataFrame({"area": [60], "district": ["zzz"]})
    assert np.isfinite(pipe.predict(new)).all()

def test_no_leakage_columns():
    X, _ = tiny_df()
    assert "price" not in X.columns
```

Các loại kiểm thử nên có:
- **Dữ liệu:** schema, kiểu, miền giá trị, không có hàng trùng giữa train và test.
- **Biến đổi:** đầu ra có đúng hình dạng, không `NaN`, đúng với ví dụ tính tay.
- **Mô hình:** pipeline `fit` được trên dữ liệu nhỏ, mô hình **thắng baseline** trên tập mẫu, kết quả **tái lập** với cùng seed.
- **Bất biến:** đổi thứ tự hàng không đổi dự đoán; thêm cột không liên quan không làm hỏng.
- **API:** dữ liệu hợp lệ trả đúng định dạng; dữ liệu sai trả lỗi rõ ràng.

Chạy: `uv run pytest`.

---

**Trước:** [Chương 8](08-hoc-khong-giam-sat.md) | **Tiếp theo:** [Chương 10: Các kiểu học khác](10-cac-kieu-hoc-khac.md)
