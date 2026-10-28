# Machine Learning: tài liệu học tập

Bộ tài liệu này đi cùng **GĐ1** trong [roadmap](../roadmap.md). Học xong bạn cần làm được: nhận một bài toán, chọn đúng kiểu học, dựng pipeline đầy đủ cho dữ liệu dạng bảng, chọn và đánh giá mô hình đúng cách, giải thích kết quả và đưa ra chạy thử.

**Điều kiện đầu vào:** đã xong GĐ0 (Python, NumPy, Pandas, đại số tuyến tính cơ bản, đạo hàm, xác suất cơ bản).

## Mục lục

Bộ tài liệu chia theo **kiểu học**, vì đó là khái niệm đầu tiên cần nắm: mô hình học từ loại tín hiệu nào (có nhãn, không nhãn, phần thưởng...).

### Phần A: Nền tảng chung (mọi kiểu học)

| # | Chương | Nội dung chính |
|---|---|---|
| 1 | [Tổng quan và các kiểu học](01-tong-quan-va-cac-kieu-hoc.md) | **Học có giám sát, không giám sát, bán giám sát, tự giám sát, tăng cường** và bản đồ chương; phân loại mô hình; loại dữ liệu; quy trình dự án; thuật ngữ |
| 2 | [Dữ liệu và feature engineering](02-du-lieu-va-feature-engineering.md) | EDA, chia dữ liệu, giá trị thiếu, outlier, mã hóa, chuẩn hóa, tạo và chọn đặc trưng, rò rỉ dữ liệu, mất cân bằng lớp |

### Phần B: Học có giám sát (có nhãn $y$)

| # | Chương | Nội dung chính |
|---|---|---|
| 3 | [Đánh giá mô hình](03-danh-gia-mo-hinh.md) | Metric hồi quy và phân loại, ROC/PR, kiểm định chéo, bias-variance, learning curve, so sánh có ý nghĩa thống kê, phân tích lỗi |
| 4 | [Toán nền tảng và tối ưu hóa](04-toan-nen-tang-va-toi-uu.md) | Loss, MLE/MAP, gradient descent và biến thể, xác suất, lý thuyết thông tin, lời nguyền số chiều |
| 5 | [Hồi quy và phân loại tuyến tính](05-hoi-quy-va-phan-loai.md) | Linear/Logistic Regression, regularization, đa thức, đa lớp và đa nhãn, ngưỡng quyết định, calibration |
| 6 | [Cây quyết định và ensemble](06-cay-va-ensemble.md) | Cây, Random Forest, Gradient Boosting, XGBoost/LightGBM/CatBoost, voting, stacking |
| 7 | [SVM, KNN, Naive Bayes](07-svm-knn-naive-bayes.md) | Lề lớn nhất và kernel, láng giềng gần nhất, xác suất Bayes, LDA/QDA |

### Phần C: Học không giám sát (không có nhãn)

| # | Chương | Nội dung chính |
|---|---|---|
| 8 | [Học không giám sát](08-hoc-khong-giam-sat.md) | K-Means, phân cấp, DBSCAN, GMM, PCA, t-SNE/UMAP, phát hiện bất thường, luật kết hợp, cách đánh giá riêng |

### Phần D: Đưa vào thực tế và mở rộng

| # | Chương | Nội dung chính |
|---|---|---|
| 9 | [Pipeline, tuning, giải thích và triển khai](09-pipeline-tuning-va-trien-khai.md) | Pipeline, Optuna, thí nghiệm có kỷ luật, SHAP, công bằng, FastAPI, giám sát drift, test |
| 10 | [Các kiểu học khác](10-cac-kieu-hoc-khac.md) | Bán giám sát, tự giám sát, chuyển giao, chủ động, trực tuyến, nhãn yếu, học tăng cường (chi tiết và code) |
| 11 | [Bài tập, dự án và tự kiểm tra](11-bai-tap-du-an-va-tu-kiem-tra.md) | Bài tập theo chương, dự án cuối giai đoạn, bộ dữ liệu, câu hỏi tự kiểm tra, lỗi thường gặp, thuật ngữ Anh–Việt |

## Vì sao sắp xếp theo thứ tự này

```
A. Nền tảng chung:   Kiểu học (1) → Dữ liệu (2)
B. Có giám sát:      Cách đo (3) → Công cụ toán (4) → Mô hình (5–7)
C. Không giám sát:   (8)
D. Thực tế, mở rộng: Đóng gói (9) → Kiểu học khác (10) → Luyện tập (11)
```

- **Kiểu học đứng đầu (chương 1)** vì mọi thứ sau đều thuộc một kiểu học nào đó. Chương 1 có bản đồ chỉ rõ chương nào dạy kiểu nào.
- **Học có giám sát đi trước học không giám sát.** Nó phổ biến hơn, có đáp án để đo nên dễ hiểu đúng sai, và nhiều ý tưởng của học không giám sát (khoảng cách, loss, chuẩn hóa) được dạy ở đó trước.
- **Trong học có giám sát, cách đo (chương 3) đứng trước mô hình.** Mọi mô hình ở chương 5 đến 7 đều cần chia dữ liệu, đo bằng metric, kiểm tra overfitting. Học thước đo trước thì mỗi mô hình mới đều có cái để so.
- **Toán đứng ngay trước hồi quy tuyến tính,** để gradient descent và MLE vừa học là dùng ngay, thay vì học trừu tượng ở đầu.
- **Mô hình đi từ đơn giản đến mạnh:** tuyến tính, cây và ensemble, rồi các mô hình khác. Chương 7 có thể đọc lướt lần đầu.
- **Chương 10** đi sâu các kiểu học còn lại (bán giám sát, tự giám sát, tăng cường...). Chương 1 đã giới thiệu khái niệm, đọc chương 10 sau khi đã vững phần chính.

## Cách học

1. **Mỗi chương: đọc, gõ lại code, rồi làm bài tập** ở chương 11. Chỉ đọc thì kiến thức rơi rất nhanh.
2. **Tự cài đặt một lần, rồi mới dùng thư viện** (hồi quy tuyến tính, logistic, KNN, K-Means, PCA).
3. **Luôn có baseline và đo bằng cross-validation.** Đây là thói quen quan trọng nhất của chương 3.
4. **Làm dự án cuối giai đoạn** (chương 11) song song từ khoảng tuần 3, áp dụng từng chương vào đó.
5. **Ghi nhật ký thí nghiệm,** gồm cả những thứ thất bại.

## Kế hoạch 8 tuần

| Tuần | Chương | Làm gì |
|---|---|---|
| 1 | 1, 2 | Đọc tổng quan; EDA một dataset (Titanic hoặc California Housing) |
| 2 | 3, 4 | Metric và CV; tự cài gradient descent; cố tình tạo rò rỉ để thấy điểm đẹp giả |
| 3 | 5 | Tự cài hồi quy tuyến tính và logistic; Ridge/Lasso; chọn ngưỡng |
| 4 | 6 | Cây, Random Forest, LightGBM; early stopping; so sánh trên cùng các fold |
| 5 | 7, đầu 8 | SVM, KNN, Naive Bayes (đọc lướt); bắt đầu K-Means |
| 6 | 8, đầu 9 | Gom cụm, PCA, bất thường; dựng pipeline `ColumnTransformer` |
| 7 | 9 | Optuna, SHAP, API FastAPI, test; hoàn thiện dự án |
| 8 | 10, 11 | Đọc các kiểu học khác; làm bài tập bán giám sát và bandit; tự kiểm tra và viết README dự án |

Nếu quỹ thời gian eo hẹp: gộp tuần 5 vào tuần 4 và đọc lướt chương 10. Đừng bỏ bài tập tự cài đặt và dự án.

## Cài đặt môi trường

```bash
uv add numpy pandas matplotlib seaborn scikit-learn scipy statsmodels lightgbm xgboost optuna shap imbalanced-learn mlxtend jupyter pytest fastapi uvicorn joblib mlflow
```

Thêm khi cần: `umap-learn` (UMAP), `gymnasium` (học tăng cường, chương 10), `catboost`.

## Ghi chú về code trong tài liệu

Các đoạn code là **ví dụ minh họa** để hiểu cơ chế và làm điểm xuất phát. Một số đoạn dùng biến giả định (`X_train`, `y_train`, `df`, `preprocess`) đã được định nghĩa ở đoạn trước hoặc ở dữ liệu của bạn. Khi thực hành, hãy chạy từng đoạn, đọc thông báo lỗi, và đối chiếu với tài liệu của phiên bản thư viện bạn đang dùng, vì API có thể thay đổi giữa các phiên bản.

## Tài liệu tham khảo

**Khóa học**
- *Machine Learning Specialization* (Andrew Ng, Coursera): trực quan, tốt cho người mới.
- *An Introduction to Statistical Learning* (ISLP) và video đi kèm: lý thuyết rõ ràng, có bản Python.
- Kaggle Learn (Intro to ML, Intermediate ML, Feature Engineering, Data Cleaning).
- StatQuest (YouTube): giải thích khái niệm thống kê và ML bằng hình ảnh.

**Sách**
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron): thực hành, bám sát quy trình trong bộ tài liệu này.
- *Pattern Recognition and Machine Learning* (Christopher Bishop): chiều sâu toán, đọc dần sau khi có nền.
- *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman): tham khảo lý thuyết.
- *Reinforcement Learning: An Introduction* (Sutton, Barto): tham khảo cho học tăng cường.

**Tài liệu kỹ thuật**
- User Guide của scikit-learn (rất đáng đọc, giải thích kèm ví dụ).
- Tài liệu LightGBM, XGBoost, CatBoost, Optuna, SHAP.

**Công cụ:** scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, SHAP, imbalanced-learn, MLflow, FastAPI.

---

**Bắt đầu:** [Chương 1: Tổng quan và các kiểu học](01-tong-quan-va-cac-kieu-hoc.md)
