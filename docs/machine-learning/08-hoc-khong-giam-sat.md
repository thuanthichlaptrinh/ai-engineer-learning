# Chương 8: Học không giám sát

Không có nhãn $y$, chỉ có $x$. Mục tiêu là tìm cấu trúc ẩn: nhóm, hướng biến thiên chính, điểm bất thường, quy luật đồng xuất hiện.

> **Liên hệ:** học không giám sát **không có nhãn**, nên 'chia train/test' và 'metric so với nhãn' ở [chương 3](03-danh-gia-mo-hinh.md) không áp dụng nguyên vẹn. Phần xử lý dữ liệu ở [chương 2](02-du-lieu-va-feature-engineering.md) (thiếu, outlier, chuẩn hóa) vẫn cần, và đặc biệt quan trọng vì các thuật toán ở đây dựa trên khoảng cách.

## Mục lục

1. [Tổng quan và thách thức](#1-tổng-quan-và-thách-thức)
2. [Gom cụm](#2-gom-cụm-clustering)
3. [Đánh giá gom cụm](#3-đánh-giá-gom-cụm)
4. [Giảm chiều](#4-giảm-chiều-dimensionality-reduction)
5. [Phát hiện bất thường](#5-phát-hiện-bất-thường-anomaly-detection)
6. [Luật kết hợp](#6-luật-kết-hợp-association-rules)
7. [Ước lượng mật độ](#7-ước-lượng-mật-độ)
8. [Ứng dụng thực tế](#8-ứng-dụng-thực-tế)

---

## 1. Tổng quan và thách thức

| Nhóm tác vụ | Câu hỏi | Thuật toán chính |
|---|---|---|
| Gom cụm | Dữ liệu chia thành những nhóm nào? | K-Means, phân cấp, DBSCAN, GMM |
| Giảm chiều | Biểu diễn gọn hơn bằng ít chiều hơn? | PCA, t-SNE, UMAP, LDA, autoencoder |
| Phát hiện bất thường | Mẫu nào khác thường? | Isolation Forest, LOF, One-class SVM |
| Luật kết hợp | Cái gì hay đi cùng cái gì? | Apriori, FP-Growth |
| Ước lượng mật độ | Phân phối của dữ liệu trông thế nào? | KDE, GMM |

**Thách thức chung:**
- **Không có đáp án đúng** để so sánh; "cụm tốt" tùy mục đích sử dụng.
- Kết quả **phụ thuộc mạnh** vào thang đo, độ đo khoảng cách, siêu tham số.
- Cần **con người diễn giải**: cụm 2 là nhóm khách hàng nào?
- Nhiều thuật toán dựa trên khoảng cách nên **bắt buộc chuẩn hóa** và nhạy với số chiều.

---

## 2. Gom cụm (Clustering)

### 2.1. K-Means

**Thuật toán** (Lloyd):
1. Chọn $K$ tâm cụm ban đầu.
2. **Gán** mỗi điểm vào tâm gần nhất.
3. **Cập nhật** mỗi tâm bằng trung bình các điểm trong cụm.
4. Lặp 2–3 đến khi các tâm không đổi.

Cực tiểu hóa **inertia** (tổng bình phương khoảng cách tới tâm của cụm mình):

$$\sum_{k=1}^K\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$$

Mỗi vòng lặp đảm bảo inertia không tăng, nên hội tụ, nhưng chỉ tới **cực tiểu địa phương**.

```python
import numpy as np

def kmeans(X, k, n_iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]      # khởi tạo ngẫu nhiên
    for _ in range(n_iters):
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)   # (n, k)
        labels = dist.argmin(axis=1)                                          # gán cụm
        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers
```

**Khởi tạo k-means++:** chọn tâm đầu ngẫu nhiên, các tâm sau chọn với xác suất tỉ lệ bình phương khoảng cách tới tâm gần nhất. Cho kết quả tốt và ổn định hơn nhiều; là mặc định của scikit-learn. Chạy nhiều lần (`n_init`) và lấy kết quả có inertia nhỏ nhất.

**Chọn $K$:**
- **Elbow:** vẽ inertia theo $K$, chọn "khuỷu tay" nơi giảm chậm lại. Mơ hồ khi không có khuỷu rõ.
- **Silhouette:** chọn $K$ cho điểm cao nhất.
- **Theo nghiệp vụ:** cần 3 nhóm khách hàng để chạy 3 chiến dịch.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    print(k, round(km.inertia_, 1), round(silhouette_score(Xs, km.labels_), 3))
```

| Ưu | Nhược |
|---|---|
| Đơn giản, nhanh ($O(nKd)$ mỗi vòng), mở rộng tốt | Phải chọn $K$ trước |
| Dễ diễn giải: tâm cụm là "điển hình" | Giả định cụm **hình cầu**, kích thước và mật độ gần nhau |
| Có phiên bản mini-batch cho dữ liệu lớn | Nhạy với outlier và khởi tạo; cần chuẩn hóa |

Biến thể: **MiniBatchKMeans** (dữ liệu rất lớn), **K-Medoids** (tâm là điểm thật, bền với outlier hơn), **K-Modes** (dữ liệu phân loại).

### 2.2. Gom cụm phân cấp (Hierarchical)

**Agglomerative** (từ dưới lên): bắt đầu mỗi điểm là một cụm, lần lượt **gộp hai cụm gần nhất** cho đến khi còn một cụm. Kết quả là **dendrogram** (cây gộp); cắt cây ở độ cao nào thì được số cụm tương ứng.

Cách tính khoảng cách giữa hai cụm (**linkage**):

| Linkage | Khoảng cách giữa hai cụm | Đặc điểm |
|---|---|---|
| Single | Gần nhất giữa hai điểm | Dễ bị "chuỗi" (chaining), nhạy nhiễu |
| Complete | Xa nhất giữa hai điểm | Cụm gọn, cỡ gần nhau |
| Average | Trung bình mọi cặp | Cân bằng |
| **Ward** | Mức tăng phương sai khi gộp | Cụm gọn, hay dùng nhất (chỉ Euclid) |

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

Z = linkage(Xs, method="ward")
dendrogram(Z, truncate_mode="lastp", p=20)
labels = fcluster(Z, t=4, criterion="maxclust")      # cắt thành 4 cụm
```

Ưu: không cần chọn $K$ trước, cho cái nhìn đa tầng. Nhược: tốn $O(n^2)$ bộ nhớ, $O(n^2\log n)$–$O(n^3)$ thời gian, không thực tế với dữ liệu lớn; quyết định gộp không thể hoàn tác.

### 2.3. DBSCAN

Gom theo **mật độ**: cụm là vùng đông đúc ngăn cách bởi vùng thưa. Hai tham số:
- `eps` ($\varepsilon$): bán kính láng giềng.
- `min_samples`: số điểm tối thiểu trong $\varepsilon$ để coi là vùng đông.

Ba loại điểm: **điểm lõi** (core, có ≥ `min_samples` láng giềng trong $\varepsilon$), **điểm biên** (border, nằm trong $\varepsilon$ của một điểm lõi), **nhiễu** (noise, gán nhãn `-1`).

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Gợi ý chọn eps: vẽ khoảng cách tới láng giềng thứ k, tìm "khuỷu"
k = 5
dist, _ = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)
kdist = np.sort(dist[:, -1])           # vẽ kdist để chọn eps

db = DBSCAN(eps=0.5, min_samples=5).fit(Xs)
print("Số cụm:", len(set(db.labels_) - {-1}), "| nhiễu:", (db.labels_ == -1).sum())
```

| Ưu | Nhược |
|---|---|
| Không cần chọn số cụm | Khó chọn `eps`, nhạy với nó |
| Cụm hình dạng **bất kỳ** (hình trăng, vòng tròn) | Kém khi các cụm có mật độ khác nhau nhiều |
| Tự phát hiện nhiễu/outlier | Kém trong không gian nhiều chiều |

**HDBSCAN** (gói `hdbscan`, hoặc `sklearn.cluster.HDBSCAN`) khắc phục việc chọn `eps` và xử lý mật độ thay đổi; nên thử trước DBSCAN thuần.

### 2.4. Gaussian Mixture Model (GMM)

Giả định dữ liệu sinh từ hỗn hợp của $K$ phân phối chuẩn:

$$p(x)=\sum_{k=1}^K\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)$$

Học tham số $(\pi_k,\mu_k,\Sigma_k)$ bằng thuật toán **EM** (Expectation-Maximization):
- **E-step:** tính "trách nhiệm" $\gamma_{ik}$ = xác suất điểm $i$ thuộc thành phần $k$ (gán **mềm**).
- **M-step:** cập nhật $\pi,\mu,\Sigma$ theo các trách nhiệm đó.

Lặp đến khi hội tụ. K-Means chính là trường hợp đặc biệt: gán **cứng** và cụm cầu cùng kích thước.

```python
from sklearn.mixture import GaussianMixture

for k in range(1, 10):
    gm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(Xs)
    print(k, round(gm.bic(Xs), 1))          # chọn k có BIC nhỏ nhất

gm = GaussianMixture(n_components=3, random_state=42).fit(Xs)
labels = gm.predict(Xs)
proba = gm.predict_proba(Xs)                # xác suất thuộc từng cụm
```

| Ưu | Nhược |
|---|---|
| Gán **mềm** (xác suất) | Giả định thành phần chuẩn |
| Cụm hình elip, kích thước khác nhau | Nhạy khởi tạo, có thể kẹt cực tiểu địa phương |
| Là mô hình sinh, dùng được cho ước lượng mật độ, phát hiện bất thường | Nhiều tham số ($\Sigma$ đầy đủ), cần đủ dữ liệu |
| Chọn $K$ bằng BIC/AIC một cách có nguyên tắc | |

`covariance_type`: `full` (linh hoạt nhất), `tied`, `diag`, `spherical` (đơn giản nhất, gần K-Means).

### 2.5. Các phương pháp khác (điểm qua)

- **Spectral clustering:** dựng đồ thị tương đồng, gom cụm trên vector riêng. Bắt được cụm phi tuyến, nhưng tốn kém.
- **Mean Shift, Affinity Propagation, OPTICS, BIRCH:** tồn tại cho các tình huống đặc thù.

### 2.6. Chọn thuật toán gom cụm

| Tình huống | Gợi ý |
|---|---|
| Cụm tròn, dữ liệu lớn, cần nhanh | K-Means / MiniBatchKMeans |
| Cụm hình dạng bất kỳ, có nhiễu | DBSCAN / HDBSCAN |
| Cần xem cấu trúc đa tầng, dữ liệu vừa | Phân cấp (Ward) |
| Cần gán mềm, cụm elip | GMM |
| Dữ liệu phân loại | K-Modes, K-Prototypes |

---

## 3. Đánh giá gom cụm

Không có nhãn thật nên có hai hướng.

### 3.1. Chỉ số nội tại (không cần nhãn)

| Chỉ số | Ý nghĩa | Tốt khi |
|---|---|---|
| **Silhouette** | $s=\dfrac{b-a}{\max(a,b)}$, với $a$ = khoảng cách trung bình tới điểm cùng cụm, $b$ = khoảng cách trung bình tới cụm gần nhất khác | Gần +1; gần 0 là cụm chồng nhau; âm là gán sai |
| **Calinski-Harabasz** | Tỉ số phương sai giữa cụm trên trong cụm | Cao |
| **Davies-Bouldin** | Độ tương đồng trung bình giữa mỗi cụm và cụm giống nó nhất | Thấp |
| **BIC / AIC** (GMM) | Độ khớp có phạt độ phức tạp | Thấp |

Các chỉ số này thiên vị cụm lồi, cầu, nên không công bằng với DBSCAN.

### 3.2. Chỉ số ngoại tại (khi có nhãn tham chiếu)

| Chỉ số | Ghi chú |
|---|---|
| **Adjusted Rand Index (ARI)** | So cặp điểm cùng/khác cụm; 1 là khớp hoàn toàn, ≈0 là ngẫu nhiên |
| **NMI** (Normalized Mutual Information) | Thông tin chung giữa cụm và nhãn |
| **Homogeneity / Completeness / V-measure** | Mỗi cụm chỉ chứa một lớp / mỗi lớp nằm gọn trong một cụm |

Dùng khi có nhãn tham chiếu để **thẩm định phương pháp** (không phải để huấn luyện).

### 3.3. Đánh giá theo mục đích (quan trọng nhất)

- **Ổn định:** gom cụm lại trên mẫu con/bootstrap, cụm có giữ nguyên không?
- **Diễn giải được:** có đặt tên được cho từng cụm không? Hồ sơ cụm (giá trị trung bình từng đặc trưng) có khác biệt có ý nghĩa nghiệp vụ không?
- **Hữu ích:** dùng cụm để chạy chiến dịch/dự báo có cải thiện kết quả thật không?

```python
profile = df.assign(cluster=labels).groupby("cluster").agg(["mean", "median"])
sizes = pd.Series(labels).value_counts(normalize=True)
```

---

## 4. Giảm chiều (Dimensionality Reduction)

Mục đích: trực quan hóa (2–3 chiều), nén, khử nhiễu, tăng tốc mô hình, đối phó lời nguyền số chiều.

**Chọn đặc trưng** giữ một tập con các cột gốc. **Trích xuất đặc trưng** (giảm chiều) tạo các chiều mới là tổ hợp của cột gốc.

### 4.1. PCA (Principal Component Analysis)

Tìm các hướng trực giao (thành phần chính) mà dữ liệu **biến thiên nhiều nhất**. Thành phần thứ nhất giữ phương sai lớn nhất, thành phần thứ hai lớn nhất trong phần còn lại và vuông góc với thành phần đầu, v.v.

**Toán:** sau khi trừ trung bình, thành phần chính là **vector riêng** của ma trận hiệp phương sai $\Sigma=\frac1{n-1}X^\top X$, sắp theo giá trị riêng giảm dần. Giá trị riêng chính là phương sai theo hướng đó. Thực tế tính qua **SVD**: $X=U\Sigma V^\top$, các thành phần là hàng của $V^\top$.

**Tỉ lệ phương sai giải thích (explained variance ratio):** mỗi thành phần giữ bao nhiêu phần trăm tổng phương sai. Chọn số thành phần sao cho tổng đạt 90–99%, hoặc nhìn biểu đồ scree.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

Xs = StandardScaler().fit_transform(X)            # BẮT BUỘC chuẩn hóa
pca = PCA(n_components=0.95).fit(Xs)              # giữ 95% phương sai
Z = pca.transform(Xs)
print(pca.n_components_, np.cumsum(pca.explained_variance_ratio_))

X_back = pca.inverse_transform(Z)                 # khôi phục xấp xỉ
recon_error = ((Xs - X_back) ** 2).mean()
```

**PCA tự cài bằng NumPy:**

```python
def pca_numpy(X, n_components):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    explained_var = (S ** 2) / (len(X) - 1)
    return Xc @ components.T, components, explained_var[:n_components]
```

**Lưu ý:**
- Chuẩn hóa trước, nếu không biến có thang lớn lấn át.
- PCA là **tuyến tính** và **không nhìn nhãn**: hướng biến thiên lớn chưa chắc hữu ích cho dự đoán.
- Thành phần chính khó diễn giải hơn đặc trưng gốc (xem `pca.components_` để biết trọng số).
- Khi dùng làm tiền xử lý cho mô hình, `fit` PCA chỉ trên train, trong pipeline.

**Biến thể:** `IncrementalPCA` (dữ liệu không nạp vừa bộ nhớ), `KernelPCA` (phi tuyến), `TruncatedSVD` (dữ liệu thưa như TF-IDF; còn gọi là LSA khi dùng cho văn bản).

### 4.2. LDA (giảm chiều có giám sát)

Khác PCA, LDA dùng nhãn để tìm hướng **tách các lớp tốt nhất** (tối đa tỉ lệ phương sai giữa lớp trên trong lớp). Tối đa $C-1$ chiều. Xem [Chương 7](07-svm-knn-naive-bayes.md).

### 4.3. t-SNE và UMAP (để trực quan hóa)

Hai kỹ thuật **phi tuyến** giữ cấu trúc **láng giềng cục bộ**, cho hình 2D đẹp, thường lộ ra cụm.

| | t-SNE | UMAP |
|---|---|---|
| Tốc độ | Chậm với dữ liệu lớn | Nhanh hơn nhiều |
| Giữ cấu trúc toàn cục | Kém | Tốt hơn |
| Biến đổi điểm mới | Không (sklearn) | Có (`transform`) |
| Tham số chính | `perplexity` (5–50) | `n_neighbors`, `min_dist` |

```python
from sklearn.manifold import TSNE
import umap       # pip install umap-learn

Z_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xs)
Z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Xs)
```

**Cảnh báo khi đọc hình t-SNE/UMAP:**
- Chỉ để **khám phá và trực quan hóa**, đừng dùng làm đặc trưng cho mô hình mà không cẩn thận.
- Kích thước cụm và khoảng cách giữa các cụm trên hình **không đáng tin**.
- Kết quả đổi theo siêu tham số và seed; thử vài giá trị.
- Nên giảm bằng PCA (50 chiều) trước khi chạy t-SNE trên dữ liệu nhiều chiều.

### 4.4. Autoencoder

Mạng neural nén đầu vào thành mã ngắn (bottleneck) rồi dựng lại. Giảm chiều phi tuyến, khử nhiễu, phát hiện bất thường (sai số tái tạo lớn ⇒ bất thường). Học ở GĐ2.

### 4.5. Chọn phương pháp

| Mục đích | Dùng |
|---|---|
| Tiền xử lý, nén, khử tương quan | PCA |
| Vẽ hình 2D để khám phá | UMAP (hoặc t-SNE) |
| Văn bản thưa | TruncatedSVD |
| Giảm chiều giữ thông tin phân biệt lớp | LDA |
| Phi tuyến, dữ liệu phức tạp, có thể dùng mạng neural | Autoencoder |

---

## 5. Phát hiện bất thường (Anomaly Detection)

Tìm mẫu **hiếm và khác thường**: gian lận, lỗi thiết bị, xâm nhập mạng, lỗi dữ liệu.

**Chọn đúng bài toán:**
- Có **nhiều nhãn** cả hai loại ⇒ đó là phân loại mất cân bằng ([Chương 2](02-du-lieu-va-feature-engineering.md)).
- Chỉ có dữ liệu **bình thường** (sạch) ⇒ **novelty detection**: học thế nào là bình thường, cái gì ngoài đó là lạ.
- Dữ liệu lẫn lộn, **không nhãn** ⇒ **outlier detection**.

### 5.1. Các phương pháp

| Phương pháp | Ý tưởng | Ghi chú |
|---|---|---|
| **Thống kê** | Z-score, IQR | Nhanh, chỉ một vài chiều |
| **Isolation Forest** | Chia ngẫu nhiên; điểm bất thường **dễ bị cô lập** (đường đi ngắn trong cây) | Nhanh, mở rộng tốt, lựa chọn đầu tiên |
| **LOF** (Local Outlier Factor) | So **mật độ cục bộ** của điểm với láng giềng | Bắt được bất thường cục bộ; chậm hơn |
| **One-class SVM** | Tìm ranh giới bao quanh dữ liệu bình thường | Nhạy với tham số, chậm khi lớn |
| **GMM / KDE** | Điểm có mật độ thấp là bất thường | Cần ước lượng mật độ tốt |
| **Autoencoder** | Sai số tái tạo lớn | Cho dữ liệu phức tạp (ảnh, chuỗi) |

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(X)
score = -iso.score_samples(X)             # điểm càng cao càng bất thường
is_anomaly = iso.predict(X) == -1         # -1 là bất thường

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
lof_pred = lof.fit_predict(X)             # -1 là bất thường
```

`contamination` là tỉ lệ bất thường dự kiến; nếu không biết, dùng điểm số liên tục (`score`) rồi chọn ngưỡng theo nghiệp vụ (ví dụ top 1% để người kiểm tra thủ công).

### 5.2. Đánh giá

Hiếm khi có nhãn đầy đủ. Dùng một tập nhỏ có nhãn để đo PR-AUC, hoặc để chuyên gia xem xét các mẫu top điểm và tính tỉ lệ thật sự bất thường (precision@k). Tránh dùng accuracy.

---

## 6. Luật kết hợp (Association Rules)

Tìm quy luật dạng "ai mua A thường mua B" trong dữ liệu giao dịch (market basket analysis).

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **Support** của $A$ | $P(A)$ | $A$ xuất hiện trong bao nhiêu phần trăm giao dịch |
| **Confidence** của $A\Rightarrow B$ | $P(B\mid A)$ | Trong các giao dịch có $A$, bao nhiêu có $B$ |
| **Lift** | $\dfrac{P(B\mid A)}{P(B)}$ | Lift > 1: $A$ và $B$ đi cùng nhau nhiều hơn ngẫu nhiên; ≈1: độc lập; <1: loại trừ |

Confidence cao có thể đánh lừa nếu $B$ vốn phổ biến (ai cũng mua túi nilon). **Lift** cho thấy quan hệ thật.

**Apriori:** dựa trên nguyên lý "mọi tập con của tập phổ biến cũng phổ biến", nên cắt tỉa được rất nhiều ứng viên. **FP-Growth** nhanh hơn, dùng cây nén.

```python
from mlxtend.frequent_patterns import apriori, association_rules   # pip install mlxtend

# basket: DataFrame bool, mỗi hàng là một giao dịch, mỗi cột là một sản phẩm
frequent = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.2)
rules.sort_values("lift", ascending=False).head(10)
```

Ứng dụng: bố trí kệ hàng, gợi ý "thường mua cùng", khuyến mãi gói. Hạn chế: bùng nổ số luật, nhiều luật hiển nhiên; cần lọc theo lift và ý nghĩa nghiệp vụ.

---

## 7. Ước lượng mật độ

Ước lượng hàm mật độ xác suất $p(x)$ từ mẫu. Dùng để biết vùng nào dữ liệu dày/thưa, phát hiện bất thường, hoặc sinh dữ liệu.

- **Histogram:** đơn giản, nhạy kích thước ô.
- **KDE (Kernel Density Estimation):** đặt một "chuông" (kernel) quanh mỗi điểm rồi cộng lại; tham số băng thông `bandwidth` điều khiển độ mượt (nhỏ ⇒ nhiễu, lớn ⇒ quá mượt).
- **GMM:** mô hình tham số, xem mục 2.4.

```python
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(Xs)
log_density = kde.score_samples(Xs)       # log p(x); thấp ⇒ hiếm
```

---

## 8. Ứng dụng thực tế

| Ứng dụng | Kỹ thuật | Cách dùng kết quả |
|---|---|---|
| **Phân khúc khách hàng** | K-Means/GMM trên đặc trưng RFM (gần đây, tần suất, giá trị) | Chiến dịch riêng cho từng nhóm |
| **Gom chủ đề văn bản** | TF-IDF/embedding + K-Means, hoặc topic model (LDA văn bản) | Tổ chức tài liệu, khám phá chủ đề |
| **Phát hiện gian lận / lỗi** | Isolation Forest, autoencoder | Hàng đợi kiểm tra thủ công |
| **Nén và khử nhiễu** | PCA, autoencoder | Tăng tốc, giảm lưu trữ |
| **Tạo đặc trưng cho mô hình giám sát** | Nhãn cụm, khoảng cách tới tâm, thành phần PCA | Thêm vào bảng đặc trưng (trong pipeline) |
| **Gợi ý sản phẩm** | Luật kết hợp, lọc cộng tác, phân rã ma trận | Tăng doanh thu chéo |
| **Nén ảnh / lượng tử hóa màu** | K-Means trên điểm ảnh | Giảm số màu |
| **Khám phá dữ liệu** | PCA, UMAP | Hình dung cấu trúc trước khi mô hình hóa |

**Quy trình điển hình cho phân khúc khách hàng:**

1. Chọn đặc trưng có ý nghĩa nghiệp vụ, xử lý lệch (log), chuẩn hóa.
2. Thử vài thuật toán và $K$; kiểm tra silhouette, độ ổn định.
3. Lập hồ sơ cụm, đặt tên, kiểm tra với chuyên gia nghiệp vụ.
4. Thử nghiệm hành động theo nhóm (A/B test) để chứng minh phân khúc có giá trị.

---

**Trước:** [Chương 7](07-svm-knn-naive-bayes.md) | **Tiếp theo:** [Chương 9: Pipeline, tuning, giải thích và triển khai](09-pipeline-tuning-va-trien-khai.md)
