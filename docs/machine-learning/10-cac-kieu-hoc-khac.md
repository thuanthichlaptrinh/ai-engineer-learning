# Chương 10: Học bán giám sát, tự giám sát, tăng cường và các kiểu học khác

[Chương 1](01-tong-quan-va-cac-kieu-hoc.md) đã giới thiệu tổng quan. Chương này đi vào chi tiết kèm code cho các kiểu học ngoài hai kiểu chính (có giám sát, không giám sát). Phần học tăng cường ở đây ở mức nền tảng; các phần liên quan mạng neural sẽ được học sâu ở GĐ2, GĐ3 và GĐ6.

## Mục lục

1. [Học bán giám sát](#1-học-bán-giám-sát)
2. [Học tự giám sát](#2-học-tự-giám-sát)
3. [Học chuyển giao](#3-học-chuyển-giao-transfer-learning)
4. [Học chủ động](#4-học-chủ-động-active-learning)
5. [Học trực tuyến](#5-học-trực-tuyến-online-learning)
6. [Học từ nhãn yếu](#6-học-từ-nhãn-yếu-weak-supervision)
7. [Học tăng cường](#7-học-tăng-cường)
8. [Bảng tổng kết các kiểu học](#8-bảng-tổng-kết-các-kiểu-học)

---

## 1. Học bán giám sát

**Bối cảnh:** có $n_l$ mẫu có nhãn (ít) và $n_u$ mẫu không nhãn (nhiều), $n_u\gg n_l$. Gán nhãn đắt, dữ liệu thô thì rẻ.

**Ý tưởng:** dữ liệu không nhãn tiết lộ **hình dạng** phân phối, giúp đặt ranh giới quyết định ở vùng thưa thay vì cắt ngang cụm dày.

**Các giả định** để học bán giám sát có tác dụng:
- **Cụm:** các điểm cùng cụm có khả năng cùng nhãn.
- **Ranh giới thấp mật độ:** ranh giới nên đi qua vùng thưa.
- **Đa tạp:** dữ liệu nằm trên cấu trúc ít chiều.

Nếu giả định sai, dữ liệu không nhãn có thể **làm hại** mô hình. Luôn so với baseline chỉ dùng phần có nhãn.

### 1.1. Self-training (pseudo-labeling)

1. Huấn luyện mô hình trên dữ liệu có nhãn.
2. Dự đoán dữ liệu không nhãn; giữ những dự đoán có độ tin cậy ≥ ngưỡng làm **nhãn giả**.
3. Thêm vào tập huấn luyện, huấn luyện lại. Lặp.

```python
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Giả lập: chỉ giữ 10% nhãn, còn lại đặt -1 (nghĩa là "không nhãn")
y_partial = y_tr.copy()
unlabeled = rng.random(len(y_tr)) < 0.9
y_partial[unlabeled] = -1

# Baseline: chỉ dùng phần có nhãn
base = RandomForestClassifier(random_state=42).fit(X_tr[~unlabeled], y_tr[~unlabeled])

# Self-training: dùng cả phần không nhãn
self_train = SelfTrainingClassifier(
    RandomForestClassifier(random_state=42), threshold=0.9
).fit(X_tr, y_partial)

print("Chỉ dùng nhãn  :", base.score(X_te, y_te))
print("Self-training  :", self_train.score(X_te, y_te))
```

**Rủi ro:** nhãn giả sai sẽ củng cố lỗi (confirmation bias). Giảm bằng ngưỡng cao, chỉ thêm dần, và kiểm chứng trên tập validation **có nhãn thật**.

### 1.2. Label propagation / Label spreading

Dựng đồ thị tương đồng giữa mọi điểm (có và không nhãn), rồi **lan truyền nhãn** từ điểm có nhãn sang láng giềng.

```python
from sklearn.semi_supervised import LabelSpreading

ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2).fit(X_tr, y_partial)
y_inferred = ls.transduction_            # nhãn suy ra cho toàn bộ tập train
```

Phù hợp khi quan hệ láng giềng có ý nghĩa; chi phí tăng nhanh theo kích thước dữ liệu.

### 1.3. Các hướng hiện đại (nắm khái niệm)

- **Consistency regularization:** mô hình phải cho dự đoán giống nhau khi đầu vào bị nhiễu hay biến đổi nhẹ (cơ sở của FixMatch, Mean Teacher).
- **Co-training:** hai mô hình dùng hai góc nhìn đặc trưng khác nhau, gán nhãn giả cho nhau.
- **Mô hình sinh bán giám sát:** VAE, GAN nửa giám sát.

Thực tế hiện nay, **tiền huấn luyện tự giám sát rồi tinh chỉnh với ít nhãn** thường hiệu quả hơn các kỹ thuật bán giám sát cổ điển, đặc biệt với ảnh và văn bản.

---

## 2. Học tự giám sát

**Ý tưởng:** tự tạo nhãn từ chính dữ liệu bằng một **bài toán giả (pretext task)**, buộc mô hình học biểu diễn hữu ích; sau đó dùng biểu diễn đó cho bài toán thật (**downstream**) với ít nhãn.

### 2.1. Các nhóm bài toán giả

| Nhóm | Cách tạo nhãn | Ví dụ |
|---|---|---|
| **Dự đoán phần bị che** | Che một phần đầu vào, nhãn là phần bị che | BERT (che từ), MAE (che mảnh ảnh) |
| **Dự đoán tiếp theo** | Nhãn là phần tử kế tiếp trong chuỗi | GPT và LLM (dự đoán token tiếp theo) |
| **Học đối sánh (contrastive)** | Hai biến đổi của cùng một mẫu là cặp "dương", mẫu khác là "âm"; kéo cặp dương lại gần, đẩy cặp âm ra xa | SimCLR, MoCo, CLIP (ghép ảnh–chữ) |
| **Khôi phục (denoising)** | Làm nhiễu đầu vào, nhãn là bản sạch | Denoising autoencoder, diffusion |
| **Dự đoán biến đổi** | Xoay/cắt ảnh, nhãn là phép biến đổi đã áp dụng | RotNet |

### 2.2. Quy trình hai giai đoạn

```
Dữ liệu thô khổng lồ, không nhãn
        │  Tiền huấn luyện tự giám sát (pretraining)
        ▼
Mô hình nền (foundation model) biết biểu diễn tốt
        │  Tinh chỉnh với ít nhãn (fine-tuning) / prompt / thêm lớp phân loại
        ▼
Mô hình cho bài toán cụ thể
```

Đây là công thức đứng sau LLM, mô hình thị giác hiện đại và mô hình đa phương thức. Bạn sẽ học chi tiết ở GĐ3 (Transformer, LLM) và GĐ6 (fine-tuning).

### 2.3. Học đối sánh: ý tưởng cốt lõi

Hàm loss InfoNCE (dạng đơn giản): với một mẫu neo $z$, một mẫu dương $z^+$ và các mẫu âm $z^-_j$:

$$L=-\log\frac{\exp(\text{sim}(z,z^+)/\tau)}{\exp(\text{sim}(z,z^+)/\tau)+\sum_j\exp(\text{sim}(z,z^-_j)/\tau)}$$

`sim` thường là cosine similarity, $\tau$ là "nhiệt độ". Cực tiểu loss này làm biểu diễn của cặp dương gần nhau hơn các cặp âm. Embedding dùng trong RAG ở GĐ4 (sentence embeddings) được huấn luyện theo tinh thần này.

### 2.4. Với dữ liệu bảng

Tự giám sát ít mang lại lợi ích rõ so với boosting trên dữ liệu bảng truyền thống. Có một số hướng (denoising autoencoder, SCARF) nhưng không phải điểm bắt đầu. Hãy đặt tự giám sát vào ngữ cảnh ảnh, văn bản, âm thanh.

---

## 3. Học chuyển giao (Transfer Learning)

**Ý tưởng:** tri thức học được từ bài toán nguồn (nhiều dữ liệu) tái sử dụng cho bài toán đích (ít dữ liệu).

| Cách | Mô tả | Khi nào |
|---|---|---|
| **Trích xuất đặc trưng** (feature extraction) | Dùng mô hình đã huấn luyện làm bộ trích đặc trưng cố định, chỉ huấn luyện bộ phân loại nhỏ phía trên | Rất ít dữ liệu, bài toán đích giống bài toán nguồn |
| **Tinh chỉnh** (fine-tuning) | Khởi tạo từ mô hình đã huấn luyện, tiếp tục huấn luyện toàn bộ hoặc một phần các tầng, learning rate nhỏ | Dữ liệu vừa phải, cần độ chính xác cao hơn |
| **Thích nghi miền** (domain adaptation) | Điều chỉnh khi phân phối nguồn và đích khác nhau | Cùng tác vụ, khác miền (ảnh ban ngày → ban đêm) |
| **Few-shot / zero-shot** | Dùng mô hình lớn với vài hoặc không ví dụ (prompt) | LLM, mô hình đa phương thức |

Gợi ý: tầng đầu của mạng học đặc trưng chung (cạnh, kết cấu), tầng sau học đặc trưng riêng tác vụ. Nên **đóng băng tầng đầu, tinh chỉnh tầng sau**, rồi nếu cần mở dần.

Với ML cổ điển, transfer learning ít gặp; nó là trụ cột của DL. Phác thảo:

```python
# Phác thảo (chi tiết ở GĐ2): ảnh, ResNet đã huấn luyện trên ImageNet
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False                        # đóng băng
model.fc = nn.Linear(model.fc.in_features, 5)      # thay đầu ra cho 5 lớp mới; chỉ lớp này được học
```

---

## 4. Học chủ động (Active Learning)

**Bối cảnh:** nhãn đắt, bạn chỉ gán được một ngân sách nhỏ. **Ý tưởng:** để mô hình **chọn** mẫu nào nên gán nhãn tiếp theo (mẫu mang nhiều thông tin nhất), thay vì gán ngẫu nhiên.

### 4.1. Vòng lặp

```
1. Gán nhãn một tập nhỏ ban đầu
2. Huấn luyện mô hình
3. Dùng mô hình chấm điểm "mức hữu ích" của các mẫu chưa nhãn
4. Chọn mẫu điểm cao nhất, nhờ người gán nhãn
5. Thêm vào tập có nhãn, quay lại bước 2
```

### 4.2. Chiến lược chọn mẫu

| Chiến lược | Chọn mẫu nào |
|---|---|
| **Uncertainty sampling** | Mô hình ít chắc chắn nhất (xác suất lớn nhất thấp, entropy cao, hoặc sát ranh giới) |
| **Query-by-committee** | Các mô hình trong "hội đồng" bất đồng nhiều nhất |
| **Expected model change / error reduction** | Mẫu dự kiến làm thay đổi mô hình hoặc giảm lỗi nhiều nhất |
| **Diversity sampling** | Mẫu đa dạng, đại diện (tránh chọn lô các mẫu giống nhau) |

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Đã có: X_labeled, y_labeled, X_pool (chưa nhãn)
model = LogisticRegression(max_iter=1000).fit(X_labeled, y_labeled)

proba = model.predict_proba(X_pool)
uncertainty = 1 - proba.max(axis=1)                   # mô hình càng ít chắc thì càng cao
# Hoặc entropy: -(proba * np.log(proba + 1e-12)).sum(axis=1)

query_idx = np.argsort(uncertainty)[-20:]             # 20 mẫu đáng gán nhãn nhất
# -> gửi X_pool[query_idx] cho người gán nhãn, thêm vào tập có nhãn, lặp lại
```

### 4.3. Lưu ý

- Mô hình ban đầu quá yếu thì lựa chọn kém (vấn đề khởi động nguội).
- Chọn theo độ bất định thuần túy dễ chọn **outlier** hoặc lô trùng lặp; kết hợp đa dạng.
- Vẫn phải giữ một tập đánh giá **ngẫu nhiên** (không do chiến lược chọn), nếu không đánh giá sẽ lệch.
- Thư viện: `modAL`, `small-text`.

---

## 5. Học trực tuyến (Online Learning)

**Bối cảnh:** dữ liệu đến liên tục (luồng), quá lớn để nạp vào bộ nhớ, hoặc quy luật thay đổi theo thời gian. Mô hình **cập nhật dần** theo từng mẫu hoặc từng lô, không huấn luyện lại từ đầu.

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

classes = np.array([0, 1])                              # phải khai báo trước
clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
scaler = StandardScaler()

for X_batch, y_batch in stream_of_batches():            # nguồn dữ liệu luồng
    scaler.partial_fit(X_batch)                         # cập nhật thống kê chuẩn hóa dần
    clf.partial_fit(scaler.transform(X_batch), y_batch, classes=classes)
```

| Thuật toán hỗ trợ `partial_fit` (scikit-learn) | Dùng cho |
|---|---|
| `SGDClassifier`, `SGDRegressor` | Mô hình tuyến tính (log loss, hinge, ...) |
| `MultinomialNB`, `GaussianNB` | Naive Bayes |
| `MiniBatchKMeans` | Gom cụm |
| `IncrementalPCA` | Giảm chiều |
| `PassiveAggressiveClassifier` | Phân loại online |

Thư viện chuyên dụng: **River** (học trực tuyến thuần túy, kiểm tra theo kiểu "prequential": dự đoán rồi mới học).

**Thách thức:**
- **Concept drift:** quy luật đổi, mô hình cần "quên" dữ liệu cũ (dùng cửa sổ trượt, trọng số suy giảm).
- **Quên thảm họa** (catastrophic forgetting) với mạng neural khi học nhiệm vụ mới.
- Khó tái lập và gỡ lỗi hơn huấn luyện theo lô.

Cách tiếp cận thực dụng khác: huấn luyện lại theo lô **định kỳ** trên cửa sổ dữ liệu gần đây, đơn giản và dễ kiểm soát hơn.

---

## 6. Học từ nhãn yếu (Weak Supervision)

**Bối cảnh:** gán nhãn tay quá đắt, nhưng bạn có thể viết **hàm gán nhãn** dựa trên luật, từ điển, mô hình khác hoặc cơ sở tri thức. Mỗi hàm cho nhãn **nhiễu và không phủ hết** dữ liệu; kết hợp nhiều hàm để được nhãn tốt hơn.

```python
# Ví dụ hàm gán nhãn cho bài toán spam (nhãn: 1 spam, 0 không, -1 bỏ qua)
def lf_has_link(text):
    return 1 if "http" in text or "bit.ly" in text else -1

def lf_greeting_from_known(text):
    return 0 if text.lower().startswith(("chào anh", "chào chị")) else -1

def lf_money_words(text):
    return 1 if any(w in text.lower() for w in ["trúng thưởng", "vay nhanh", "miễn phí"]) else -1
```

Các hàm sẽ bất đồng, phủ chồng lấn nhau. Một **mô hình nhãn** (label model, như trong Snorkel) học độ chính xác và tương quan của từng hàm để gộp thành **nhãn xác suất**, rồi dùng huấn luyện mô hình thật.

Ngoài ra: **distant supervision** (nhãn từ cơ sở tri thức có sẵn), **dùng LLM để gán nhãn nháp** rồi kiểm tra lấy mẫu bằng người. Luôn có một tập nhỏ nhãn tay chất lượng cao để **đánh giá**.

---

## 7. Học tăng cường

### 7.1. Khung bài toán: MDP

Học tăng cường mô hình hóa bằng **quá trình quyết định Markov (MDP)** gồm $(S,A,P,R,\gamma)$:

- $S$: tập trạng thái; $A$: tập hành động.
- $P(s'\mid s,a)$: xác suất chuyển trạng thái.
- $R(s,a,s')$: phần thưởng.
- $\gamma\in[0,1)$: hệ số chiết khấu.

Tác tử chọn hành động theo **chính sách** $\pi(a\mid s)$ để tối đa **lợi tức (return)** kỳ vọng:

$$G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\dots=\sum_{k=0}^\infty\gamma^kr_{t+k+1}$$

**Hàm giá trị:**
- $V^\pi(s)$: lợi tức kỳ vọng khi bắt đầu từ $s$ và theo $\pi$.
- $Q^\pi(s,a)$: lợi tức kỳ vọng khi làm $a$ ở $s$ rồi theo $\pi$.

**Phương trình Bellman tối ưu** (cơ sở của hầu hết thuật toán):

$$Q^*(s,a)=\mathbb E\big[r+\gamma\max_{a'}Q^*(s',a')\big]$$

### 7.2. Khám phá và khai thác

Phải cân bằng **khai thác** (chọn hành động đang tốt nhất) và **khám phá** (thử hành động khác, có thể tìm ra thứ tốt hơn). Cách đơn giản nhất: **$\varepsilon$-greedy**, với xác suất $\varepsilon$ chọn ngẫu nhiên, còn lại chọn tốt nhất; giảm $\varepsilon$ dần theo thời gian.

### 7.3. Bandit nhiều tay (multi-armed bandit)

Trường hợp đơn giản nhất: **không có trạng thái**, chỉ có $K$ "cần gạt" mỗi cái có xác suất thưởng chưa biết. Ứng dụng thật: chọn mẫu quảng cáo, tiêu đề bài viết, gợi ý đầu trang.

```python
import numpy as np

rng = np.random.default_rng(0)
true_p = np.array([0.2, 0.5, 0.7])        # xác suất thắng thật của 3 cần gạt (tác tử không biết)
K = len(true_p)
Q = np.zeros(K)                           # ước lượng giá trị mỗi cần
N = np.zeros(K)                           # số lần đã chọn
eps = 0.1
total = 0

for t in range(5000):
    a = rng.integers(K) if rng.random() < eps else int(Q.argmax())   # epsilon-greedy
    r = float(rng.random() < true_p[a])                              # nhận thưởng 0/1
    N[a] += 1
    Q[a] += (r - Q[a]) / N[a]                                        # cập nhật trung bình trượt
    total += r

print("Ước lượng:", Q.round(3), "| lượt chọn:", N, "| tổng thưởng:", total)
```

Biến thể tốt hơn: **UCB** (cộng thêm "thưởng bất định" cho cần ít được thử) và **Thompson sampling** (lấy mẫu từ phân phối hậu nghiệm). Cả hai thường thắng $\varepsilon$-greedy.

### 7.4. Q-learning (bảng)

Học trực tiếp $Q(s,a)$ bằng bảng với trạng thái rời rạc, cập nhật sau mỗi bước:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]$$

Số trong ngoặc là **sai số TD** (temporal difference): chênh giữa ước lượng mới và cũ.

```python
# pip install gymnasium
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)       # lưới 4x4, tìm đường tới đích
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, eps = 0.1, 0.99, 0.2
rng = np.random.default_rng(0)

for episode in range(5000):
    s, _ = env.reset(seed=int(rng.integers(1_000_000)))
    done = False
    while not done:
        a = env.action_space.sample() if rng.random() < eps else int(Q[s].argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        target = r + gamma * Q[s2].max() * (not terminated)   # không bootstrap khi kết thúc thật
        Q[s, a] += alpha * (target - Q[s, a])
        s = s2

policy = Q.argmax(axis=1)                                     # hành động tốt nhất mỗi ô
print(policy.reshape(4, 4))
```

### 7.5. Từ bảng đến mạng neural

Bảng $Q$ không dùng được khi trạng thái quá nhiều (ảnh, liên tục). Thay bảng bằng mạng neural:

| Họ thuật toán | Học gì | Ví dụ |
|---|---|---|
| **Value-based** | Hàm giá trị $Q$ | **DQN** (chơi game Atari) |
| **Policy-based** | Trực tiếp chính sách $\pi$ | REINFORCE, **PPO** |
| **Actor-Critic** | Cả chính sách (actor) và giá trị (critic) | A2C, PPO, SAC |
| **Dựa trên mô hình** (model-based) | Học mô hình môi trường rồi lập kế hoạch | Dreamer, AlphaZero (kết hợp tìm kiếm) |

Các phần này thuộc Deep RL, nằm ngoài GĐ1, nằm trong mục mở rộng của roadmap.

### 7.6. Học tăng cường trong LLM (RLHF)

Sau khi tiền huấn luyện tự giám sát, LLM thường được **căn chỉnh** theo ý người dùng:

1. **SFT:** tinh chỉnh có giám sát trên cặp (câu hỏi, câu trả lời tốt).
2. **Mô hình thưởng** (reward model): học dự đoán câu trả lời nào được người đánh giá ưa hơn.
3. **RL** (thường PPO hoặc biến thể): tối ưu LLM để tối đa điểm thưởng, kèm ràng buộc không lệch quá xa mô hình gốc.

Các phương pháp thay thế đơn giản hơn: **DPO** (tối ưu trực tiếp trên cặp ưu tiên, không cần mô hình thưởng và RL tường minh). Bạn sẽ gặp ở GĐ6.

### 7.7. Khi nào dùng học tăng cường

Dùng khi: quyết định **tuần tự** ảnh hưởng đến tương lai, có thể mô phỏng hoặc thử nhiều lần, phần thưởng định nghĩa được. Cân nhắc kỹ vì RL **tốn dữ liệu, khó huấn luyện, khó tái lập**. Nhiều bài toán nghe như RL thực ra giải tốt bằng học có giám sát hoặc bandit.

---

## 8. Bảng tổng kết các kiểu học

| Kiểu học | Dữ liệu cần | Mục tiêu | Ví dụ | Thuật toán / công cụ |
|---|---|---|---|---|
| Có giám sát | Nhiều cặp $(x,y)$ | Dự đoán $y$ | Giá nhà, spam | Chương 5–6 |
| Không giám sát | Chỉ $x$ | Tìm cấu trúc | Phân khúc, bất thường | Chương 8 |
| Bán giám sát | Ít $(x,y)$, nhiều $x$ | Dùng $x$ không nhãn để học tốt hơn | Ảnh y tế ít nhãn | Self-training, label spreading |
| Tự giám sát | Nhiều $x$ | Học biểu diễn từ nhãn tự tạo | LLM, CLIP | Masked LM, contrastive (GĐ3) |
| Tăng cường | Tương tác với môi trường | Tối đa phần thưởng tích lũy | Game, robot, quảng cáo | Q-learning, PPO, bandit |
| Chuyển giao | Mô hình nguồn + ít dữ liệu đích | Tái dùng tri thức | Phân loại ảnh ít dữ liệu | Fine-tuning (GĐ2) |
| Chủ động | Pool chưa nhãn + ngân sách gán nhãn | Giảm chi phí gán nhãn | Dự án gán nhãn đắt | Uncertainty sampling |
| Trực tuyến | Luồng dữ liệu | Cập nhật liên tục | Giá, tin nhắn | `partial_fit`, River |
| Nhãn yếu | Hàm gán nhãn nhiễu | Tạo nhãn rẻ | Tạo tập huấn luyện nhanh | Snorkel, LLM gán nhãn |

**Câu hỏi quyết định:** *Bạn có loại tín hiệu nào, và tốn bao nhiêu để có thêm?* Câu trả lời dẫn tới kiểu học phù hợp nhất.

---

**Trước:** [Chương 9](09-pipeline-tuning-va-trien-khai.md) | **Tiếp theo:** [Chương 11: Bài tập, dự án và tự kiểm tra](11-bai-tap-du-an-va-tu-kiem-tra.md)
