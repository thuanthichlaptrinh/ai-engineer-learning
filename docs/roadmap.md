# Roadmap học AI Engineer

Lộ trình đi từ nền tảng đến Machine Learning, Deep Learning, LLM, RAG, Agents, Fine-tuning, MLOps và Big Data.
Mỗi giai đoạn tương ứng một thư mục trong repo (đánh số **theo đúng thứ tự học**) và kết thúc bằng một sản phẩm cụ thể.

**Mục tiêu mặc định:** AI/LLM Engineer ứng dụng (xây hệ thống dùng LLM chạy thật), có nền ML/DL vững để hiểu cái mình đang dùng. Nếu mục tiêu khác, xem phần [Chọn nhánh theo mục tiêu](#chọn-nhánh-theo-mục-tiêu).

**Giả định thời gian:** 10–15 giờ mỗi tuần. Tổng khoảng **10–13 tháng**. Học xen kẽ không làm giảm số giờ cần học, chỉ đổi thứ tự, nên các mốc dưới đây đã cộng đủ.

Ký hiệu: **[Cốt lõi]** bắt buộc, **[Tùy chọn]** học khi có thời gian hoặc khi mục tiêu cần.

## Tổng quan

| GĐ | Giai đoạn | Thư mục | Thời gian | Sản phẩm đầu ra |
|---|---|---|---|---|
| 0 | Nền tảng: Python, kỹ năng phần mềm, Toán, SQL | `00-foundations/` | 5–6 tuần | Phân tích dataset thật bằng Pandas, repo có test |
| 1 | Machine Learning | `01-machine-learning/` | 6–8 tuần | Dự đoán giá nhà |
| 2 | Deep Learning | `02-deep-learning/` | 6–8 tuần | Bộ phân loại ảnh |
| 3 | NLP và LLM | `03-nlp-llm/` | 5–7 tuần | GPT mini tự viết, chạy model mã nguồn mở |
| 4 | RAG và đánh giá | `04-rag/` | 5–6 tuần | Hệ thống hỏi đáp tài liệu có bộ đánh giá |
| 5 | Agents | `05-agents/` | 4–5 tuần | Agent dùng tool và MCP server |
| 6 | Fine-tuning | `06-fine-tuning/` | 3–4 tuần | LLM nhỏ tinh chỉnh cho một miền hẹp |
| 7 | MLOps và LLMOps | `07-mlops-deployment/` | 5–6 tuần | Dịch vụ LLM đóng gói, có CI và monitoring |
| 8 | Big Data **[Tùy chọn một phần]** | `08-big-data/` | 3–4 tuần | Pipeline dữ liệu chạy theo lịch |
| – | Mở rộng **[Tùy chọn]** | `09-optional/` | tùy chọn | CV nâng cao, diffusion, RL, time series, recsys, speech |

Cộng lại: 42–54 tuần. Bỏ phần Kafka/Spark của GĐ8 thì bớt được khoảng 2 tuần.

**Học nhẹ xen kẽ từ sớm** (không tính thành giai đoạn riêng):
- Git, `pytest`, GitHub: từ tuần đầu tiên.
- Docker và FastAPI mức cơ bản: từ khoảng tháng 5, bọc thử một model đã train thành API.
- SQL và DuckDB: trong GĐ1, khi bạn cần xử lý dữ liệu lớn hơn bộ nhớ.

## GĐ0: Nền tảng (5–6 tuần) [Cốt lõi]

**Mục tiêu:** viết Python sạch, hiểu toán vừa đủ, quen công cụ của người làm phần mềm.

Python và kỹ năng phần mềm:
- [ ] Python: kiểu dữ liệu, OOP, decorator, generator, typing, xử lý file, `async`
- [ ] NumPy, Pandas, Matplotlib
- [ ] Git, GitHub, dòng lệnh Linux, `uv` (WSL2 nếu cần trên Windows)
- [ ] Viết test với `pytest`, cấu trúc package, logging, xử lý lỗi
- [ ] SQL cơ bản (JOIN, GROUP BY, window function)

Toán:
- [ ] Đại số tuyến tính: vector, ma trận, nhân ma trận, eigenvalue
- [ ] Giải tích: đạo hàm, gradient, chain rule
- [ ] Xác suất và thống kê: phân phối, kỳ vọng, phương sai, Bayes, kiểm định giả thuyết
- [ ] Lý thuyết thông tin: entropy, cross-entropy, KL divergence (nền của loss và đánh giá LLM)
- [ ] Tối ưu hóa: hàm lồi và không lồi, gradient descent, momentum

**Tài liệu gợi ý:** 3Blue1Brown (Essence of Linear Algebra, Calculus), Khan Academy, "Python for Data Analysis" của Wes McKinney.

**Hoàn thành khi:** bạn tự làm sạch và phân tích được một dataset thật bằng Pandas, và repo có test chạy bằng một lệnh.

> Nếu bạn chưa từng lập trình, dành thêm 3–4 tuần cho Python trước khi vào phần toán.

## GĐ1: Machine Learning (6–8 tuần) [Cốt lõi]

**Mục tiêu:** hiểu quy trình ML từ dữ liệu đến đánh giá mô hình.

- [ ] Hồi quy tuyến tính và logistic, gradient descent (tự cài đặt bằng NumPy)
- [ ] Cây quyết định, Random Forest, Gradient Boosting (XGBoost, LightGBM)
- [ ] SVM, KNN, Naive Bayes
- [ ] Clustering (K-Means, DBSCAN), PCA
- [ ] Đánh giá: train/val/test, cross-validation, precision/recall/F1, ROC-AUC
- [ ] Overfitting, regularization, bias-variance, data leakage
- [ ] Feature engineering, xử lý dữ liệu thiếu và mất cân bằng
- [ ] Tuning siêu tham số (Optuna)
- [ ] Thiết kế thí nghiệm: baseline, ablation, so sánh có ý nghĩa thống kê
- [ ] Giải thích mô hình (SHAP, feature importance), công bằng và thiên lệch (bias) của dữ liệu

**Tài liệu chi tiết:** [machine-learning.md](machine-learning.md)

**Tài liệu gợi ý:** Machine Learning Specialization của Andrew Ng, "Hands-On Machine Learning" của Aurélien Géron.

**Dự án:** `projects/01-house-price-ml/`, pipeline đầy đủ (EDA, feature engineering, so sánh mô hình, báo cáo).

**Hoàn thành khi:** bạn giải thích được vì sao chọn mô hình và metric đó, và tự cài đặt được hồi quy logistic từ đầu.

## GĐ2: Deep Learning (6–8 tuần) [Cốt lõi]

**Mục tiêu:** xây và huấn luyện mạng neural bằng PyTorch.

- [ ] Perceptron, MLP, backpropagation (tự cài bằng NumPy một lần)
- [ ] PyTorch: tensor, autograd, `nn.Module`, DataLoader, training loop
- [ ] Hàm kích hoạt, loss, optimizer (SGD, Adam), learning rate scheduler
- [ ] Batch norm, layer norm, dropout, khởi tạo trọng số
- [ ] CNN: tích chập, pooling, ResNet, transfer learning
- [ ] RNN, LSTM, GRU (học để hiểu lịch sử, không cần đào sâu)
- [ ] Huấn luyện trên GPU, mixed precision, gradient accumulation
- [ ] Theo dõi thí nghiệm bằng Weights & Biases hoặc MLflow
- [ ] Huấn luyện phân tán (DDP, FSDP, DeepSpeed) ở mức khái niệm: vì sao cần và chia việc thế nào

**Tài liệu gợi ý:** Deep Learning Specialization của Andrew Ng, fast.ai Practical Deep Learning, "Neural Networks: Zero to Hero" của Andrej Karpathy (phần backprop và makemore).

**Dự án:** `projects/02-image-classifier/`, phân loại ảnh bằng CNN và transfer learning.

**Hoàn thành khi:** bạn viết được training loop không cần copy, và hiểu từng bước của backpropagation.

## GĐ3: NLP và LLM (5–7 tuần) [Cốt lõi]

**Mục tiêu:** hiểu LLM hoạt động bên trong và dùng được model mã nguồn mở.

Nền tảng:
- [ ] Tokenization (BPE), embeddings (Word2Vec, sentence embeddings)
- [ ] Self-attention, multi-head attention, positional encoding
- [ ] Kiến trúc Transformer đầy đủ (encoder, decoder, decoder-only)
- [ ] Tự cài đặt GPT mini (theo Karpathy "Let's build GPT")
- [ ] Pretraining và scaling laws ở mức khái niệm

LLM hiện đại:
- [ ] RoPE, Grouped-Query Attention, Mixture of Experts (mức khái niệm)
- [ ] KV cache, quantization, vì sao inference tốn bộ nhớ
- [ ] RLHF và alignment ở mức khái niệm
- [ ] Mô hình đa phương thức (vision-language) ở mức khái niệm

Thực hành:
- [ ] HuggingFace Transformers: tải model, inference, pipeline
- [ ] Sampling: temperature, top-k, top-p
- [ ] Prompt engineering, structured output, gọi API LLM
- [ ] Đánh giá LLM: perplexity, benchmark, LLM-as-judge

**Tài liệu gợi ý:** "Attention Is All You Need", Karpathy "Let's build GPT" và "Let's build the GPT Tokenizer", khóa NLP của HuggingFace.

**Hoàn thành khi:** bạn giải thích được một lần forward của Transformer và đã chạy được một model mã nguồn mở.

## GĐ4: RAG và đánh giá (4–5 tuần) [Cốt lõi]

**Mục tiêu:** xây hệ thống hỏi đáp trên tài liệu riêng và chứng minh được chất lượng bằng số liệu.

- [ ] Vector database: Chroma, Qdrant, FAISS
- [ ] Tìm kiếm xấp xỉ (ANN): HNSW, IVF, đánh đổi giữa tốc độ, bộ nhớ và độ chính xác
- [ ] Chunking: theo ký tự, theo cấu trúc, semantic chunking
- [ ] Embedding model, tìm kiếm ngữ nghĩa
- [ ] Hybrid search (BM25 + vector), reranking
- [ ] Query rewriting, HyDE, multi-query
- [ ] Xử lý PDF, bảng biểu, OCR
- [ ] RAG nâng cao: GraphRAG, text-to-SQL trên dữ liệu có cấu trúc, RAG đa phương thức, cập nhật dữ liệu định kỳ
- [ ] Xây bộ dữ liệu đánh giá (câu hỏi, đáp án chuẩn, đoạn nguồn)
- [ ] Đo: context precision/recall, faithfulness, answer relevancy (Ragas hoặc tự viết)
- [ ] Tự viết bản RAG thuần trước, sau đó mới dùng LangChain hoặc LlamaIndex

**Dự án:** `projects/03-doc-qa-rag/`, hỏi đáp trên bộ tài liệu thật, có giao diện và bộ đánh giá.

**Hoàn thành khi:** bạn có số liệu cho thấy một thay đổi (chunk size, reranker...) cải thiện hay làm xấu kết quả.

> Nếu muốn có sản phẩm chạy được sớm cho đỡ nản, bạn có thể làm bản RAG đơn giản ngay sau GĐ1, rồi quay lại làm đầy đủ ở GĐ4.

## GĐ5: Agents (3–4 tuần) [Cốt lõi]

**Mục tiêu:** xây hệ thống LLM biết dùng công cụ để hoàn thành tác vụ nhiều bước.

- [ ] Function calling và tool use
- [ ] Vòng lặp agent: lập kế hoạch, hành động, quan sát
- [ ] MCP (Model Context Protocol): viết một MCP server
- [ ] Memory và quản lý ngữ cảnh (context engineering): cắt, tóm tắt, chọn thông tin đưa vào prompt
- [ ] Prompt caching, human-in-the-loop, workflow chạy dài có thể khôi phục
- [ ] Multi-agent và orchestration (mức cơ bản)
- [ ] Đánh giá agent: bộ tác vụ có đáp án, đo tỉ lệ thành công
- [ ] Guardrails, prompt injection, quyền hạn của tool

**Dự án:** `projects/04-research-agent/`, agent giải quyết một tác vụ thật và có bộ test đo độ tin cậy.

**Hoàn thành khi:** agent chạy ổn định trên bộ test của bạn và bạn biết nó thất bại ở đâu.

## GĐ6: Fine-tuning (3–4 tuần) [Cốt lõi]

**Mục tiêu:** tinh chỉnh LLM cho một tác vụ hoặc miền cụ thể.

- [ ] Khi nào nên fine-tune, khi nào dùng RAG hoặc prompt
- [ ] Chuẩn bị dataset: định dạng instruction, làm sạch, tạo dữ liệu tổng hợp
- [ ] SFT với `trl`
- [ ] LoRA, QLoRA với `peft` và `bitsandbytes`
- [ ] Preference tuning: DPO (RL kiểu GRPO chỉ cần nắm khái niệm)
- [ ] Đánh giá trước và sau fine-tune trên bộ test riêng
- [ ] Merge adapter, lượng tử hóa (GGUF, AWQ), đẩy lên HuggingFace Hub

**Dự án:** `projects/05-finetuned-domain-llm/`.

**Về phần cứng:** mặc định dùng model **1–4B** với QLoRA trên Colab hoặc Kaggle. Model 7–8B chạy rất sát giới hạn bộ nhớ của GPU T4 miễn phí, nên chỉ thử khi có GPU lớn hơn (thuê theo giờ).

**Hoàn thành khi:** model sau fine-tune thắng model gốc trên bộ đánh giá riêng của bạn.

## GĐ7: MLOps và LLMOps (4–5 tuần) [Cốt lõi]

**Mục tiêu:** đưa hệ thống vào chạy thật và theo dõi được nó.

- [ ] FastAPI cho model serving, thiết kế API, streaming
- [ ] Docker, docker-compose
- [ ] Theo dõi thí nghiệm và model: MLflow hoặc Weights & Biases
- [ ] Quản lý dữ liệu và model: DVC, HuggingFace Hub
- [ ] CI/CD với GitHub Actions, test cho code ML
- [ ] Serving LLM: vLLM hoặc Ollama, batching, giới hạn tốc độ
- [ ] LLMOps: tracing (Langfuse hoặc LangSmith), theo dõi chi phí token và độ trễ, caching
- [ ] Monitoring: log, cảnh báo, data drift
- [ ] Đánh giá trực tuyến: A/B testing, thu phản hồi người dùng
- [ ] Thiết kế hệ thống AI: độ trễ, chi phí, mở rộng, fallback giữa các model, xử lý lỗi
- [ ] Bảo mật và quyền riêng tư: PII, quản lý API key, prompt injection ở tầng hệ thống
- [ ] Cloud cơ bản (AWS, GCP hoặc Azure); Kubernetes chỉ cần khái niệm

**Dự án:** `projects/06-llm-service/`, đóng gói dự án RAG hoặc Agent thành dịch vụ chạy trong container, có CI và dashboard theo dõi.

**Hoàn thành khi:** dịch vụ chạy bằng một lệnh `docker compose up`, có test tự động và bạn xem được chi phí cùng độ trễ của từng request.

## GĐ8: Big Data (3–4 tuần) [Tùy chọn một phần]

**Mục tiêu:** xử lý dữ liệu vượt quá bộ nhớ một máy và xây pipeline dữ liệu.

Nên học (gần như vị trí nào cũng cần):
- [ ] Định dạng cột: Parquet, Arrow
- [ ] DuckDB cho phân tích cục bộ quy mô lớn
- [ ] Điều phối pipeline: Airflow hoặc Prefect

Học nếu vị trí hướng tới dữ liệu lớn:
- [ ] PySpark: DataFrame, partition, shuffle
- [ ] Kafka: producer, consumer, streaming cơ bản
- [ ] Data lake, lakehouse (Delta Lake, Iceberg) ở mức khái niệm

**Dự án:** `projects/07-data-pipeline/`, pipeline chạy theo lịch, đọc dữ liệu thô và ghi ra Parquet đã xử lý.

**Hoàn thành khi:** pipeline chạy tự động theo lịch và bạn giải thích được vì sao chọn công cụ đó cho khối lượng dữ liệu của mình.

## Mở rộng [Tùy chọn] (`09-optional/`)

Chỉ học khi mục tiêu hoặc dự án của bạn cần:
- Computer Vision nâng cao: object detection (YOLO), segmentation
- Mô hình sinh ảnh: diffusion
- Xử lý giọng nói: Whisper, TTS
- Reinforcement Learning
- Time series
- Recommender system
- Nén và xuất model: ONNX, pruning, distillation
- Pretraining model từ đầu ở quy mô nhỏ

## Phạm vi và giới hạn

Làm hết GĐ0–8 cùng các dự án, bạn có đủ nền tảng của một AI Engineer ứng dụng ở mức mới vào nghề: hiểu ML/DL, hiểu LLM bên trong, và xây, đánh giá, triển khai được hệ thống RAG, agent, fine-tuning.

Roadmap **không** bao gồm:
- Nghiên cứu mô hình mới và huấn luyện LLM quy mô lớn (cần nhiều GPU và nền toán sâu hơn)
- Chuyên sâu một lĩnh vực hẹp (thị giác máy tính, giọng nói, robot...), xem phần mở rộng
- Kinh nghiệm vận hành hệ thống thật ở quy mô lớn, thứ chỉ có khi chạy sản phẩm thật trong thời gian dài

Đọc roadmap không thay thế được việc làm. Độ sâu đến từ các dự án cuối giai đoạn, nên đừng bỏ qua.

## Chọn nhánh theo mục tiêu

| Mục tiêu | Giữ | Rút gọn hoặc bỏ | Đẩy sớm |
|---|---|---|---|
| **AI/LLM Engineer ứng dụng** (mặc định) | GĐ0–7 | GĐ8 chỉ học phần "nên học" | GĐ4, GĐ5 |
| **ML Engineer / Data Scientist** | GĐ0–3, 6–8 | GĐ5 rút gọn | GĐ1, GĐ2, GĐ8 đầy đủ, thêm `09-optional` |
| **Research / nghiên cứu** | GĐ0–3, 6 | GĐ7, GĐ8 | Toán sâu hơn, đọc paper, RL, pretraining |

## Quy tắc làm việc

1. **Mỗi tuần** có một mục tiêu cụ thể và một thứ chạy được.
2. **Notebook để khám phá, `src/` để lưu code đã chốt.**
3. **Tự cài đặt một lần, rồi mới dùng thư viện.** Áp dụng cho gradient descent, backprop, attention, RAG.
4. **Đo trước khi tối ưu.** Luôn có baseline và bộ đánh giá, đặc biệt với RAG, agent và fine-tuning.
5. **Mỗi giai đoạn kết thúc bằng một dự án** có README ghi rõ vấn đề, cách làm, kết quả và bài học.
6. **Không đẩy dữ liệu và model lớn lên git.** Dùng `.gitignore`, DVC hoặc HuggingFace Hub.
7. **Đọc ít nhất một paper mỗi tháng,** tóm tắt vào `notes/`. Đọc tài liệu tiếng Anh là thói quen cần tập từ đầu.

## Lịch gợi ý theo tháng

| Tháng | Nội dung |
|---|---|
| 1 | GĐ0; làm quen Git và test ngay từ tuần đầu |
| 2 | Hết GĐ0, bắt đầu GĐ1 |
| 3 | GĐ1, học thêm SQL và DuckDB |
| 4 | GĐ2 |
| 5 | Hết GĐ2, bắt đầu GĐ3; thử bọc model thành API bằng FastAPI và Docker |
| 6 | GĐ3 |
| 7 | GĐ4 (RAG) |
| 8 | GĐ5 (Agents) |
| 9 | GĐ6 (Fine-tuning) |
| 10 | GĐ7 (MLOps/LLMOps), bắt đầu GĐ8 vào cuối tháng |
| 11 | GĐ8 (Big Data) |
| 12–13 | Dự phòng cho phần học chậm hơn dự kiến, ôn lại, làm dự án tổng hợp ghép RAG, agent và triển khai |

Lịch này chỉ là ước tính. Nếu một giai đoạn trượt thì dời các giai đoạn sau, đừng cắt bớt bài tập tự cài đặt hoặc dự án cuối giai đoạn.

## Theo dõi tiến độ

Tick vào các checkbox ở trên khi hoàn thành. Cuối mỗi tháng, ghi lại ngắn gọn:

- Đã học được gì
- Đã làm ra gì (link commit hoặc dự án)
- Điều còn mơ hồ cần ôn lại
- Mục tiêu tháng sau
