# Roadmap AI Engineer

Lộ trình này bám theo [AI Engineer Roadmap của roadmap.sh](https://roadmap.sh/ai-engineer): cùng thứ tự, cùng các mục, tên mục giữ nguyên tiếng Anh để dễ tra cứu. Phần tiếng Việt chỉ là giải thích ngắn. Đối chiếu với phiên bản roadmap.sh cập nhật ngày 16/10/2026.

**AI Engineer làm gì:** dùng model đã huấn luyện sẵn và các công cụ AI có sẵn để xây sản phẩm. Khác với ML Engineer hay AI Researcher, AI Engineer thường không tạo model mới từ đầu.

**Cách đọc:** học từ trên xuống. Mỗi phần có các chủ đề chính (tiêu đề `###`) và các mục con để tick khi xong.

## Tổng quan

| Phần | Nhóm | Chủ đề chính |
|---|---|---|
| 0 | Điều kiện tiên quyết | Frontend, Backend hoặc Full-Stack (chọn một) |
| 1 | Introduction | AI Engineer là gì, vai trò, so sánh với ML Engineer |
| 2 | Working With LLMs | How LLMs Work → Prompt Engineering → Context Engineering |
| 3 | AI Models | Type of Models → Choosing the Right Model |
| 4 | Embeddings & Vector Databases | What are Embeddings → Embedding Models → Vector Databases |
| 5 | RAGs | What are RAGs? |
| 6 | AI Agents | AI Agents → Model Context Protocol (MCP) |
| 7 | AI Safety and Ethics | AI Safety and Ethics |
| 8 | Evaluation & Observability | LLM Observability → LLM Evaluations → Regression Testing |
| 9 | Other AI Applications | Multimodal AI → Development Tools |

## Phần 0: Điều kiện tiên quyết

Biết một trong các hướng sau, vì AI Engineer trước hết vẫn là người xây phần mềm:

- [ ] [Frontend](https://roadmap.sh/frontend)
- [ ] [Backend](https://roadmap.sh/backend)
- [ ] [Full-Stack](https://roadmap.sh/full-stack)

## Phần 1: Introduction

- [ ] What is an AI Engineer? (AI Engineer là gì)
- [ ] Roles and Responsibilities (vai trò và trách nhiệm)
- [ ] Impact on Product Development (tác động lên quá trình phát triển sản phẩm)
- [ ] AI Engineer vs ML Engineer (khác nhau ở đâu)

## Phần 2: Working With LLMs

### How LLMs Work

**Core LLM Elements** (thành phần cốt lõi):

- [ ] Tokens (đơn vị văn bản mà model xử lý)
- [ ] Context (thông tin model nhìn thấy khi trả lời)
- [ ] Sampling Parameters (tham số điều khiển cách chọn token tiếp theo)
  - [ ] Temperature (độ ngẫu nhiên)
  - [ ] Top-K (chỉ chọn trong K token xác suất cao nhất)
  - [ ] Top-P (chỉ chọn trong nhóm token có tổng xác suất đạt P)
  - [ ] Repetition Penalties (giảm lặp từ)

**Common Terminology** (thuật ngữ phổ biến):

- [ ] AI vs AGI
- [ ] Large Language Model (LLM)
- [ ] Embeddings
- [ ] Training (huấn luyện)
- [ ] Inference (suy luận, tức chạy model để lấy kết quả)
- [ ] Vector DBs
- [ ] AI Agents
- [ ] RAGs
- [ ] Context Window (giới hạn lượng token trong một lần gọi)
- [ ] Fine-tuning (tinh chỉnh model cho tác vụ riêng)
- [ ] Prompt Engineering
- [ ] Inference Engineering (phục vụ model nhanh hơn, rẻ hơn, ổn định hơn)
- [ ] Context Engineering

### Prompt Engineering

Xem thêm roadmap riêng: [Prompt Engineering](https://roadmap.sh/prompt-engineering).

**Prompt Anatomy** (cấu trúc của prompt):

- [ ] Input Format (định dạng đầu vào)
- [ ] System Prompting
  - [ ] Role & Behavior (vai trò và hành vi)
  - [ ] Context (bối cảnh)
  - [ ] Constraints (ràng buộc)
  - [ ] Structured Output (đầu ra có cấu trúc, ví dụ JSON)

**Prompting Techniques** (kỹ thuật viết prompt):

- [ ] Zero-Shot (không cho ví dụ)
- [ ] Few-Shot (cho vài ví dụ)
- [ ] ReAct (xen kẽ suy luận và hành động)
- [ ] CoT (Chain of Thought, suy luận từng bước)

**Model Interaction** (tương tác với model):

- [ ] Function Calling (để model gọi hàm của bạn)
- [ ] Prompt Caching (dùng lại phần prompt đã xử lý để giảm chi phí và độ trễ)
- [ ] Streaming Responses (trả kết quả dần theo thời gian thực)

### Context Engineering

Chọn và sắp xếp đúng thông tin đưa cho model.

**Fundamentals** (cơ bản):

- [ ] Context vs Prompt Engineering (khác nhau ở đâu)
- [ ] What is a Context Layer? (tầng thu thập và chọn thông tin trước khi đưa cho model)
- [ ] Context Sources (nguồn ngữ cảnh)
  - [ ] MCP
- [ ] Context Security (bảo mật ngữ cảnh)
- [ ] Context Evaluation (đánh giá chất lượng ngữ cảnh)

**Context Engineering Techniques** (kỹ thuật):

- [ ] RAG & Dynamic Filters (truy xuất kèm bộ lọc động)
- [ ] Memory Systems (hệ thống bộ nhớ)
- [ ] Context Compaction (nén ngữ cảnh)
- [ ] Long-Context Processing (xử lý ngữ cảnh dài)
- [ ] State & Historical Context (trạng thái và lịch sử)
- [ ] Multi-agent Context Sharing (chia sẻ ngữ cảnh giữa các agent)
- [ ] Context Isolation (tách biệt ngữ cảnh)
- [ ] Context Failure Modes (các kiểu lỗi do ngữ cảnh)

**Tools & Context Warehouses** (công cụ):

- [ ] modus
- [ ] DataHub
- [ ] Atlan
- [ ] PostHog

## Phần 3: AI Models

### Type of Models

- [ ] Pre-trained Models (model đã huấn luyện sẵn)
- [ ] Closed vs Open Source Models (model đóng và model mã nguồn mở)
- [ ] Self-Hosted Models (model tự vận hành trên hạ tầng riêng)

### Choosing the Right Model

- [ ] Choosing the Right Model (cân nhắc độ khó của tác vụ, chi phí, độ chính xác và tốc độ)

**Closed Models** (model đóng):

- [ ] Anthropic Claude
- [ ] Google Gemini
- [ ] OpenAI (GPT, o-series)
- [ ] Cohere
- [ ] Mistral

**Open Source Models** (model mã nguồn mở):

- [ ] Meta Llama
- [ ] DeepSeek
- [ ] Qwen
- [ ] Gemma

**Platforms & Ecosystem** (nền tảng):

- [ ] Hugging Face
  - [ ] Hugging Face Tasks
  - [ ] Hugging Face Hub
  - [ ] Transformers.js
- [ ] Ollama (chạy model trên máy cá nhân)
- [ ] LM Studio (ứng dụng desktop để chạy model trên máy cá nhân)
- [ ] OpenRouter (một API chung cho nhiều nhà cung cấp model)

**APIs & SDKs**:

- [ ] OpenAI Response API
- [ ] Claude Messages API
- [ ] Google Gemini API
- [ ] Hugging Face Inference SDK
- [ ] OpenAI-compatible APIs (API tương thích chuẩn của OpenAI)

## Phần 4: Embeddings & Vector Databases

### What are Embeddings

**Use Cases for Embeddings** (ứng dụng):

- [ ] Semantic Search (tìm kiếm theo ý nghĩa)
- [ ] Data Classification (phân loại dữ liệu)
- [ ] Recommendation Systems (hệ thống gợi ý)
- [ ] Anomaly Detection (phát hiện bất thường)

### Embedding Models

**Proprietary Models** (model thương mại):

- [ ] OpenAI Embeddings API
- [ ] Gemini Embedding
- [ ] Cohere

**Open Source Models** (model mã nguồn mở):

- [ ] Sentence Transformers
- [ ] Models on Hugging Face
- [ ] Jina

### Vector Databases

- [ ] Purpose and Functionality (mục đích và chức năng)

**Popular Vector DBs** (chọn một để học):

- [ ] Chroma
- [ ] Pinecone
- [ ] Weaviate
- [ ] FAISS
- [ ] LanceDB
- [ ] Qdrant
- [ ] Supabase
- [ ] MongoDB Atlas

**Implementing Vector Search** (triển khai tìm kiếm vector):

- [ ] Indexing Embeddings (lập chỉ mục)
- [ ] Performing Similarity Search (tìm kiếm tương đồng)

## Phần 5: RAGs

### What are RAGs?

- [ ] RAG Usecases (khi nào nên dùng)
- [ ] RAG vs Fine-tuning (chọn cách nào)

**Implementing RAG** (các bước triển khai):

- [ ] Chunking (chia nhỏ tài liệu)
- [ ] Embedding (chuyển đoạn văn thành vector)
- [ ] Vector Database (lưu vector)
- [ ] Retrieval Process (truy xuất các đoạn liên quan)
- [ ] Generation (sinh câu trả lời từ các đoạn đã truy xuất)

**Ways of Implementing RAG** (cách xây dựng):

- [ ] Using SDKs Directly (dùng trực tiếp SDK)
- [ ] Langchain
- [ ] LlamaIndex
- [ ] Haystack
- [ ] RAGFlow

Tài liệu trong repo: [RAG Từ Cơ Bản Đến Thực Chiến](<rag/RAG Từ Cơ Bản Đến Thực Chiến - Xây Hệ Thống AI Biết Tra Cứu Dữ Liệu.pdf>).

## Phần 6: AI Agents

### AI Agents

Xem thêm roadmap riêng: [AI Agents](https://roadmap.sh/ai-agents).

- [ ] Agents Usecases (khi nào nên dùng agent)
- [ ] ReAct Prompting
- [ ] Tools & Function Calling
- [ ] Multi-agents

**Building AI Agents** (cách xây dựng):

- [ ] Manual Implementation (tự viết không dùng framework)
- [ ] OpenAI AgentKit & Agent SDK
- [ ] Claude Agent SDK
- [ ] Vertex AI Agent Builder
- [ ] Google ADK

### Model Context Protocol (MCP)

Giao thức chuẩn để kết nối ứng dụng AI với công cụ và dữ liệu bên ngoài.

**Core Components** (thành phần cốt lõi):

- [ ] MCP Host
- [ ] MCP Client
- [ ] MCP Server
- [ ] Data Layer
- [ ] Transport Layer

**Developing with MCP** (phát triển với MCP):

- [ ] Building an MCP Server
- [ ] Building an MCP Client
- [ ] Connect to Local Server
- [ ] Connect to Remote Server

## Phần 7: AI Safety and Ethics

### AI Safety and Ethics

**Understanding AI Safety Issues** (hiểu các vấn đề an toàn):

- [ ] Prompt Injection Attacks (tấn công chèn lệnh qua prompt)
- [ ] Security and Privacy Concerns (bảo mật và quyền riêng tư)
- [ ] Bias and Fairness (thiên lệch và công bằng)

**Safety Best Practices** (thực hành tốt):

- [ ] Content Moderation APIs (API kiểm duyệt nội dung)
- [ ] Adding end-user IDs in prompts (gắn ID người dùng để truy vết và ngăn lạm dụng)
- [ ] Conducting adversarial testing (kiểm thử đối kháng)
- [ ] Robust prompt engineering (viết prompt khó bị lợi dụng)
- [ ] Know your Customers / Usecases (hiểu người dùng và tình huống sử dụng)
- [ ] Constraining outputs and inputs (giới hạn đầu vào và đầu ra)

## Phần 8: Evaluation & Observability

### LLM Observability

Theo dõi hệ thống khi chạy: đã gửi prompt gì, nhận về gì, mất bao lâu, tốn bao nhiêu token.

- [ ] Tracing & logging
- [ ] Cost/latency monitoring (theo dõi chi phí và độ trễ)
- [ ] Production monitoring (giám sát môi trường production)

**Observability Tools** (công cụ):

- [ ] LangSmith
- [ ] Langfuse
- [ ] Helicone
- [ ] Arize AI

### LLM Evaluations

Đo chất lượng bằng bộ kiểm thử lặp lại được, thay vì đánh giá cảm tính.

**Evaluation Types** (các loại đánh giá):

- [ ] Deterministic Evals (kiểm tra bằng luật cố định)
- [ ] Model-Based Evals (dùng model để chấm)
- [ ] Human Evals (người chấm)
- [ ] Evaluation Metrics (các chỉ số đánh giá)

**Evaluation Tools** (công cụ):

- [ ] DeepEval
- [ ] RAGAS

### Regression Testing

- [ ] Regression Testing (đảm bảo thay đổi prompt, model hoặc cách truy xuất không làm hỏng các trường hợp đang chạy tốt)

## Phần 9: Other AI Applications

### Multimodal AI

Xử lý và sinh nhiều loại dữ liệu: văn bản, ảnh, âm thanh, video.

- [ ] Multimodal AI Usecases (ứng dụng)

**Multimodal AI Tasks** (các tác vụ):

- [ ] Image Understanding (hiểu ảnh)
- [ ] Image Generation (sinh ảnh)
- [ ] Video Understanding (hiểu video)
- [ ] Audio Processing (xử lý âm thanh)
- [ ] Text-to-Speech (văn bản thành giọng nói)
- [ ] Speech-to-Text (giọng nói thành văn bản)

**Implementing Multimodal AI** (triển khai):

- [ ] OpenAI Vision API
- [ ] DALL-E API
- [ ] NanoBanana API
- [ ] Whisper API
- [ ] Hugging Face Models
- [ ] LangChain for Multimodal Apps
- [ ] LlamaIndex for Multimodal Apps

### Development Tools

**AI Assisted Coding Tools** (công cụ hỗ trợ lập trình bằng AI):

- [ ] Claude Code
- [ ] Gemini
- [ ] Codex
- [ ] Cursor
- [ ] OpenCode
- [ ] Replit

Xem thêm roadmap riêng: [Vibe Coding](https://roadmap.sh/vibe-coding).

## Học tiếp

Các hướng roadmap.sh gợi ý sau khi xong lộ trình này:

- [Forward Deployed Engineer](https://roadmap.sh/forward-deployed-engineer)
- [Prompt Engineering](https://roadmap.sh/prompt-engineering)
- [Inference Engineering](https://roadmap.sh/inference-engineering)

## Tài liệu khác trong repo

Các chủ đề dưới đây không nằm trong roadmap.sh AI Engineer, nhưng vẫn hữu ích nếu muốn hiểu sâu hơn hoặc đi theo hướng ML Engineer:

- [Machine Learning](machine-learning/README.md)
- [Big Data](big-data/)
- [Trí tuệ nhân tạo](artificial-intelligence/)
