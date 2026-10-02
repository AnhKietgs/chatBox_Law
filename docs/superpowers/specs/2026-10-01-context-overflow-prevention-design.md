# Design: Context Window Overflow Prevention
**Date:** 2026-10-01  
**Status:** Approved by user  
**Scope:** Backend – `backend/app/services/` và `backend/app/api/`

---

## 1. Vấn đề cần giải quyết

Pipeline RAG hiện tại không kiểm soát kích thước prompt trước khi gửi lên Ollama:

| Điểm yếu | Vị trí | Hậu quả |
|---|---|---|
| Không đếm token của sources | `answering.py` | Overflow khi provisions dài |
| Inject top-8 provisions | `retrieval.py` (default `limit=8`) | Context quá lớn |
| History giới hạn bằng char, không token | `conversation.py` | Ước tính sai với tiếng Việt |
| Không nén lịch sử cũ | `models.py`, `api/chat.py` | History tích lũy vô hạn |
| Không xử lý batch provisions dài | `answering.py` | LLM fail silently khi document lớn |

---

## 2. Mục tiêu thiết kế

1. **Không bao giờ gửi prompt vượt quá budget token** đã cấu hình.
2. **Recall không giảm** — vẫn retrieve rộng (top-20), chỉ cắt ở bước inject.
3. **Lịch sử hội thoại không tích lũy vô hạn** — sliding window + memory summary.
4. **Document siêu dài vẫn được xử lý** — Map-Reduce qua nhiều batch.
5. **Không thêm dependency bắt buộc mới** — dùng char-based estimate (len/3.5) phù hợp tiếng Việt; `tiktoken` là optional.

---

## 3. Kiến trúc tổng thể

```
User question
     │
     ▼
[Sliding Window + Memory Summary]  ← models.Conversation.memory_summary (DB)
     │  conversation_context = summary_paragraph + N recent turns
     ▼
[HybridVectorStore]  retrieve top-{retrieval_recall_pool} candidates (default 20)
     │
     ▼
[Reranker]  top-40 → top-K_inject (default 3) best provisions
     │
     ▼
[TokenBudget.fit_sources_to_budget()]
     │ under budget ──────────────────────────────────┐
     │ over budget                                     ▼
     ▼                                        [Ollama LLM → grounded answer]
[MapReduceSummarizer]
  batch provisions → LLM summarize each batch
  → merge summaries → single condensed source list
     │
     └──────────────────────────────────────────────▶ [Ollama LLM]
```

---

## 4. Các module mới / sửa đổi

### 4.1 `token_budget.py` ← **MỚI**

**Mục đích:** Ước tính token và đảm bảo prompt không vượt budget.

```python
# Interface công khai
class TokenBudget:
    def __init__(self, model_context_limit: int, reserved_output: int, overhead: int): ...
    def estimate(self, text: str) -> int: ...       # len(text) / 3.5, ceil
    def remaining(self, used: int) -> int: ...
    def fit_sources_to_budget(
        self,
        sources: list[dict],           # {"source_id", "citation", "text", "score"}
        fixed_prompt_text: str,        # system_prompt + conversation_context + question
    ) -> tuple[list[dict], bool]:      # (fitted_sources, was_truncated)
```

**Thuật toán `fit_sources_to_budget`:**
1. Tính token của `fixed_prompt_text`.
2. Tính token còn lại = `model_context_limit - reserved_output - overhead - fixed_tokens`.
3. Lần lượt thêm source theo thứ tự score cao → thấp cho đến khi hết budget.
4. Trả về danh sách fitted và cờ `was_truncated`.

**Token estimate:** `ceil(len(text) / 3.5)` — bảo thủ cho tiếng Việt UTF-8 (trung bình ~3-4 bytes/token).

**Settings mới:**
```
context_model_limit: int = 8192          # token limit của model
context_reserved_output: int = 1024      # giữ cho phần sinh ra
context_overhead: int = 256              # system markers, JSON wrapper
```

---

### 4.2 `map_reduce.py` ← **MỚI**

**Mục đích:** Xử lý danh sách provisions khi tổng vượt budget — map từng batch qua LLM, reduce thành 1 summary.

```python
@dataclass(frozen=True)
class MapReduceResult:
    merged_text: str          # nội dung đã tóm tắt, fit vào budget
    batch_count: int
    total_input_tokens: int

class MapReduceSummarizer:
    async def summarize(
        self,
        question: str,
        sources: list[dict],       # danh sách source objects
        target_token_budget: int,  # token tối đa cho output
    ) -> MapReduceResult: ...
```

**Thuật toán:**
1. **Map:** chia `sources` thành batches, mỗi batch ≤ `map_batch_token_limit` token.  
   Với mỗi batch, gọi Ollama để tóm tắt các điều khoản liên quan đến `question`.
2. **Reduce:** ghép tất cả batch-summaries lại, nếu vẫn còn > budget thì gọi Ollama lần nữa để merge.
3. **Output:** một đoạn văn bản duy nhất, fit trong `target_token_budget`.

**Settings mới:**
```
map_reduce_enabled: bool = False     # mặc định tắt, bật trong .env khi cần
map_batch_token_limit: int = 2000        # token tối đa mỗi batch trong map phase
map_reduce_timeout_seconds: float = 60.0
```

---

### 4.3 `memory.py` ← **MỚI**

**Mục đích:** Tóm tắt các turns cũ thành 1 paragraph ngắn để giảm context.

```python
@dataclass(frozen=True)
class MemoryUpdate:
    summary: str
    turns_summarized: int
    llm_ms: float

async def summarize_history(
    messages: list[ChatMessage],   # các turns CŨ (ngoài sliding window)
    existing_summary: str | None,  # summary trước đó nếu có
) -> MemoryUpdate: ...
```

**Thuật toán:**
1. Lấy tất cả messages ngoài `history_window_turns` gần nhất.
2. Nếu `existing_summary` tồn tại, concat với messages mới cũ hơn.
3. Gọi Ollama để tóm tắt thành ≤150 token (khoảng 1 paragraph).
4. Lưu kết quả vào `Conversation.memory_summary`.

**Settings mới:**
```
history_window_turns: int = 5            # số turns gần nhất giữ nguyên
memory_summary_enabled: bool = True
memory_summary_max_tokens: int = 150
```

---

### 4.4 `conversation.py` ← **SỬA**

**Thay đổi trong `build_conversation_context`:**
- Tham số mới: `memory_summary: str | None`, `window_turns: int`
- Output: `f"{memory_summary}\n\n{recent_turns_text}"` — summary làm prefix, window làm suffix.
- Giữ backward compatibility: khi `memory_summary=None` thì giống hệt hành vi cũ.

---

### 4.5 `retrieval.py` ← **SỬA**

**Thay đổi:**
- `recall_pool_size` (settings): số candidates retrieve (default **20**, từ `limit*2`).
- `inject_limit` (settings): số provisions inject vào prompt (default **5**).
- `LegalRetriever.retrieve_with_metrics` nhận `inject_limit` riêng, tách khỏi `recall_pool_size`.
- Reranker vẫn sort top-N, chỉ inject_limit đầu được đưa vào answering.

**Settings mới:**
```
retrieval_recall_pool: int = 20      # candidates để rerank
retrieval_inject_limit: int = 5      # provisions inject vào prompt
```

---

### 4.6 `answering.py` ← **SỬA**

**Thay đổi trong `GroundedAnswerService.generate`:**
1. Sau khi build `sources` list, gọi `TokenBudget.fit_sources_to_budget()`.
2. Nếu `was_truncated=True` và `map_reduce_enabled`, gọi `MapReduceSummarizer.summarize()` thay vì cắt bỏ.
3. Log warning nếu truncated; log info nếu map-reduce activated.
4. Field `LatencyBreakdown` thêm `map_reduce_ms: float = 0`.

---

### 4.7 `models.py` ← **SỬA**

Thêm column vào `Conversation`:
```python
memory_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
```
Cần Alembic migration hoặc `ALTER TABLE` thủ công.

---

### 4.8 `config.py` ← **SỬA**

Thêm 9 settings mới (xem tổng hợp bên dưới).

---

### 4.9 `api/chat.py` ← **SỬA**

Sau khi `save_turn()`, kiểm tra xem có cần trigger memory summarization không:
```python
total_turns = db.scalar(select(func.count(ChatMessage.id)).where(...))
if settings.memory_summary_enabled and total_turns > settings.history_window_turns * 2:
    memory_update = await summarize_history(old_messages, conversation.memory_summary)
    conversation.memory_summary = memory_update.summary
    db.commit()
```

---

### 4.10 `schemas.py` ← **SỬA**

Thêm field vào `LatencyBreakdown`:
```python
map_reduce_ms: float = 0
token_budget_ms: float = 0
```

---

## 5. Tổng hợp 11 Settings mới trong `config.py`

| Key | Default | Mô tả |
|---|---|---|
| `context_model_limit` | `8192` | Token limit của Ollama model |
| `context_reserved_output` | `1024` | Token giữ lại cho output |
| `context_overhead` | `256` | Overhead JSON/system markers |
| `retrieval_recall_pool` | `20` | Pool retrieve để rerank |
| `retrieval_inject_limit` | `5` | Provisions inject vào prompt |
| `map_reduce_enabled` | `False` | Bật/tắt Map-Reduce (mặc định tắt, bật khi cần) |
| `map_batch_token_limit` | `2000` | Token tối đa mỗi map batch |
| `map_reduce_timeout_seconds` | `60.0` | Timeout mỗi LLM call trong MR |
| `history_window_turns` | `5` | Số turns gần nhất giữ nguyên |
| `memory_summary_enabled` | `True` | Bật/tắt memory summary |
| `memory_summary_max_tokens` | `150` | Token tối đa của summary |

---

## 6. Database migration

Chỉ có 1 thay đổi schema: thêm column `memory_summary TEXT NULL` vào bảng `conversations`.

Migration script (raw SQL, Alembic-compatible):
```sql
ALTER TABLE conversations ADD COLUMN memory_summary TEXT;
```
Column nullable → không ảnh hưởng rows hiện có.

---

## 7. Chiến lược test

| Module | Test cases |
|---|---|
| `token_budget.py` | estimate chính xác, fit cắt đúng source, fit không cắt khi under budget |
| `map_reduce.py` | mock LLM, 1 batch, 3 batches, reduce phase kích hoạt |
| `memory.py` | summary ngắn hơn input, existing_summary được include |
| `conversation.py` | backward compat (no summary), prefix + suffix output |
| `retrieval.py` | inject_limit < recall_pool, config override |
| `answering.py` | token OK path, truncated → map-reduce path, truncated → cắt path |

Tất cả dùng `pytest` + mock Ollama responses, không cần infra thật.

---

## 8. Thứ tự triển khai

1. `token_budget.py` + tests (không dependency)
2. `map_reduce.py` + tests (depends: token_budget)
3. `memory.py` + tests (depends: Ollama mock)
4. `models.py` + migration SQL
5. `config.py` (add settings)
6. `schemas.py` (add fields)
7. `conversation.py` (add memory_summary param)
8. `retrieval.py` (recall_pool / inject_limit split)
9. `answering.py` (integrate token_budget + map_reduce)
10. `api/chat.py` (integrate memory + pass inject_limit)

---

## 9. Ràng buộc và quyết định thiết kế

- **Không dùng `tiktoken`** (cần OpenAI API key / thêm dependency). Char-estimate đủ chính xác cho mục đích safety margin.
- **Map-Reduce chỉ là fallback** — happy path là token budget đủ sau khi inject top-3.
- **Memory summary không block request** — nếu LLM summarize fail, log warning và tiếp tục với history thô.
- **Conversation.memory_summary** là nullable — rows cũ không bị ảnh hưởng.
- **Backward compatibility**: tất cả settings đều có default, không cần sửa `.env` để chạy.
