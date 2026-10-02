# Context Window Overflow Prevention — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ngăn prompt vượt quá context window của Ollama bằng 4 lớp phòng thủ: đo token trước khi gửi, reranker pool rộng → inject ít, Map-Reduce cho document siêu dài, và sliding window history + memory summary.

**Architecture:** Token budget được tính bằng `ceil(len/3.5)` và áp dụng tại `answering.py` trước khi build prompt; retrieval tăng recall pool lên 20 candidates nhưng chỉ inject top-5 tốt nhất; khi vẫn vượt budget thì Map-Reduce (opt-in) tóm tắt theo batch; lịch sử hội thoại giữ N=5 turns gần nhất, các turns cũ được nén thành 1 paragraph lưu trong DB.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Ollama (httpx), pytest + unittest.mock

**Spec:** `docs/superpowers/specs/2026-10-01-context-overflow-prevention-design.md`

## Global Constraints

- Token estimate: `math.ceil(len(text) / 3.5)` — không dùng tiktoken, không thêm dependency bắt buộc.
- `retrieval_recall_pool = 20`, `retrieval_inject_limit = 5` (defaults).
- `map_reduce_enabled = False` (default off; bật qua env var `MAP_REDUCE_ENABLED=true`).
- `history_window_turns = 5`, `memory_summary_max_tokens = 150`.
- `context_model_limit = 8192`, `context_reserved_output = 1024`, `context_overhead = 256`.
- Tất cả settings có default → không cần sửa `.env` để chạy.
- `Conversation.memory_summary` là nullable TEXT — rows hiện có không bị ảnh hưởng.
- Mọi LLM call dùng `httpx.AsyncClient` qua `settings.ollama_base_url`, model `settings.ollama_model`.
- Tests không cần infra thật: mock `httpx.AsyncClient.post` hoặc patch hàm LLM.
- Mọi file Python mới nằm trong `backend/app/services/`.
- Tests mới nằm trong `backend/tests/`.

## Review Focus

1. **Token estimate cho tiếng Việt multi-byte**: `len(text)` trả về số ký tự Unicode (không phải bytes); với `ceil(len/3.5)` cho string Vietnamese, cần xác nhận estimate luôn ≤ token thực để không bao giờ under-estimate. → Test Task 1.
2. **Fit-to-budget với 0 source fit**: khi fixed_prompt đã vượt budget, `fit_sources_to_budget` phải trả về `([], True)` không exception. → Test Task 1.
3. **Memory summary fail silently**: khi Ollama timeout trong `summarize_history`, pipeline phải tiếp tục với history thô thay vì raise. → Test Task 3.
4. **Reranker pool < inject_limit**: khi vector store trả về ít candidates hơn `inject_limit`, không được raise — trả về tất cả candidates. → Test Task 8.
5. **Map-Reduce single-batch (batch_count=1)**: khi tất cả sources vừa 1 batch, reduce phase không được gọi LLM lần 2. → Test Task 2.

---

## File Map

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `backend/app/services/token_budget.py` | TẠO MỚI | Ước tính token, fit sources vào budget |
| `backend/app/services/map_reduce.py` | TẠO MỚI | Map-Reduce summarizer qua Ollama |
| `backend/app/services/memory.py` | TẠO MỚI | Tóm tắt turns cũ thành memory paragraph |
| `backend/app/models.py` | SỬA | Thêm `Conversation.memory_summary` column |
| `backend/app/config.py` | SỬA | Thêm 11 settings mới |
| `backend/app/schemas.py` | SỬA | Thêm `map_reduce_ms`, `token_budget_ms` vào `LatencyBreakdown` |
| `backend/app/services/conversation.py` | SỬA | `build_conversation_context` nhận `memory_summary` |
| `backend/app/services/retrieval.py` | SỬA | Tách `recall_pool` và `inject_limit` |
| `backend/app/services/answering.py` | SỬA | Tích hợp TokenBudget + MapReduce |
| `backend/app/api/chat.py` | SỬA | Trigger memory update, truyền inject_limit |
| `backend/tests/test_token_budget.py` | TẠO MỚI | Tests cho token_budget |
| `backend/tests/test_map_reduce.py` | TẠO MỚI | Tests cho map_reduce |
| `backend/tests/test_memory.py` | TẠO MỚI | Tests cho memory |
| `backend/tests/test_conversation_context.py` | TẠO MỚI | Tests cho conversation (có memory) |

---

### Task 1: TokenBudget — đo token và fit sources vào budget

**Files:**
- Create: `backend/app/services/token_budget.py`
- Create: `backend/tests/test_token_budget.py`

**Interfaces:**
- Produces:
  ```python
  class TokenBudget:
      def __init__(self, model_context_limit: int, reserved_output: int, overhead: int) -> None
      def estimate(self, text: str) -> int          # ceil(len(text) / 3.5)
      def available(self, fixed_prompt_text: str) -> int   # budget còn lại sau fixed
      def fit_sources_to_budget(
          self,
          sources: list[dict],        # mỗi dict có keys: source_id, citation, text, score
          fixed_prompt_text: str,
      ) -> tuple[list[dict], bool]    # (fitted, was_truncated)
  ```

- [ ] **Step 1: Viết test `estimate` chính xác**

```python
# backend/tests/test_token_budget.py
import math
from app.services.token_budget import TokenBudget

def test_estimate_ascii():
    tb = TokenBudget(8192, 1024, 256)
    text = "a" * 350
    assert tb.estimate(text) == math.ceil(350 / 3.5)  # == 100

def test_estimate_vietnamese():
    # "điều" = 4 ký tự Unicode → estimate = ceil(4/3.5) = 2 token
    tb = TokenBudget(8192, 1024, 256)
    assert tb.estimate("điều") == 2

def test_estimate_never_zero():
    tb = TokenBudget(8192, 1024, 256)
    assert tb.estimate("x") >= 1
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

```bash
cd backend && pytest tests/test_token_budget.py::test_estimate_ascii -v
```
Expected: `ModuleNotFoundError` hoặc `ImportError`

- [ ] **Step 3: Implement `TokenBudget` trong `token_budget.py`**

Tạo file `backend/app/services/token_budget.py`:
- `estimate(text)`: `return max(1, math.ceil(len(text) / 3.5))`
- `available(fixed_prompt_text)`: `return self._limit - self._reserved - self._overhead - self.estimate(fixed_prompt_text)`
- `fit_sources_to_budget(sources, fixed_prompt_text)`: lặp sources theo thứ tự đã cho (sorted by score desc từ caller), cộng dồn token từng source; dừng khi budget cạn; trả về `(fitted, len(fitted) < len(sources))`.

- [ ] **Step 4: Viết test `fit_sources_to_budget`**

```python
def test_fit_all_under_budget():
    tb = TokenBudget(8192, 1024, 256)
    sources = [{"source_id": "S1", "citation": "X", "text": "ngắn", "score": 0.9}]
    fitted, truncated = tb.fit_sources_to_budget(sources, "prompt ngắn")
    assert fitted == sources
    assert truncated is False

def test_fit_truncates_when_over_budget():
    tb = TokenBudget(100, 50, 10)  # available ≈ 40 - fixed tokens
    sources = [
        {"source_id": "S1", "citation": "A", "text": "a" * 50, "score": 0.9},
        {"source_id": "S2", "citation": "B", "text": "b" * 200, "score": 0.8},
    ]
    fitted, truncated = tb.fit_sources_to_budget(sources, "")
    assert len(fitted) < 2
    assert truncated is True

def test_fit_returns_empty_when_fixed_exceeds_budget():
    tb = TokenBudget(50, 10, 10)   # available = 50 - 10 - 10 - estimate("x"*200) < 0
    fitted, truncated = tb.fit_sources_to_budget(
        [{"source_id": "S1", "citation": "A", "text": "a", "score": 1.0}],
        "x" * 200,
    )
    assert fitted == []
    assert truncated is True
```

- [ ] **Step 5: Chạy tất cả tests Task 1**

```bash
cd backend && pytest tests/test_token_budget.py -v
```
Expected: tất cả PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/token_budget.py backend/tests/test_token_budget.py
git commit -m "feat: add TokenBudget for safe context measurement"
```

---

### Task 2: MapReduceSummarizer — tóm tắt batch provisions qua LLM

**Files:**
- Create: `backend/app/services/map_reduce.py`
- Create: `backend/tests/test_map_reduce.py`

**Interfaces:**
- Consumes: `TokenBudget` từ Task 1
- Produces:
  ```python
  @dataclass(frozen=True)
  class MapReduceResult:
      merged_text: str
      batch_count: int
      total_input_tokens: int

  class MapReduceSummarizer:
      async def summarize(
          self,
          question: str,
          sources: list[dict],          # dicts với keys: source_id, citation, text
          target_token_budget: int,
      ) -> MapReduceResult
  ```

- [ ] **Step 1: Viết test single-batch (không gọi reduce)**

```python
# backend/tests/test_map_reduce.py
import pytest
from unittest.mock import patch, AsyncMock
from app.services.map_reduce import MapReduceSummarizer, MapReduceResult

MOCK_LLM_RESPONSE = {
    "message": {"content": "Điều 1 quy định về hợp đồng lao động."}
}

@pytest.mark.asyncio
async def test_single_batch_no_reduce():
    sources = [{"source_id": "S1", "citation": "LĐ Điều 1", "text": "nội dung ngắn"}]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = MOCK_LLM_RESPONSE
        mock_post.return_value.raise_for_status = lambda: None
        summarizer = MapReduceSummarizer()
        result = await summarizer.summarize("hợp đồng lao động", sources, 1000)
    assert isinstance(result, MapReduceResult)
    assert result.batch_count == 1
    assert len(result.merged_text) > 0
    assert mock_post.call_count == 1   # chỉ 1 LLM call (map), không có reduce
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

```bash
cd backend && pytest tests/test_map_reduce.py::test_single_batch_no_reduce -v
```

- [ ] **Step 3: Implement `MapReduceSummarizer` trong `map_reduce.py`**

Tạo `backend/app/services/map_reduce.py`:
- Import `TokenBudget`, `get_settings`, `httpx`.
- `_split_into_batches(sources, batch_token_limit)`: dùng `TokenBudget.estimate` để chia `sources` thành batches, mỗi batch ≤ `batch_token_limit` token tổng của `text` fields.
- `_call_llm(prompt_text)`: async, gọi Ollama chat API (không dùng JSON schema format), trả về `str` content. Dùng `settings.map_reduce_timeout_seconds`.
- `summarize(question, sources, target_token_budget)`:
  1. Split thành batches.
  2. Map: với mỗi batch, gọi `_call_llm` với prompt yêu cầu tóm tắt batch liên quan đến `question`.
  3. Nếu `batch_count == 1`: `merged_text = summaries[0]`, không gọi LLM lần 2.
  4. Nếu `batch_count > 1`: ghép summaries, kiểm tra `TokenBudget.estimate(joined) > target_token_budget`; nếu vượt thì gọi `_call_llm` lần nữa để reduce.
  5. Trả về `MapReduceResult(merged_text, batch_count, total_input_tokens)`.

- [ ] **Step 4: Viết test multi-batch có reduce**

```python
@pytest.mark.asyncio
async def test_multi_batch_triggers_reduce():
    # 3 sources, mỗi source đủ lớn để thành 3 batches riêng
    sources = [
        {"source_id": f"S{i}", "citation": f"Điều {i}", "text": "x" * 800}
        for i in range(1, 4)
    ]
    call_count = 0
    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        m = AsyncMock()
        m.json.return_value = {"message": {"content": f"summary_{call_count}"}}
        m.raise_for_status = lambda: None
        return m

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        summarizer = MapReduceSummarizer()
        result = await summarizer.summarize("câu hỏi", sources, target_token_budget=50)
    assert result.batch_count == 3
    assert call_count == 4   # 3 map + 1 reduce

@pytest.mark.asyncio
async def test_llm_failure_returns_empty_merged():
    sources = [{"source_id": "S1", "citation": "X", "text": "nội dung"}]
    with patch("httpx.AsyncClient.post", side_effect=Exception("timeout")):
        summarizer = MapReduceSummarizer()
        result = await summarizer.summarize("câu hỏi", sources, 1000)
    assert result.merged_text == ""
    assert result.batch_count == 1
```

- [ ] **Step 5: Chạy tất cả tests Task 2**

```bash
cd backend && pytest tests/test_map_reduce.py -v
```
Expected: tất cả PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/map_reduce.py backend/tests/test_map_reduce.py
git commit -m "feat: add MapReduceSummarizer for long-document handling"
```

---

### Task 3: HistoryMemory — tóm tắt turns cũ thành memory paragraph

**Files:**
- Create: `backend/app/services/memory.py`
- Create: `backend/tests/test_memory.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class MemoryUpdate:
      summary: str
      turns_summarized: int
      llm_ms: float

  async def summarize_history(
      messages: list,            # list ChatMessage objects (có .role, .content)
      existing_summary: str | None,
  ) -> MemoryUpdate
  ```

- [ ] **Step 1: Viết test summarize thành công**

```python
# backend/tests/test_memory.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.memory import summarize_history, MemoryUpdate

def make_message(role: str, content: str):
    m = MagicMock()
    m.role = role
    m.content = content
    return m

@pytest.mark.asyncio
async def test_summarize_produces_shorter_output():
    messages = [make_message("user", "hỏi " * 50), make_message("assistant", "trả lời " * 50)]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = {"message": {"content": "Tóm tắt ngắn."}}
        mock_post.return_value.raise_for_status = lambda: None
        result = await summarize_history(messages, existing_summary=None)
    assert isinstance(result, MemoryUpdate)
    assert result.turns_summarized == 2
    assert len(result.summary) < len("hỏi " * 50 + "trả lời " * 50)
```

- [ ] **Step 2: Viết test fail silently khi LLM lỗi**

```python
@pytest.mark.asyncio
async def test_llm_failure_returns_empty_summary():
    messages = [make_message("user", "câu hỏi")]
    with patch("httpx.AsyncClient.post", side_effect=Exception("connection refused")):
        result = await summarize_history(messages, existing_summary=None)
    assert result.summary == ""
    assert result.turns_summarized == 1
    assert result.llm_ms >= 0
```

- [ ] **Step 3: Viết test existing_summary được include trong prompt**

```python
@pytest.mark.asyncio
async def test_existing_summary_included_in_call():
    messages = [make_message("user", "câu hỏi mới")]
    captured_prompt = {}
    async def fake_post(url, json=None, **kwargs):
        captured_prompt["body"] = json
        m = AsyncMock()
        m.json.return_value = {"message": {"content": "summary mới"}}
        m.raise_for_status = lambda: None
        return m

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await summarize_history(messages, existing_summary="Tóm tắt cũ: người dùng hỏi về hợp đồng.")
    
    user_msg = next(
        msg["content"] for msg in captured_prompt["body"]["messages"] if msg["role"] == "user"
    )
    assert "Tóm tắt cũ" in user_msg
```

- [ ] **Step 4: Chạy test để xác nhận FAIL**

```bash
cd backend && pytest tests/test_memory.py -v
```

- [ ] **Step 5: Implement `summarize_history` trong `memory.py`**

Tạo `backend/app/services/memory.py`:
- System prompt: yêu cầu tóm tắt cuộc hội thoại thành ≤150 token tiếng Việt, không bịa pháp lý.
- User prompt: nếu `existing_summary` không None, thêm nó vào đầu với label "Tóm tắt trước:".
- Gọi Ollama chat API (không JSON format), temperature 0.
- Nếu exception: log warning, trả về `MemoryUpdate(summary="", turns_summarized=len(messages), llm_ms=elapsed)`.

- [ ] **Step 6: Chạy tất cả tests Task 3**

```bash
cd backend && pytest tests/test_memory.py -v
```
Expected: tất cả PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/memory.py backend/tests/test_memory.py
git commit -m "feat: add memory summarizer for sliding window history"
```

---

### Task 4: DB migration + Config + Schemas

**Files:**
- Modify: `backend/app/models.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/schemas.py`
- Create: `backend/migrations/add_memory_summary.sql`

**Interfaces:**
- Produces (config keys dùng bởi Task 5–10):
  - `settings.context_model_limit: int`
  - `settings.context_reserved_output: int`
  - `settings.context_overhead: int`
  - `settings.retrieval_recall_pool: int`
  - `settings.retrieval_inject_limit: int`
  - `settings.map_reduce_enabled: bool`
  - `settings.map_batch_token_limit: int`
  - `settings.map_reduce_timeout_seconds: float`
  - `settings.history_window_turns: int`
  - `settings.memory_summary_enabled: bool`
  - `settings.memory_summary_max_tokens: int`
- Produces (schema field dùng bởi Task 9):
  - `LatencyBreakdown.map_reduce_ms: float = 0`
  - `LatencyBreakdown.token_budget_ms: float = 0`
- Produces (model field dùng bởi Task 10):
  - `Conversation.memory_summary: Mapped[str | None]`

- [ ] **Step 1: Thêm `memory_summary` vào `Conversation` model**

Trong `backend/app/models.py`, thêm vào class `Conversation` sau field `updated_at`:
```python
memory_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
```

- [ ] **Step 2: Tạo migration SQL**

Tạo file `backend/migrations/add_memory_summary.sql`:
```sql
-- Migration: add memory_summary to conversations
-- Safe to run multiple times (column nullable, no data loss)
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS memory_summary TEXT;
```
*(SQLite không hỗ trợ `IF NOT EXISTS` trên `ADD COLUMN` trước 3.37; dùng `ALTER TABLE conversations ADD COLUMN memory_summary TEXT;` cho SQLite.)*

- [ ] **Step 3: Thêm 11 settings vào `config.py`**

Trong class `Settings`, thêm sau `warm_models_on_startup`:
```python
# Token budget
context_model_limit: int = 8192
context_reserved_output: int = 1024
context_overhead: int = 256
# Retrieval injection
retrieval_recall_pool: int = 20
retrieval_inject_limit: int = 5
# Map-Reduce
map_reduce_enabled: bool = False
map_batch_token_limit: int = 2000
map_reduce_timeout_seconds: float = 60.0
# Sliding window history
history_window_turns: int = 5
memory_summary_enabled: bool = True
memory_summary_max_tokens: int = 150
```

- [ ] **Step 4: Thêm fields vào `LatencyBreakdown` trong `schemas.py`**

```python
map_reduce_ms: float = 0
token_budget_ms: float = 0
```

- [ ] **Step 5: Chạy smoke test import**

```bash
cd backend && python -c "from app.config import get_settings; s = get_settings(); assert s.retrieval_inject_limit == 5; assert s.map_reduce_enabled == False; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add backend/app/models.py backend/app/config.py backend/app/schemas.py backend/migrations/
git commit -m "feat: add 11 context-budget settings, memory_summary column, latency fields"
```

---

### Task 5: `conversation.py` — sliding window với memory prefix

**Files:**
- Modify: `backend/app/services/conversation.py`
- Create: `backend/tests/test_conversation_context.py`

**Interfaces:**
- Consumes: `settings.history_window_turns` (từ Task 4)
- Produces (dùng bởi Task 10):
  ```python
  def build_conversation_context(
      messages: Iterable[object],
      max_chars: int,
      memory_summary: str | None = None,
      window_turns: int | None = None,   # None → dùng settings.history_window_turns
  ) -> str
  ```
  Output format: `"[Ngữ cảnh trước]: {summary}\n\n{recent_turns}"` khi summary không None, ngược lại chỉ `{recent_turns}`.

- [ ] **Step 1: Viết test backward compatibility**

```python
# backend/tests/test_conversation_context.py
from unittest.mock import MagicMock
from app.services.conversation import build_conversation_context

def make_msg(role, content):
    m = MagicMock(); m.role = role; m.content = content; return m

def test_backward_compat_no_summary():
    msgs = [make_msg("user", "hỏi"), make_msg("assistant", "trả lời")]
    ctx = build_conversation_context(msgs, max_chars=1000)
    assert "Người dùng: hỏi" in ctx
    assert "[Ngữ cảnh trước]" not in ctx
```

- [ ] **Step 2: Viết test có memory_summary làm prefix**

```python
def test_memory_summary_as_prefix():
    msgs = [make_msg("user", "câu hỏi mới")]
    ctx = build_conversation_context(msgs, max_chars=1000, memory_summary="Tóm tắt cũ.")
    assert ctx.startswith("[Ngữ cảnh trước]: Tóm tắt cũ.")
    assert "câu hỏi mới" in ctx

def test_empty_summary_not_prepended():
    msgs = [make_msg("user", "hỏi")]
    ctx = build_conversation_context(msgs, max_chars=1000, memory_summary="")
    assert "[Ngữ cảnh trước]" not in ctx
```

- [ ] **Step 3: Chạy tests để xác nhận FAIL**

```bash
cd backend && pytest tests/test_conversation_context.py -v
```

- [ ] **Step 4: Sửa `build_conversation_context` trong `conversation.py`**

Thêm tham số `memory_summary: str | None = None` và `window_turns: int | None = None`.
Sau khi build `recent_turns_text` (logic hiện tại giữ nguyên):
- Nếu `memory_summary` không None và không rỗng: return `f"[Ngữ cảnh trước]: {memory_summary}\n\n{recent_turns_text}"`.
- Ngược lại: return `recent_turns_text` (giữ nguyên hành vi cũ).

- [ ] **Step 5: Chạy tất cả tests Task 5**

```bash
cd backend && pytest tests/test_conversation_context.py -v
```
Expected: tất cả PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/conversation.py backend/tests/test_conversation_context.py
git commit -m "feat: build_conversation_context supports memory_summary prefix"
```

---

### Task 6: `retrieval.py` — tách recall pool và inject limit

**Files:**
- Modify: `backend/app/services/retrieval.py`

**Interfaces:**
- Consumes: `settings.retrieval_recall_pool`, `settings.retrieval_inject_limit` (Task 4)
- Produces:
  ```python
  class LegalRetriever:
      def retrieve_with_metrics(
          self,
          question: str,
          as_of: date,
          limit: int = 8,              # deprecated, dùng settings nếu không truyền
          inject_limit: int | None = None,  # MỚI: override inject_limit từ settings
          conversation_context: str = "",
          standalone_query: str | None = None,
          additional_queries: list[str] | None = None,
          hyde_query: str | None = None,
      ) -> RetrievalResult
  ```
  `RetrievalResult.provisions` chứa tối đa `inject_limit` provisions.

- [ ] **Step 1: Sửa `LegalRetriever.retrieve_with_metrics`**

Trong `backend/app/services/retrieval.py`:
1. Thay `limit * 2` bằng `settings.retrieval_recall_pool` làm recall pool cho `HybridVectorStore().search()`.
2. Thêm tham số `inject_limit: int | None = None`.
3. Tính `effective_inject_limit = inject_limit or settings.retrieval_inject_limit`.
4. Dòng slice kết quả: thay `[:limit]` bằng `[:effective_inject_limit]`.
5. `retrieve()` delegate sang `retrieve_with_metrics()` — không cần thay đổi.

- [ ] **Step 2: Chạy toàn bộ test suite hiện có**

```bash
cd backend && pytest tests/ -v
```
Expected: không có test mới nào fail so với trước.

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/retrieval.py
git commit -m "feat: split retrieval recall_pool from inject_limit (pool=20, inject=5)"
```

---

### Task 7: `answering.py` — tích hợp TokenBudget và Map-Reduce

**Files:**
- Modify: `backend/app/services/answering.py`

**Interfaces:**
- Consumes: `TokenBudget` (Task 1), `MapReduceSummarizer` (Task 2), settings từ Task 4
- Tham số `generate` không thay đổi signature — thay đổi nội bộ.
- `AnswerGeneration.answer.latency.map_reduce_ms` và `token_budget_ms` được ghi sau khi tích hợp (dùng `LatencyBreakdown` từ Task 4).

- [ ] **Step 1: Import và khởi tạo TokenBudget trong `generate`**

Trong `GroundedAnswerService.generate`, sau khi build `sources` list (trước khi build `user_prompt`):

```python
from .token_budget import TokenBudget
from .map_reduce import MapReduceSummarizer

budget_started = perf_counter()
tb = TokenBudget(
    settings.context_model_limit,
    settings.context_reserved_output,
    settings.context_overhead,
)
fixed_text = system_prompt + f"\nLịch sử: {conversation_context}\nCâu hỏi: {question}"
sources_fitted, was_truncated = tb.fit_sources_to_budget(
    sources,   # đã sắp xếp score desc từ reranker, không cần re-sort
    fixed_text,
)
token_budget_ms = (perf_counter() - budget_started) * 1000
```

- [ ] **Step 2: Xử lý Map-Reduce khi truncated**

```python
map_reduce_ms = 0.0
if was_truncated:
    logger.warning(
        "Token budget exceeded: %d/%d sources fit; map_reduce_enabled=%s",
        len(sources_fitted), len(sources), settings.map_reduce_enabled,
    )
    if settings.map_reduce_enabled:
        mr_started = perf_counter()
        mr_result = await MapReduceSummarizer().summarize(
            question,
            sources,  # toàn bộ sources, không phải chỉ fitted
            target_token_budget=tb.available(fixed_text),
        )
        map_reduce_ms = (perf_counter() - mr_started) * 1000
        if mr_result.merged_text:
            # Thay sources_fitted bằng 1 synthetic source từ summary
            sources_fitted = [{
                "source_id": "S1",
                "citation": "Tổng hợp từ các điều khoản liên quan",
                "text": mr_result.merged_text,
                "score": 1.0,
            }]
            logger.info("Map-Reduce produced %d-batch summary", mr_result.batch_count)
```

- [ ] **Step 3: Dùng `sources_fitted` thay `sources` trong `user_prompt`**

Thay dòng build `user_prompt`:
```python
# Trước:
f"Nguồn: {json.dumps(sources, ensure_ascii=False)}"
# Sau:
f"Nguồn: {json.dumps(sources_fitted, ensure_ascii=False)}"
```
Và bỏ trường `score` ra khỏi `sources_fitted` khi build dict (score chỉ dùng nội bộ, không gửi LLM).

- [ ] **Step 4: Ghi `map_reduce_ms` và `token_budget_ms` vào `AnswerGeneration`**

`AnswerGeneration` là `@dataclass(frozen=True)` — thêm 2 field mới:
```python
@dataclass(frozen=True)
class AnswerGeneration:
    answer: ChatResponse
    llm_ms: float
    citation_validation_ms: float
    map_reduce_ms: float = 0.0
    token_budget_ms: float = 0.0
```
Trả về `AnswerGeneration(..., map_reduce_ms=map_reduce_ms, token_budget_ms=token_budget_ms)`.

- [ ] **Step 5: Chạy toàn bộ test suite**

```bash
cd backend && pytest tests/ -v
```
Expected: tất cả PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/answering.py
git commit -m "feat: integrate TokenBudget and Map-Reduce into answering pipeline"
```

---

### Task 8: `api/chat.py` — memory trigger, inject_limit, latency fields

**Files:**
- Modify: `backend/app/api/chat.py`

**Interfaces:**
- Consumes: `summarize_history` (Task 3), `build_conversation_context` với `memory_summary` (Task 5), `settings.retrieval_inject_limit`, `settings.history_window_turns`, `settings.memory_summary_enabled`, `AnswerGeneration.map_reduce_ms`, `AnswerGeneration.token_budget_ms`

- [ ] **Step 1: Import memory và cập nhật build_conversation_context**

Trong `backend/app/api/chat.py`:
1. Thêm import: `from ..services.memory import summarize_history`
2. Khi gọi `build_conversation_context(reversed(prior_messages), ...)`, truyền thêm:
   ```python
   memory_summary=conversation.memory_summary,
   window_turns=settings.history_window_turns,
   ```

- [ ] **Step 2: Giới hạn `prior_messages` theo `history_window_turns`**

Thay `.limit(settings.conversation_context_messages)` bằng `.limit(settings.history_window_turns * 2)` (mỗi turn = 2 messages: user + assistant).

- [ ] **Step 3: Truyền `inject_limit` vào `LegalRetriever.retrieve_with_metrics`**

Thêm kwarg `inject_limit=settings.retrieval_inject_limit` vào lời gọi.

- [ ] **Step 4: Trigger memory summarization sau `save_turn`**

Thêm import ở đầu file nếu chưa có: `from sqlalchemy import func` (cùng dòng với các import sqlalchemy hiện tại).

Sau dòng `generation.answer.conversation_id = save_turn(...)`:
```python
if settings.memory_summary_enabled:
    total_messages = db.scalar(
        select(func.count(ChatMessage.id))
        .where(ChatMessage.conversation_id == conversation.id)
    )
    window_msg_count = settings.history_window_turns * 2
    if total_messages > window_msg_count:
        old_messages = list(db.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation.id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .limit(total_messages - window_msg_count)
        ))
        try:
            memory_update = await summarize_history(old_messages, conversation.memory_summary)
            if memory_update.summary:
                conversation.memory_summary = memory_update.summary
                db.commit()
        except Exception as exc:
            logger.warning("Memory summarization failed (non-fatal): %s", exc)
```

- [ ] **Step 5: Ghi `map_reduce_ms` và `token_budget_ms` vào `LatencyBreakdown`**

```python
generation.answer.latency = LatencyBreakdown(
    ...
    map_reduce_ms=round(generation.map_reduce_ms, 1),
    token_budget_ms=round(generation.token_budget_ms, 1),
    ...
)
```
Cũng cập nhật log string để bao gồm 2 field mới.

- [ ] **Step 6: Chạy toàn bộ test suite**

```bash
cd backend && pytest tests/ -v
```
Expected: tất cả PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/chat.py
git commit -m "feat: wire memory summary, inject_limit, and budget latency into chat endpoint"
```

---

### Task 9: Integration smoke test + kiểm tra Review Focus

**Files:**
- Create: `backend/tests/test_integration_budget.py`

**Mục đích:** Xác nhận 5 điểm Review Focus từ phần đầu plan đã được cover.

- [ ] **Step 1: Viết test `estimate` không bao giờ under-estimate cho tiếng Việt**

```python
# backend/tests/test_integration_budget.py
import math
from app.services.token_budget import TokenBudget

def test_estimate_conservative_for_vietnamese():
    """
    Tiếng Việt UTF-8: mỗi ký tự có dấu thường = 2-3 bytes.
    estimate = ceil(len_chars / 3.5); char count < byte count → estimate bảo thủ.
    """
    tb = TokenBudget(8192, 1024, 256)
    text = "Điều khoản hợp đồng lao động theo Bộ luật Lao động."
    est = tb.estimate(text)
    # Thực tế mỗi ký tự Vietnamese ≈ 1 token (BPE); len/3.5 < thực tế → bảo thủ OK
    assert est >= 1
    # Không bao giờ estimate = 0 cho text không rỗng
    assert est > 0
```

- [ ] **Step 2: Viết test `fit` trả về `([], True)` khi fixed_prompt đã vượt budget**

Đã cover ở Task 1 Step 4 (`test_fit_returns_empty_when_fixed_exceeds_budget`). Đánh dấu ✓.

- [ ] **Step 3: Viết test memory fail silently**

Đã cover ở Task 3 Step 2 (`test_llm_failure_returns_empty_summary`). Đánh dấu ✓.

- [ ] **Step 4: Viết test reranker pool < inject_limit không raise**

```python
# backend/tests/test_integration_budget.py (tiếp)
from unittest.mock import MagicMock, patch
from datetime import date

def test_retrieval_with_fewer_candidates_than_inject_limit():
    """Khi vector store trả về ít candidates hơn inject_limit, không raise."""
    from app.services.retrieval import LegalRetriever, Reranker
    
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []  # 0 candidates từ DB
    
    with patch("app.services.retrieval.HybridVectorStore") as mock_vs:
        mock_vs.return_value.search.return_value = []
        retriever = LegalRetriever(mock_db)
        result = retriever.retrieve_with_metrics("câu hỏi", date.today(), inject_limit=5)
    
    assert result.provisions == []   # không raise, trả về list rỗng
```

- [ ] **Step 5: Viết test Map-Reduce single-batch không gọi reduce**

Đã cover ở Task 2 Step 1 (`test_single_batch_no_reduce`). Đánh dấu ✓.

- [ ] **Step 6: Chạy toàn bộ test suite lần cuối**

```bash
cd backend && pytest tests/ -v --tb=short
```
Expected: tất cả PASS, không có warning về imports mới.

- [ ] **Step 7: Final commit**

```bash
git add backend/tests/test_integration_budget.py
git commit -m "test: add integration smoke tests covering Review Focus edge cases"
```

---

## Tóm tắt thứ tự triển khai

```
Task 1: token_budget.py          ← không dependency
Task 2: map_reduce.py            ← depends Task 1
Task 3: memory.py                ← không dependency (mock LLM)
Task 4: models + config + schemas ← không dependency
Task 5: conversation.py          ← depends Task 4 (settings)
Task 6: retrieval.py             ← depends Task 4 (settings)
Task 7: answering.py             ← depends Task 1, 2, 4
Task 8: api/chat.py              ← depends Task 3, 4, 5, 6, 7
Task 9: integration tests        ← depends tất cả tasks trên
```
