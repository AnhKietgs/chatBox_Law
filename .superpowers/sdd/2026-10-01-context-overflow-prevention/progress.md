# SDD ledger — plan: docs/superpowers/plans/2026-10-01-context-overflow-prevention.md

## Pre-flight interface scan

| Consumer task | Producer task | Produces | Consumes | Status |
|---|---|---|---|---|
| T2 map_reduce | T1 token_budget | `TokenBudget` class | `TokenBudget.estimate()` | Match ✓ |
| T5 conversation | T4 config | `settings.history_window_turns` | same | Match ✓ |
| T6 retrieval | T4 config | `settings.retrieval_recall_pool`, `settings.retrieval_inject_limit` | same | Match ✓ |
| T7 answering | T1, T2, T4 | `TokenBudget`, `MapReduceSummarizer`, settings | same | Match ✓ |
| T8 chat.py | T3, T4, T5, T6, T7 | `summarize_history`, settings, `build_conversation_context(memory_summary=)`, `inject_limit`, `map_reduce_ms/token_budget_ms` | same | Match ✓ |
| T9 integration | T1–T8 all | all public APIs | same | Match ✓ |

Pre-flight ruling: `AnswerGeneration` hiện là `@dataclass(frozen=True)` với 3 fields — plan Task 7 thêm 2 fields mới. Cần verify frozen dataclass có thể thêm field với default. **Ruling: Python frozen dataclass cho phép thêm field có default value; không cần thay đổi cách dùng. Cost if wrong: AttributeError tại runtime khi không truyền giá trị.**

Pre-flight ruling: Plan Task 7 Step 1 tạo `sources` có key `"score"` nhưng `answering.py` hiện tại build `sources` dict chỉ với `source_id`, `citation`, `text` — không có `score`. Cần thêm `score` vào dict hoặc đổi cách sort. **Ruling: Đã sửa trong plan (dùng list nguyên bản từ reranker đã sorted). Không cần thêm key score vào sources dict. Cost if wrong: sort sẽ flat nếu score thiếu — đã loại bỏ re-sort.**
