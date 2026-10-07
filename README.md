# Law RAG — Chatbot pháp luật thương mại

MVP hỏi đáp tiếng Việt về hợp đồng thương mại, mua bán hàng hóa và phạt vi phạm. Hệ thống chỉ trả lời khi từng nhận định gắn với Điều/Khoản/Điểm trong một phiên bản văn bản đã được quản trị viên duyệt.

## Kiến trúc

| Thành phần | Vai trò |
| --- | --- |
| React + Nginx | Chat công khai, chọn thời điểm áp dụng, hiển thị citation; khu vực quản trị ingest/duyệt. |
| FastAPI | API chat, JWT admin, kiểm tra citation, metadata/hiệu lực văn bản. |
| PostgreSQL + MinIO | Cấu trúc pháp lý/versioning và tệp gốc/kết quả parser. |
| Celery + Redis | Xử lý tệp bất đồng bộ. |
| Docling → RapidOCR → PaddleOCR | Trích xuất văn bản; chỉ fallback OCR khi Docling không lấy được text. |
| Qdrant + BGE-M3 | Hybrid dense+sparse recall trên đơn vị Điều/Khoản/Điểm. |
| bge-reranker-v2-m3 + Ollama/Qwen | Rerank rồi sinh JSON có citation ID; backend từ chối citation không thuộc tập retrieved. |

## Chạy bằng Docker Compose

1. Sao chép cấu hình: `Copy-Item .env.example .env`, sau đó thay toàn bộ mật khẩu và `JWT_SECRET` trong `.env`.
2. Khởi động: `docker compose up --build -d`.
3. Chờ Ollama khởi động, sau đó kéo model: `docker compose exec ollama ollama pull Qwen3.5:2b`.
4. Mở [http://localhost:8080](http://localhost:8080). Khu vực admin ở [http://localhost:8080/#admin](http://localhost:8080/#admin); thông tin đăng nhập lấy từ `ADMIN_EMAIL`/`ADMIN_PASSWORD`.

API healthcheck: [http://localhost:8000/healthz](http://localhost:8000/healthz). API docs: [http://localhost:8000/docs](http://localhost:8000/docs).

## Luồng quản trị kho luật

1. Admin tải PDF/DOCX bản chuẩn, điền mã/tên văn bản, URL HTTPS chính thức, nhãn phiên bản, ngày hiệu lực, **lĩnh vực** (`commercial`, `civil`, `labor`, `enterprise`, `tax`…) và nhóm nghiệp vụ. Metadata này được lưu ở PostgreSQL và mirror vào Qdrant.
2. Worker lưu tệp vào MinIO, dùng Docling và OCR fallback, sau đó tách các đơn vị `Điều → Khoản → Điểm` vào PostgreSQL. Với PDF lai, worker kiểm tra từng trang: trang có ít hơn 100 ký tự embedded text sẽ được render và OCR riêng bằng RapidOCR/PaddleOCR. Job chỉ chuyển sang `PENDING_REVIEW`, chưa hề xuất hiện trong chatbot.

Trước khi ingest, hệ thống quét dấu hiệu prompt injection ở nội dung đọc trực tiếp; worker **quét lại sau Docling/OCR** để phát hiện văn bản ẩn trong PDF/DOCX nén. Nếu gặp chỉ dẫn như “bỏ qua mọi chỉ dẫn”, system prompt, đổi vai trò hoặc trích xuất bí mật, version bị `FAILED`/cách ly và không có chunk hay vector nào được ghi vào Qdrant. Đây là signal-based scanner, không thay thế bước admin kiểm tra nguồn chính thức.

Khi ingest, worker luôn log `len(text)` sau khi load. Với PDF lai, log cũng nêu số ký tự từng trang và các trang đã OCR, ví dụ: `Document load complete: filename=... len(text)=12345 source=hybrid_pdf`.

### Chính sách chunking pháp lý

- Đơn vị chính là `Điều → Khoản → Điểm`; mỗi chunk luôn giữ locator pháp lý đầy đủ.
- Khi một đơn vị vượt `CHUNK_MAX_TOKENS` (mặc định `450` theo ước tính `ceil(len/3.5)`), hệ thống mới tách thêm ở ranh giới đoạn/câu. Các phần vẫn cùng Điều/Khoản/Điểm và được đánh số `Phần 1`, `Phần 2`… để audit.
- Không dùng overlap giữa hai đơn vị pháp lý khác nhau. Chỉ các phần của **cùng một** đơn vị quá dài mới nhận tối đa `CHUNK_SPLIT_OVERLAP_TOKENS` (mặc định `40`) từ phần ngay trước; điều này tránh làm nhiễu citation bởi nội dung luật bị lặp.
- Sau mỗi ingest, worker log số chunk, token estimate `min/p50/p95/max`, số chunk đã split, và 5 mẫu deterministic (locator, phần, token estimate, excerpt) để admin/dev kiểm tra.
3. Admin chọn **Kiểm tra cấu trúc** để xem toàn bộ Điều/Khoản/Điểm và đoạn trích đã parse. Sau khi xác nhận, admin tick ô kiểm tra rồi mới chọn **Duyệt & xuất bản**. Backend xác minh hash của đúng cấu trúc đã xem trước khi index vào Qdrant.
4. Khi xuất bản phiên bản mới có hiệu lực về sau, phiên bản mở trước đó được tự đóng hiệu lực vào ngày liền trước; lịch sử vẫn được giữ để trả lời câu hỏi quá khứ.
5. Nếu phát hiện dữ liệu không an toàn, admin chọn **Thu hồi**. Phiên bản bị loại ngay khỏi retrieval nhưng vẫn giữ bản gốc/audit trail; có thể tải lại cùng nhãn phiên bản để parse lại.

Với văn bản đã `PUBLISHED` trước khi nâng cấp metadata, admin chọn **Lập chỉ mục lại metadata** một lần. Thao tác giữ nguyên cấu trúc/embedding nhưng upsert lại payload Qdrant gồm `document_code`, `domain`, `category`, trạng thái và hiệu lực.

Nguồn khởi tạo được chốt là Luật Thương mại 2005 và Bộ luật Dân sự 2015. Chỉ tải bản có nguồn URL chính thức đã được người có thẩm quyền xác thực.

## Hợp đồng API quan trọng

- `POST /api/v1/chat/query`: `{ question, as_of_date?, conversation_id? }` → `grounded` hoặc `abstained`, claims và citation chứa `document_code`, `article`, `clause`, `point`, `excerpt`, `official_url`.
- `POST /api/v1/admin/token`: nhận email/mật khẩu admin, trả JWT 8 giờ.
- `POST /api/v1/admin/evaluation/query`: **chỉ admin/evaluator**; chạy cùng pipeline retrieval/generation và trả thêm toàn bộ provision đã cấp cho model. Endpoint này chỉ dùng cho RAGAS, không lộ cho chat công khai.
- `POST /api/v1/admin/versions`: multipart file/metadata, trả ingestion job 202.
- `GET /api/v1/admin/jobs/{id}`, `GET /api/v1/admin/versions`, `GET /api/v1/admin/versions/{id}/review`, `POST /api/v1/admin/versions/{id}/publish`, `POST /api/v1/admin/versions/{id}/withdraw`: chỉ dành cho admin.

Nếu retrieval rỗng, model lỗi, JSON sai schema hoặc bất kỳ claim nào không trỏ tới một citation đã truy xuất, API trả `abstained`; nó không tạo câu trả lời không nguồn.

## Kiểm thử và quality gate

Chạy test backend trong container:

```powershell
docker compose exec api pytest -q
```

`evaluation/golden_questions.example.jsonl` là **template, không phải ground truth**. Chuyên gia pháp lý phải tạo `evaluation/golden_questions.jsonl` gồm đúng 100 câu, mỗi dòng có đáp án tham chiếu (`reference_answer`), Điều/Khoản/Điểm vàng, người duyệt và ngày duyệt. Validator từ chối placeholder hoặc dữ liệu chưa duyệt:

```powershell
python evaluation/validate_golden.py evaluation/golden_questions.jsonl --required-count 100
```

Sau đó chạy kiểm tra xác định (status/citation) và RAGAS. RAGAS là dependency của evaluator riêng, không làm nặng container API/worker:

```powershell
python evaluation/run_benchmark.py evaluation/golden_questions.jsonl
pip install -r evaluation/requirements-ragas.txt
python evaluation/run_ragas.py evaluation/golden_questions.jsonl `
  --api-base-url http://localhost:8000 `
  --admin-email $env:ADMIN_EMAIL --admin-password $env:ADMIN_PASSWORD `
  --judge-base-url http://localhost:11434/v1 --judge-model Qwen3.5:4b
```

`run_ragas.py` tạo report JSON và chỉ pass khi: **Faithfulness ≥ 0.90**, **Answer Relevancy ≥ 0.80**, **Context Precision ≥ 0.80**, **Context Recall ≥ 0.90**; đồng thời citation chính xác 100%, grounded ≥ 90% và từ chối an toàn ≥ 95%. Không nên công bố chatbot trước khi bộ 100 câu được chuyên gia thẩm định.

GitHub Actions có hai workflow: `Validate RAG contracts` kiểm tra contract trên mọi PR; `RAGAS quality gate` chạy khi merge `main`, mỗi tuần hoặc chạy thủ công. Để gate hoạt động, cấu hình GitHub secrets `EVAL_API_BASE_URL`, `EVAL_ADMIN_EMAIL`, `EVAL_ADMIN_PASSWORD`, `RAGAS_JUDGE_BASE_URL`, `RAGAS_JUDGE_API_KEY` và variable `RAGAS_JUDGE_MODEL`; đồng thời đặt workflow này là required check của môi trường production. Không đưa JWT, mật khẩu hay URL nội bộ vào repository.

## Load test số người dùng đồng thời

Sau khi có ít nhất một văn bản `PUBLISHED` và không có tác vụ xuất bản đang chạy, chạy lần lượt 1, 2, 3 và 5 virtual users (mỗi user gửi đúng một câu hỏi đồng thời):

```powershell
$env:VUS="1"; Get-Content -Raw evaluation/load_test.js | docker run --rm -i -e VUS=$env:VUS grafana/k6 run -
$env:VUS="2"; Get-Content -Raw evaluation/load_test.js | docker run --rm -i -e VUS=$env:VUS grafana/k6 run -
$env:VUS="3"; Get-Content -Raw evaluation/load_test.js | docker run --rm -i -e VUS=$env:VUS grafana/k6 run -
$env:VUS="5"; Get-Content -Raw evaluation/load_test.js | docker run --rm -i -e VUS=$env:VUS grafana/k6 run -
```

Lấy mức VU cao nhất có `http_req_failed` bằng 0, `http_req_duration` p95 dưới 60 giây, không vượt 85% RAM và không bị treo làm kết quả. Đây là công suất kiểm chứng được trên chính máy chạy test, không phải số người dùng Internet tổng quát.

## Lưu ý vận hành

- Cấu hình `OLLAMA_MODEL` cho phép benchmark `Qwen3.5:4b` hoặc `Qwen3.5:9b` mà không đổi mã nguồn. `LLM_TEMPERATURE` mặc định là `0.1`, dùng chung cho toàn bộ call Ollama để câu trả lời đủ ổn định nhưng không quá cứng nhắc.
- `CONFIDENCE_THRESHOLD` lọc nguồn sau rerank; `CLAIM_SUPPORT_THRESHOLD` bắt buộc mỗi claim do LLM sinh phải khớp với một đoạn trích đã dẫn. Chỉ thay đổi các ngưỡng này sau khi đánh giá bằng bộ câu vàng.
- Mọi `POST /chat/query` có rate limit theo IP (`PUBLIC_RATE_LIMIT_PER_MINUTE`, mặc định 20); đăng nhập admin có limit riêng (mặc định 8/phút). Limiter dùng Redis để chia sẻ giữa nhiều API instance và chỉ fallback local khi Redis lỗi (có warning log). Không tin `X-Forwarded-For` trực tiếp; reverse proxy phải chịu trách nhiệm strip/set header đó nếu cần định danh IP thật.
- Nguồn retriever được bao trong `<documents><document>…</document></documents>` với XML escaping; system prompt khẳng định câu hỏi, lịch sử và nội dung nguồn là dữ liệu không đáng tin cậy, không phải chỉ dẫn. Claim/citation validation vẫn là lớp chặn cuối.
- Retrieval chỉ thêm filter Qdrant `domain` khi câu hỏi có tín hiệu lĩnh vực rõ ràng; câu chung chung vẫn search toàn corpus. Câu hỏi đa miền bị chặn trước retrieval để yêu cầu người dùng tách/ưu tiên nội dung, không tự chọn một lĩnh vực.
- `INTENT_SHIFT_DETECTION_ENABLED=true` (mặc định) bảo vệ Conversational RAG khi người dùng đổi lĩnh vực đột ngột trong cùng đoạn chat. Nếu lượt gần nhất có lĩnh vực rõ ràng khác với câu hiện tại (ví dụ thương mại → lao động), hệ thống giữ lịch sử để xem lại nhưng không gửi lịch sử đó vào contextualizer, retrieval hoặc prompt sinh đáp án của lượt mới. Câu hỏi ngắn/generic như “nếu vi phạm thì sao?” vẫn giữ ngữ cảnh cũ.
- `MAP_REDUCE_ENABLED=true` là mặc định. Khi toàn bộ nguồn đã rerank vượt context budget, MapReduce cô đọng **toàn bộ** nguồn nhưng bắt buộc giữ nhãn `S1/S2/...`; lượt sinh đáp án cuối vẫn kiểm tra citation với điều/khoản gốc. Nếu MapReduce lỗi hoặc làm mất nhãn nguồn, hệ thống chỉ dùng tập nguồn gốc vừa budget (hoặc từ chối an toàn), không dùng bản tóm tắt không kiểm chứng. Chỉ đặt `false` ở môi trường thiếu tài nguyên.
- LangGraph chưa nằm trên critical path; lớp `GroundedAnswerService` là điểm thay thế workflow khi cần nhiều nhánh/đánh giá hơn.
- Đây là công cụ tra cứu có căn cứ, không thay thế tư vấn luật sư cho tình huống cụ thể.
