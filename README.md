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

1. Admin tải PDF/DOCX bản chuẩn, điền mã/tên văn bản, URL HTTPS chính thức, nhãn phiên bản và ngày hiệu lực.
2. Worker lưu tệp vào MinIO, dùng Docling và OCR fallback, sau đó tách các đơn vị `Điều → Khoản → Điểm` vào PostgreSQL. Job chỉ chuyển sang `PENDING_REVIEW`, chưa hề xuất hiện trong chatbot.
3. Admin chọn **Kiểm tra cấu trúc** để xem toàn bộ Điều/Khoản/Điểm và đoạn trích đã parse. Sau khi xác nhận, admin tick ô kiểm tra rồi mới chọn **Duyệt & xuất bản**. Backend xác minh hash của đúng cấu trúc đã xem trước khi index vào Qdrant.
4. Khi xuất bản phiên bản mới có hiệu lực về sau, phiên bản mở trước đó được tự đóng hiệu lực vào ngày liền trước; lịch sử vẫn được giữ để trả lời câu hỏi quá khứ.
5. Nếu phát hiện dữ liệu không an toàn, admin chọn **Thu hồi**. Phiên bản bị loại ngay khỏi retrieval nhưng vẫn giữ bản gốc/audit trail; có thể tải lại cùng nhãn phiên bản để parse lại.

Nguồn khởi tạo được chốt là Luật Thương mại 2005 và Bộ luật Dân sự 2015. Chỉ tải bản có nguồn URL chính thức đã được người có thẩm quyền xác thực.

## Hợp đồng API quan trọng

- `POST /api/v1/chat/query`: `{ question, as_of_date?, conversation_id? }` → `grounded` hoặc `abstained`, claims và citation chứa `document_code`, `article`, `clause`, `point`, `excerpt`, `official_url`.
- `POST /api/v1/admin/token`: nhận email/mật khẩu admin, trả JWT 8 giờ.
- `POST /api/v1/admin/versions`: multipart file/metadata, trả ingestion job 202.
- `GET /api/v1/admin/jobs/{id}`, `GET /api/v1/admin/versions`, `GET /api/v1/admin/versions/{id}/review`, `POST /api/v1/admin/versions/{id}/publish`, `POST /api/v1/admin/versions/{id}/withdraw`: chỉ dành cho admin.

Nếu retrieval rỗng, model lỗi, JSON sai schema hoặc bất kỳ claim nào không trỏ tới một citation đã truy xuất, API trả `abstained`; nó không tạo câu trả lời không nguồn.

## Kiểm thử và quality gate

Chạy test backend trong container:

```powershell
docker compose exec api pytest -q
```

`evaluation/golden_questions.example.jsonl` mô tả định dạng bộ benchmark. Một quản trị viên/chuyên gia pháp lý phải tạo chính xác 100 câu đã gán citation vàng vào `evaluation/golden_questions.jsonl`, sau đó chạy:

```powershell
python evaluation/run_benchmark.py evaluation/golden_questions.jsonl
```

Lệnh chỉ pass khi citation chính xác 100%, tỷ lệ trả lời grounded đúng tối thiểu 90% và tỷ lệ từ chối an toàn tối thiểu 95%. Không nên công bố chatbot trước khi bộ 100 câu được chuyên gia thẩm định.

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

- Cấu hình `OLLAMA_MODEL` cho phép benchmark `Qwen3.5:4b` hoặc `Qwen3.5:9b` mà không đổi mã nguồn.
- `CONFIDENCE_THRESHOLD` lọc nguồn sau rerank; `CLAIM_SUPPORT_THRESHOLD` bắt buộc mỗi claim do LLM sinh phải khớp với một đoạn trích đã dẫn. Chỉ thay đổi các ngưỡng này sau khi đánh giá bằng bộ câu vàng.
- Rate limit chat hiện là in-memory cho một instance. Khi mở rộng API nhiều instance, thay bằng limiter Redis trước khi triển khai công khai.
- LangGraph chưa nằm trên critical path; lớp `GroundedAnswerService` là điểm thay thế workflow khi cần nhiều nhánh/đánh giá hơn.
- Đây là công cụ tra cứu có căn cứ, không thay thế tư vấn luật sư cho tình huống cụ thể.
