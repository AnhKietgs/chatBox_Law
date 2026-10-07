# Law RAG — Chatbot pháp luật Việt Nam có căn cứ

Chatbot RAG tiếng Việt cho Luật Thương mại 2005 và Bộ luật Dân sự 2015. Hệ thống chỉ trả lời khi có căn cứ truy xuất được đến **Điều / Khoản / Điểm**; nếu không đủ nguồn đã duyệt, chatbot sẽ từ chối an toàn thay vì suy đoán.

> Đây là công cụ tra cứu tham khảo, không thay thế tư vấn luật sư.

## Chạy ngay bằng một lệnh

### Windows PowerShell

Sao chép lệnh dưới đây vào PowerShell. Lệnh sẽ clone project, tạo `.env`, dựng các container và tải model Qwen dùng cho chatbot.

```powershell
git clone https://github.com/AnhKietgs/chatBox_Law.git; Set-Location chatBox_Law; Copy-Item .env.example .env; docker compose up --build -d; docker compose exec ollama ollama pull Qwen3.5:2b
```

Sau khi hoàn tất, mở:

- Chat: [http://localhost:8080](http://localhost:8080)
- Admin: [http://localhost:8080/#admin](http://localhost:8080/#admin)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### macOS / Linux

```bash
git clone https://github.com/AnhKietgs/chatBox_Law.git && cd chatBox_Law && cp .env.example .env && docker compose up --build -d && docker compose exec ollama ollama pull Qwen3.5:2b
```

## Điều kiện cần

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) đang chạy.
- Docker có tối thiểu **8 GB RAM**. Lần chạy đầu có thể mất thời gian vì cần tải Python/ML dependencies, embedding model và Qwen.
- Cổng `8080`, `8000` chưa bị ứng dụng khác dùng.

> Lệnh nhanh dùng cấu hình mẫu trong `.env.example`. Trước khi demo công khai hoặc deploy, hãy thay `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `JWT_SECRET`, mật khẩu PostgreSQL và MinIO trong `.env`.

## Dùng lần đầu

Chatbot sẽ **chưa trả lời có căn cứ** cho đến khi có ít nhất một văn bản được xuất bản.

1. Mở [trang quản trị](http://localhost:8080/#admin) và đăng nhập bằng `ADMIN_EMAIL` / `ADMIN_PASSWORD` trong `.env`.
2. Tải PDF/DOCX bản chính thức của **Luật Thương mại 2005** hoặc **Bộ luật Dân sự 2015**; điền URL nguồn chính thức và ngày hiệu lực.
3. Chờ job xử lý chuyển sang `PENDING_REVIEW`.
4. Chọn **Kiểm tra cấu trúc**, kiểm tra Điều/Khoản/Điểm, tick xác nhận rồi bấm **Duyệt & xuất bản**.
5. Quay lại [chat](http://localhost:8080) để hỏi pháp luật.

Câu hỏi thử:

> Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?

Kết quả đúng phải dẫn **Điều 301 Luật Thương mại 2005** và hiển thị liên kết văn bản gốc.

## Tính năng chính

- Trả lời có citation Điều/Khoản/Điểm, đoạn trích và URL nguồn chính thức.
- Safe abstention: không có nguồn phù hợp, citation không khớp hoặc LLM trả sai schema thì từ chối trả lời.
- Legal-structure-aware chunking: giữ cấu trúc `Điều → Khoản → Điểm`; chỉ tách phụ khi đơn vị quá dài.
- Hybrid retrieval: BGE-M3 dense + sparse, RRF, bge-reranker-v2-m3.
- Conversational RAG: lưu hội thoại, contextualize câu hỏi theo ngữ cảnh và phát hiện đổi chủ đề.
- HyDE + MultiQuery có kiểm soát cho câu hỏi ngắn/mơ hồ; không tạo căn cứ pháp lý mới.
- Versioning theo thời điểm hiệu lực: lưu dữ liệu lịch sử để tra cứu tình huống quá khứ.
- Admin ingest PDF/DOCX, Docling và OCR fallback, review trước khi publish.
- Chống prompt injection trong nội dung tài liệu, JWT admin và rate limit Redis cho API công khai.
- Evaluation: golden set, kiểm tra citation và RAGAS quality gate.

## Kiến trúc

```text
React + Nginx
      │
      ▼
FastAPI ───── PostgreSQL (metadata, version, Điều/Khoản/Điểm)
  │   │
  │   ├──── Qdrant (BGE-M3 dense + sparse vectors)
  │   ├──── Ollama (Qwen3.5)
  │   └──── Redis + Celery (ingest bất đồng bộ)
  │
  └──────── MinIO (PDF/DOCX gốc và artefacts parser/OCR)
```

| Thành phần | Công nghệ |
| --- | --- |
| Frontend | React, Vite, Nginx |
| Backend | FastAPI |
| Parser / OCR | Docling → RapidOCR → PaddleOCR fallback |
| Relational DB / file store | PostgreSQL / MinIO |
| Vector DB / embedding | Qdrant / BGE-M3 |
| Retrieval | Hybrid dense + sparse, RRF, bge-reranker-v2-m3 |
| LLM | Ollama + Qwen3.5:2b (có thể benchmark 4b/9b) |
| Async | Celery + Redis |

## Các lệnh hữu ích

```powershell
# Xem trạng thái container
docker compose ps

# Xem log API và worker
docker compose logs -f api worker

# Dừng hệ thống, nhưng giữ dữ liệu đã ingest
docker compose down

# Chạy lại sau này (không cần --build nếu không sửa code)
docker compose up -d

# Kiểm tra model đã tải
docker compose exec ollama ollama list

# Chạy backend tests
docker compose exec api pytest -q
```

## API chính

```http
POST /api/v1/chat/query
Content-Type: application/json

{
  "question": "Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?",
  "conversation_id": "optional-uuid"
}
```

Response có `grounded` hoặc `abstained`, câu trả lời, claims, citations và thông tin latency. Mọi citation được backend kiểm tra phải thuộc đúng tập provision vừa retrieval.

## Đánh giá chất lượng

`evaluation/golden_questions.example.jsonl` chỉ là template. Trước khi phát hành, chuyên gia pháp lý cần tạo `evaluation/golden_questions.jsonl` gồm 100 câu có đáp án và căn cứ vàng.

```powershell
python evaluation/validate_golden.py evaluation/golden_questions.jsonl --required-count 100
python evaluation/run_benchmark.py evaluation/golden_questions.jsonl
```

Quality gate mục tiêu: citation đúng 100%, grounded ≥ 90%, safe abstention ≥ 95%. RAGAS có script riêng tại `evaluation/run_ragas.py`.

## Lưu ý vận hành

- Sửa model bằng `OLLAMA_MODEL` trong `.env`, ví dụ `Qwen3.5:4b` hoặc `Qwen3.5:9b`, rồi pull đúng model qua Ollama.
- `LLM_TEMPERATURE=0.1` để ưu tiên tính ổn định cho nghiệp vụ pháp lý.
- Chỉ xuất bản văn bản có nguồn chính thức và đã được người có thẩm quyền kiểm tra cấu trúc.
- Không commit `.env`, mật khẩu, JWT secret hoặc tài liệu pháp lý không được phép công bố.
