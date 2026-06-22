# Reflection — Lab 18: Production RAG Pipeline

**Họ tên:** Phan Võ Trọng Tiên
**Ngày:** 2026-06-22
**Cấu hình chạy thật:** Hierarchical chunking (M1) → Enrichment combined 1-call/chunk (M5) → Hybrid BM25+Dense+RRF (M2) → CrossEncoder rerank top-3 (M3) → LLM `gemini-2.5-flash-lite` → RAGAS (M4), 20 câu test.

---

## Phần 1 — Mapping bài giảng → Code

| Lecture Concept | Module | Hàm cụ thể | Observation (từ lần chạy thật) |
|----------------|--------|-------------|-------------------------------|
| Semantic chunking (nhóm câu theo độ tương đồng) | M1 | `chunk_semantic()` | Dùng `all-MiniLM-L6-v2` + cosine similarity, tách chunk khi `sim < threshold`. Threshold 0.85 tạo ít chunk hơn basic vì gộp câu cùng chủ đề. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` | Parent 2048 / child 256, mỗi child có `parent_id`. Lần chạy: 100 child chunks. **Bài học:** child 256 ký tự khá nhỏ → góp phần làm `context_recall` thấp hơn baseline. |
| Structure-aware (theo header markdown) | M1 | `chunk_structure_aware()` | Split bằng regex `^#{1,3}\s+`, giữ header trong `metadata["section"]` — không cắt giữa section. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | `score(d) = Σ 1/(k+rank+1)`, k=60. RRF giải quyết việc 2 hệ điểm khác thang (BM25 raw vs cosine) bằng cách chỉ dùng **thứ hạng**, không dùng giá trị điểm. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` (vd `nghỉ_phép`); phải `replace("_"," ")` nếu không BM25 split-by-space sẽ không khớp query 2 token. |
| Dense retrieval | M2 | `DenseSearch.search()` | bge-m3 (1024-dim) + Qdrant `query_points()` (API mới, không phải `search()`). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder` (không dùng FlagEmbedding vì crash với transformers>=5). Rerank top-20 → top-3. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness 0.70 / AnswerRelevancy 0.47 / ContextPrecision 0.775 / ContextRecall 0.783. **Metric thấp nhất = answer_relevancy** → vấn đề ở khâu *sinh câu trả lời*, không phải retrieval. |
| Failure analysis (Diagnostic Tree) | M4 | `failure_analysis()` | Map `worst_metric` → (diagnosis, fix). Phát hiện pattern: nhiều câu faithfulness=0 dù context precision/recall=1.0 → lỗi Generation. |
| Contextual embeddings (Anthropic style) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả ngữ cảnh trước khi embed. **Bài học ngược:** trong lần chạy này enrichment lại *làm giảm* điểm so với baseline (xem Phần 2 & failure_analysis.md). |

---

## Phần 2 — Khó khăn & cách giải quyết

**1. Output bị cụt khi dùng model "thinking" (`gemini-2.5-flash`).**
- Hàm `summarize_chunk()` chỉ trả về 1 từ ("Nhân viên") thay vì 2-3 câu.
- Debug: in raw response → phát hiện model tiêu reasoning tokens, ăn hết `max_tokens=150` nên gần như không còn token cho phần output hiển thị.
- Fix: đổi sang `gemini-2.5-flash-lite` (thinking nhẹ hơn) → summary đầy đủ. **Bài học:** chọn model phải xét cả đặc tính thinking/budget token, không chỉ tên model.

**2. `503 high demand` rải rác khi enrich 100 chunks + enrichment chạy ~16 phút.**
- Exact error: `Error code: 503 - This model is currently experiencing high demand`.
- Code đã bọc `try/except` fallback extractive nên 503 không làm vỡ pipeline, nhưng chạy lại từ đầu rất tốn thời gian.
- Fix: thêm **cache enrichment** ra `reports/enriched_chunks.json`; lần chạy sau tái dùng cache → chỉ chạy lại index + eval (vài phút) thay vì enrich lại 16 phút.

**3. Qdrant container chết giữa chừng → indexing fail.**
- Exact error: `WinError 10061 - No connection could be made because the target machine actively refused it` tại `DenseSearch.index()`.
- Debug: `docker ps -a` cho thấy container `Exited (255)` trong lúc enrichment chạy dài.
- Fix: `docker compose up -d` lại + thêm health-check (`curl /collections`) trước khi index; nhờ cache ở (2) nên không mất 16 phút enrichment.

**4. (Khó khăn về kết quả) Production THẤP HƠN Baseline ở cả 4 metric.**
- Đây là khó khăn "đau" nhất vì code đúng (42/42 test pass) mà điểm lại giảm: recall −0.10, faithfulness −0.05…
- Debug: đối chiếu config 2 pipeline → nghi 2 thủ phạm: (a) hierarchical **child chunk 256 ký tự** quá nhỏ làm mất ngữ cảnh (recall giảm mạnh nhất); (b) **enrichment** prepend câu context + thay metadata gốc làm "loãng" embedding, lại bị 503 nên fallback không đồng đều.
- Hướng xử lý: tăng `child_size`, và A/B test từng kỹ thuật enrichment thay vì bật tất cả.

**Kiến thức thiếu → cách bổ sung:** trước đây mình mặc định "pipeline phức tạp hơn = tốt hơn". Lab dạy mình phải **luôn đo baseline trước** và chỉ giữ lại bước nào *thực sự* tăng điểm trên RAGAS. Cách bổ sung: viết smoke-test 1 call trước khi chạy batch dài, và đọc kỹ thông báo lỗi của thư viện/model thay vì đoán.

---

## Phần 3 — Action Plan cho project cá nhân

### Project: EduDraft — App sinh giáo án CV 5512 từ SGK

**Hiện tại:** Ephemeral RAG — mỗi request upload SGK → OCR Azure (layout→Markdown) → chunk → embed Voyage-4 (1024) → Chroma, scope per-upload, không tái dùng. Retrieval theo 4 bucket mục 5512 (objectives/warmup/knowledge/practice).

**Known issues:** chunk lại mỗi request (tốn chi phí RAG, không tái dùng); bài 5–6 trang RAG không hơn direct-injection; Chroma trống lại sau redeploy (Render FS ephemeral); chưa verify end-to-end với embedding thật.

**Plan áp dụng (rút ra từ Lab 18):**
1. [ ] **Chunking:** gộp theo mục + size cap (max_chars/overlap) + gắn content_type rule-based, clean_ocr_text(). Thay vì cắt-mỗi-dòng-trống băm vụn (~140 chunk/bài → 16 chunk, avg ~500 ký tự). *(Lab: chunk quá nhỏ làm giảm recall.)*
2. [ ] **Search:** Dense hiện tại → cân nhắc **Hybrid (BM25 + Dense)** cho thuật ngữ hoá học/số bài cần khớp lexical. Chốt sau khi có eval.
3. [ ] **Reranking:** chưa có; thêm cross-encoder (bge-reranker) khi chuyển persistent, bật khi eval thấy nhiễu trong sources.
4. [ ] **Evaluation:** custom metrics trước (retrieval hit theo bucket, sai content_type) → RAGAS (faithfulness/answer relevancy) khi có bộ câu hỏi vàng. *(Lab: luôn đo baseline trước.)*
5. [ ] **Enrichment:** **persistent store + metadata filter** (bộ sách/khối/bài lân cận) — OCR+chunk+embed 1 lần/tài liệu, tái dùng → giải nỗi đau "kẹt giữa". Đổi embedding sang **bge-m3** (local, free, 1024); dedup theo hash file.

**Timeline:**
- **Tuần này:** deploy prod (Render BE + Vercel FE + Supabase DB), verify đăng nhập + sinh giáo án.
- **Tuần kế:** đổi `.env` sang bge-m3; pipeline + endpoint ingest; persistent store (bỏ scope_id, cân nhắc Supabase pgvector / Chroma Cloud).
- **Tuần sau:** retrieval lọc metadata (subject/grade/lesson_no) + dedup hash; eval custom + RAGAS; cân nhắc Hybrid + reranker dựa trên kết quả.

> Chi tiết RAG persistent: `PERSISTENT_RAG_PLAN.md`. Web-search học liệu là tool riêng (Tavily/SerpAPI), không dùng RAG.

---

## Phần 4 — Tự đánh giá

| Tiêu chí | Tự chấm (1–5) | Lý do |
|----------|:-------------:|-------|
| Hiểu bài giảng | 5 | Map được cả 5 module vào hàm cụ thể + giải thích được tại sao RRF, hierarchical, contextual prepend hoạt động. |
| Code quality | 5 | 42/42 test pass, 0 TODO; mọi LLM call có `try/except` + fallback; thêm cache enrichment. |
| Tự học & vận hành | 4 | Tự xử lý được lỗi model/infra (thinking-token, 503, Qdrant) nhưng còn dựa nhiều vào thử-sai, chưa đọc kỹ doc trước. |
| Problem solving | 5 | Debug được chuỗi lỗi end-to-end và rút ra finding "Production < Baseline" — chẩn đoán đúng nguyên nhân (chunk size + enrichment) thay vì chỉ báo lỗi. |

**Tổng kết:** Nắm vững pipeline RAG production và quan trọng hơn là tư duy *đo lường trước khi tối ưu* — bài học mang thẳng sang project EduDraft.
