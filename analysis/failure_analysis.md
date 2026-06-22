# Failure Analysis — Lab 18: Production RAG

**Cấu hình chạy:** Hierarchical chunking (M1) → Enrichment combined 1-call/chunk (M5) → Hybrid BM25+Dense+RRF (M2) → CrossEncoder rerank top-3 (M3) → LLM answer (`gemini-2.5-flash-lite`) → RAGAS (M4)
**Test set:** 20 câu hỏi · **Embedding:** bge-m3 (retrieval) + all-MiniLM-L6-v2 (RAGAS)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|------|
| Faithfulness | 0.754 | 0.700 | −0.054 |
| Answer Relevancy | 0.532 | 0.470 | −0.062 |
| Context Precision | 0.808 | 0.775 | −0.033 |
| Context Recall | 0.883 | 0.783 | −0.100 |

> **3/4 metric Production ≥ 0.70** (đạt mức cao nhất thang RAGAS rubric).
>
> ⚠️ **Finding quan trọng — Production THẤP HƠN Baseline ở cả 4 metric.** "Phức tạp hơn ≠ tốt hơn".
> Đây không phải lỗi code (cả 2 pipeline chạy đúng) mà là bài học production thực tế:
> 1. **Enrichment phản tác dụng**: nhiều chunk bị `503` khi gọi LLM enrich → rơi về fallback extractive (chất lượng không đồng đều); prepend context + thay metadata gốc bằng auto-metadata làm "loãng" embedding so với raw text.
> 2. **Hierarchical child chunks nhỏ (256 ký tự × 100)** vs basic (500 ký tự × 57) → mỗi chunk chứa ít thông tin ⇒ `context_recall` giảm mạnh nhất (−0.10).
> 3. **Variance**: RAGAS đánh giá bằng model lite "thinking" trên 20 câu ⇒ chênh 0.03–0.06 có phần là nhiễu thống kê.
>
> **Hành động đề xuất:** (a) chỉ enrich khi LLM mạnh & ổn định (không 503); (b) tăng child_size hoặc dùng hierarchical retrieve-child-return-parent đúng cách để không mất context; (c) A/B test từng kỹ thuật riêng thay vì bật tất cả.

---

## Phân tích tổng thể (Diagnostic Tree)

Bottom-10 cho thấy 2 nhóm lỗi rõ rệt:

1. **Faithfulness = 0.0 dù context ĐÚNG** (precision/recall = 1.0): nhóm câu thử việc/tạm ứng. LLM trả lời nhưng RAGAS không trích được "statement" khớp context → câu trả lời quá ngắn/diễn giải lệch ("No statements were generated from the answer"). → **Lỗi ở Generation, không phải Retrieval.**
2. **answer_relevancy thấp dù faithfulness = 1.0**: câu trả lời đúng-với-context nhưng *không trả lời thẳng câu hỏi* (lan man hoặc trả lời thiếu trọng tâm).

→ Cả hai đều trỏ về **prompt sinh câu trả lời** + đặc tính "thinking" của model lite, **không phải** lỗi chunking/search.

---

## Bottom-5 Failures

### #1 — avg 0.188
- **Question:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **Worst metric:** faithfulness = 0.0 (precision 0.0, recall 0.5, relevancy 0.25)
- **Error Tree:** Output sai → Context SAI (precision 0.0) → câu multi-hop (nghỉ phép + lương) cần gộp 2 nguồn → retrieval chỉ lấy 1 phần
- **Root cause:** Câu hỏi *multi-hop* (2 dữ kiện ở 2 tài liệu khác nhau); top-3 sau rerank không đủ phủ cả 2.
- **Suggested fix:** Tăng `RERANK_TOP_K` cho câu multi-hop, hoặc query decomposition (tách thành 2 sub-query).

### #2 — avg 0.378
- **Question:** "Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"
- **Worst metric:** faithfulness = 0.0 (recall 1.0, precision 0.33, relevancy 0.18)
- **Error Tree:** Output sai → Context ĐÚNG (recall 1.0) → Generation hỏng
- **Root cause:** Context có thông tin (recall=1.0) nhưng câu trả lời không bám số liệu cụ thể → faithfulness 0.
- **Suggested fix:** Prompt yêu cầu trích **nguyên văn con số** từ context; `temperature=0`.

### #3 — avg 0.415
- **Question:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"
- **Worst metric:** context_precision = 0.0 (faithfulness 1.0, recall 0.0)
- **Error Tree:** Output đúng-format → Context SAI (precision 0.0, recall 0.0) → retrieval lấy nhầm ngưỡng phê duyệt
- **Root cause:** Câu *numeric-threshold* (55 triệu rơi vào bậc phê duyệt nào) — search không phân biệt được các mốc 30tr/50tr/100tr.
- **Suggested fix:** Metadata filter theo khoảng giá trị, hoặc enrichment thêm bảng ngưỡng vào chunk.

### #4 — avg 0.500
- **Question:** "Nhân viên thử việc có được nghỉ phép năm không?"
- **Worst metric:** faithfulness = 0.0 (precision 1.0, recall 1.0, relevancy 0.0)
- **Error Tree:** Output sai → Context HOÀN HẢO (precision=recall=1.0) → Generation hỏng hoàn toàn
- **Root cause:** Câu *negation/yes-no*; context đầy đủ nhưng LLM trả lời không thành câu khẳng định trích được → relevancy & faithfulness = 0.
- **Suggested fix:** Prompt few-shot cho dạng yes/no ("Trả lời Có/Không + trích điều khoản").

### #5 — avg 0.500
- **Question:** "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"
- **Worst metric:** faithfulness = 0.0 (precision 1.0, recall 1.0, relevancy 0.0)
- **Error Tree:** Giống #4 — context hoàn hảo, generation hỏng
- **Root cause:** Cùng pattern negation/yes-no; model lite trả lời cụt.
- **Suggested fix:** Như #4 + tăng `max_tokens` cho câu trả lời để model không bị cụt do thinking-token.

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên thử việc có được nghỉ phép năm không?" (#4)

**Error Tree walkthrough:**
1. Output đúng? → **Không** (faithfulness 0.0, relevancy 0.0)
2. Context đúng? → **Có** (precision 1.0, recall 1.0) — chunk đúng đã được retrieve
3. Query rewrite OK? → Có — câu rõ ràng
4. **Fix ở bước: Generation (prompt + model)** — không phải retrieval

**Kết luận:** Pipeline retrieval (M1/M2/M3/M5) đã làm tốt phần khó. Nút thắt còn lại nằm ở **bước sinh câu trả lời**: model `gemini-2.5-flash-lite` (thinking model) trả lời cụt/lệch trọng tâm với câu yes-no và numeric.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Prompt template chuyên biệt cho từng loại câu (yes/no, numeric, multi-hop) + few-shot.
- Tăng `max_tokens` câu trả lời, `temperature=0`, hoặc đổi sang model không-thinking để tránh bị cụt.
- Query decomposition cho câu multi-hop (#1).
