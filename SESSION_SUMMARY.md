# Session Summary — 2026-07-11

## Pipeline đã chạy end-to-end ✅

```
> phân tích giá iphone 15 trên cellphones.com.vn
  [PLAN] data_provider -> analyze_pricing (2 task(s))
  [...] data_provider...    (crawl 5 URL, LLM extract, auto-wrap)
  [OK] Completed data_provider.
  [...] analyze_pricing...  (analyze_prices 1 call, handler compute)
  [OK] Giá trung bình (median) là 24.5 triệu VNĐ, dao động từ 13.0 triệu
       đến 32.0 triệu VNĐ dựa trên 12 sản phẩm...
```

## Đã làm được

### 1. Financial-research pattern: 1 tool = 1 complete unit of work
- **data_provider:** `search_listings(platform, query)` → handler tự crawl 5 URL + extract + aggregate → `ScrapeResult`
- **analyze_pricing:** `analyze_prices(dataset)` → handler tự compute stats + outliers + segments + recommendation text
- LLM chỉ gọi 1 action, nhận kết quả hoàn chỉnh — không aggregate, không format JSON

### 2. LLMExtractionStrategy (crawl4ai reference)
- eBay/Amazon → `JsonCssExtractionStrategy` (CSS)
- Site lạ → `LLMExtractionStrategy` + schema + instruction + `force_json_response`
- Instruction xử lý Vietnamese price format (12.990.000₫ → 12990000)

### 3. Auto-wrap + skip Reviewer
- Executor tự detect handler output → auto-wrap `{status, output, concerns}`
- Flag `_handler_produced` → Orchestrator skip Reviewer
- Không còn LLM format JSON → không còn Reviewer fail vô lý

### 4. Fix race condition `_handle_agent_response()`
- Method được gọi nhưng chưa định nghĩa → message loss → timeout → hang

### 5. Custom action `input_schema` auto-generate
- Handler methods có `input_schema` từ signature + docstring
- Hỗ trợ `list[float]` → `array items: number`

### 6. URL cache + accumulator
- `_url_cache` + `_accumulated_listings` trong `web_fetcher.py`
- Mỗi `fetch_web_page` tự cộng dồn, có `total_accumulated`

### 7. Unicode/emoji crash fix (Windows cp1252)
- `progress.py`: emoji → ASCII markers

### 8. Cleanup
- Xóa `session.start`/`session.complete` gửi tới `"runtime"` (không route được)
- Xóa `parse_listing_html` khỏi data_provider actions
- Xóa file `.csv` rác, thêm `.gitignore`
- Update Spec 001, 002, 004

---

## Bugs hiện tại

### A. Classifier sai: conversational → task
- **Triệu chứng:** "vậy thì mình mua để dùng sau đó bán thì tập trung vào sản phẩm nào" → classify thành `task` → search eBay "sản phẩm mua để dùng sau đó bán"
- **File:** `llend/orchestrator/classifier.py`
- **Hướng fix:** Tham khảo financial-research-analyst-agent xem họ route follow-up question như thế nào. Có thể thêm session context (completed tasks) vào classification prompt để LLM biết đây là follow-up.

### B. CSV export format
- File `iphone_15_cellphoneS_pricing_analysis.csv` ra được nhưng format lạ
- **File:** `llend/parsers/csv_exporter.py`

### C. `requests` dependency warning
- `pip install --upgrade requests urllib3 chardet charset_normalizer`

### D. `data_provider` enforcement nên để `suggested` (1 retry) hay `strict` (3 retries)?
- Hiện tại đang `suggested`. Với auto-wrap + skip Reviewer thì không cần retry nữa.

---

## Kiến trúc sau refactor

```
User input
  → Orchestrator classify → "task"
  → SkillPipeline: [data_provider, analyze_pricing]
  
  Task 1: data_provider
    → Executor LLM gọi search_listings(platform, query) ← 1 CALL
    → Handler tự crawl 5 URL + extract + aggregate
    → Auto-wrap → skip Reviewer → [OK] Completed
  
  Task 2: analyze_pricing  
    → Executor LLM gọi analyze_prices(dataset) ← 1 CALL
    → Handler tự compute stats + outliers + segments + recommendation text
    → Auto-wrap → skip Reviewer → [OK] <recommendation text>
```

---

## Cách test

```powershell
python -m llend
> phân tích giá iphone 15 trên cellphones.com.vn
# Expect: sau ~20s, hiện text phân tích giá
```
