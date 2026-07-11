# Session Summary — 2026-07-11

## Đã làm được

### 1. LLMExtractionStrategy cho site lạ (CellphoneS)
- **File:** `llend/parsers/web_fetcher.py`
- Dùng crawl4ai's `LLMExtractionStrategy` với schema + instruction đúng chuẩn
- eBay/Amazon → `JsonCssExtractionStrategy` (CSS, nhanh, 0 token)
- Site lạ → `LLMExtractionStrategy` (DeepSeek extract, tự động)
- Tự detect platform từ URL domain
- Đã extract được giá thật từ CellphoneS: median 22.9tr, min 18.9tr, max 30.9tr

### 2. Fix race condition `_handle_agent_response()`
- **File:** `llend/orchestrator/agent.py`
- Method được gọi nhưng chưa định nghĩa → `_main_loop` race với `_await_response`
- Fix: re-queue message để `_await_response` consume

### 3. Xóa `session.start` / `session.complete` gửi tới `"runtime"`
- **File:** `llend/orchestrator/agent.py`
- `"runtime"` không phải agent type → message luôn bị drop

### 4. Xóa `parse_listing_html` khỏi data_provider actions
- **File:** `llend/skills/data_provider/skill.md`
- Theo crawl4ai pattern: fetch đã bao gồm extract, không cần parse riêng
- Thêm `get_cached_listings` custom action để handler aggregate thay vì LLM

### 5. Custom action `input_schema` auto-generate
- **Files:** `llend/tool_bridge/bridge.py`, `llend/registry/registry.py`
- Handler methods giờ có input_schema từ signature → LLM biết params cần truyền
- Hỗ trợ generic types (`list[float]` → array items type)
- Parse docstring cho parameter descriptions

### 6. URL cache + response trimming
- **File:** `llend/parsers/web_fetcher.py`
- Module-level `_url_cache` ngăn crawl trùng lặp
- Không trả markdown khi đã có structured listings (giảm context overload)

### 7. Unicode/emoji crash trên Windows cp1252
- **File:** `llend/orchestrator/progress.py`
- Thay emoji bằng ASCII markers: 📋→[PLAN], ✅→[OK], v.v.
- Failure UX message: English text thay vì tiếng Việt có dấu

### 8. Spec documents updated
- **Files:** `docs/specs/001-*.md`, `002-*.md`, `004-*.md`
- Deprecated `session.start`/`session.complete`
- Documented extraction strategy, race condition fix, routing note

### 9. Cleanup
- Xóa file `.csv` rác khỏi git, thêm `.gitignore`

---

## Bugs hiện tại

### A. data_provider vẫn retry 1 lần rồi skip
- **Triệu chứng:** Log hiện `[WARN] [data_provider] Retrying (1/1)... → Skipping task`
- **Nguyên nhân:** `enforcement: suggested` → 1 retry. Executor LLM (DeepSeek) aggregate chưa chuẩn `ScrapeResult` format, Reviewer fail.
- **Impact:** Pipeline vẫn chạy tiếp (analyze_pricing vẫn spawn), nhưng data_provider output có thể thiếu field.
- **Fix đề xuất:** Tăng enforcement lên `strict` (3 retries) trong `skill.md`. Hoặc code handler `fetch_web_page` tự accumulate thay vì để LLM gọi `get_cached_listings`.

### B. analyze_pricing handler sai params
- **Triệu chứng:** `TypeError: calculate_market_median() missing 1 required positional argument: 'prices'`
- **Nguyên nhân:** Đã fix bằng `signature_to_schema` auto-generate. Nhưng cần verify lại sau khi registry reload.
- **Impact:** analyze_pricing fallback về LLM tự tính toán (vẫn ra kết quả gần đúng).

### C. Executor LLM crawl quá nhiều URL
- **Triệu chứng:** 7-14 URL được fetch thay vì 3-5 như skill.md yêu cầu
- **Nguyên nhân:** LLM không tuân thủ instruction "limit 3-5 URLs"
- **Fix đề xuất:** Limit trong code (handler `fetch_web_page` max calls) hoặc giảm `max_tool_calls`

### D. `requests` dependency warning
- **Triệu chứng:** `RequestsDependencyWarning: urllib3 (1.26.15) or chardet...`
- **Fix:** `pip install --upgrade requests urllib3 chardet charset_normalizer` (đã chạy 1 lần, có thể cần pin versions)

### E. analyze_pricing CSV export format
- **Triệu chứng:** CSV ra format hơi lạ (1 dòng data, header không đúng chuẩn)
- **File:** `iphone_15_pricing_analysis.csv`
- **Fix đề xuất:** Sửa `csv_exporter.py` hoặc `analyze_pricing` handler export logic

---

## Branch / commits

Tất cả đã merge vào `master`. Commits chính:
```
493604d fix: LLM extraction instruction for Vietnamese price format
eb3609b feat: add get_cached_listings handler action
d5d3199 perf: trim response payload when listings are present
c2bad1d fix: auto-generate input_schema for custom handler actions
c016d6f fix: default extract_listings=True in fetch_web_page
c000eb0 docs: update Spec 001 & 002 to reflect v0.1 changes
584a1ef chore: remove stale csv/xlsx artifacts
b7d73da fix: remove parse_listing_html from data_provider actions
5ca6602 fix: LLMExtractionStrategy schema + instruction correctly
9d68a96 fix: cache lookup for ALL unknown platforms
6d5a2a8 fix: check web_fetcher cache in parse_product_listing
bdececb fix: remove session.complete message
0574230 fix: replace emoji with ASCII
6c1ae45 fix: replace arrow and em dash
4b83857 fix: define _handle_agent_response, remove dead session.start
```

---

## Cách test nhanh

```powershell
cd "D:\FULearning\Over\Python\Agent Harness\Code\llend_harness"
python -m llend
> phân tích giá iphone 15 trên cellphones.com.vn
```

Kết quả mong đợi: sau 2-5 phút, ra file `iphone_15_pricing_analysis.csv` với median ~22.9tr.
