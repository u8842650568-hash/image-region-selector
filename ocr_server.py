#!/usr/bin/env python3
"""
AI 图片识别服务
使用 OpenAI 兼容格式的视觉模型（GLM-4V 等）进行图文识别
"""

import json
import base64
import os
import sys
import urllib.request
import urllib.error
import traceback
import re
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import HTTPServer, BaseHTTPRequestHandler


def _build_opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({})
    )


def _api_headers(api_key=None):
    return {
        "Content-Type": "application/json",
        "x-api-key": api_key or os.environ.get("VISION_API_KEY", ""),
        "anthropic-version": "2023-06-01",
    }


def ai_review(ocr_text, api_key=None, base_url=None):
    """Call AI to review and fix OCR text errors. Returns reviewed text or original on failure."""
    api_key = api_key or os.environ.get("VISION_API_KEY", "")
    if not api_key:
        return ocr_text

    base_url = base_url or os.environ.get("VISION_API_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    req_body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": f"""以下是OCR识别出的文字。请修正明显的识别错误，但不要改变原文内容或添加新内容。

修正规则：
- 修正形近字错误（如 0/O, 1/l/I 混淆）
- 修正明显断行错误（同一句被错误拆成两行时合并）
- 修正标点符号错误
- 保持原文的所有换行和段落结构
- 不要翻译，不要改写，不要删减

---OCR原文---
{ocr_text}

---修正后---
"""
        }]
    }).encode("utf-8")

    req = urllib.request.Request(
        base_url + "/v1/messages",
        data=req_body,
        headers=_api_headers(api_key),
        method="POST",
    )
    opener = _build_opener()

    try:
        print("[AI Review] Calling API...", file=sys.stderr)
        with opener.open(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("content", [])
            if content and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            print("[AI Review] Done", file=sys.stderr)
                            return text
            return ocr_text
    except Exception as e:
        print(f"[AI Review] Failed: {e}", file=sys.stderr)
        return ocr_text


def _is_complete_markdown_table(text):
    """Check if text is a single complete Markdown table. Rejects multiple tables with repeated headers."""
    lines = text.strip().split('\n')
    header_count = 0
    sep_found = False
    data_rows = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                sep_found = True
                continue
            if sep_found:
                # After separator: non-numeric line = likely a new table header
                chars = stripped.replace('|', '').strip()
                if chars and not re.search(r'\d', chars):
                    header_count += 1
                    sep_found = False  # reset for next table block
                else:
                    data_rows += 1
            else:
                header_count += 1
    # Only skip Stage 2 if exactly one table header found
    return header_count == 1 and data_rows >= 1


def _extract_table(text):
    """Extract first Markdown table from text, stripping any thinking/reasoning before it."""
    # Find first line starting with |
    lines = text.split('\n')
    table_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('|') and '序号' in stripped:
            table_start = i
            break
    if table_start >= 0:
        return '\n'.join(lines[table_start:])
    # Fallback: find any line starting with |
    for i, line in enumerate(lines):
        if line.strip().startswith('|'):
            table_start = i
            break
    if table_start >= 0:
        return '\n'.join(lines[table_start:])
    return text


def ai_structure_table(ocr_text, api_key=None, base_url=None, model=None):
    """Call AI to convert raw OCR text into structured merged table.
    Supports Anthropic Messages API and OpenAI Chat Completions API based on base_url/model."""
    api_key = api_key or os.environ.get("VISION_API_KEY", "")
    base_url = base_url or os.environ.get("VISION_API_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    model = model or "claude-sonnet-4-20250514"
    if not api_key:
        return None

    prompt = f"""以下是OCR从一张单据图片中转录出的原始文字（可能格式混乱）。请将其整理为一个结构化的Markdown表格。

重要提示：
- OCR原文可能将相邻两条记录横向拼在同一行（如 | 编号A | 编号B |），你需要将它们拆分为独立的行
- 每个编号对应一条独立记录，请确保每条记录独占一行
- 不要遗漏任何一条记录，即使数据看起来重复也要保留

规则：
1. 找出所有散落的元信息（如编号、单位、日期、车号、制表人等），提取为表头的独立列，放在表格前面
2. 表头第一列固定为"序号"，后面依次放元信息字段名，最后放实际数据的列名
3. 元信息的值写在第一行对应列中，后续数据行这些列留空
4. 所有记录合并为一个表格，数据行按编号顺序排列，序号递增
5. 每一列的内容只放该列对应的信息
6. 只输出一个完整的Markdown表格，不要输出多个表格
7. 不要输出任何描述、解释、总结或思考过程，直接输出表格
8. 严格忠实于原文数据，不要编造或猜测数据
9. 统计原文中出现了多少个不同的编号，确保输出的数据行数量与编号数量完全一致

示例输出格式：
| 序号 | 编号 | 单位 | 日期 | 品名 | 毛重 | 皮重 | 净重 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0001 | XX公司 | 2026-04-30 | 粉条 | 55.34 | 25.33 | 24.74 |
| 2 | 0002 | XX公司 | 2026-04-30 | 豆腐 | 55.32 | 10.76 | 44.56 |

---OCR转录原文---
{ocr_text}

---结构化表格---
"""

    # Detect API format: Anthropic vs OpenAI
    use_anthropic = "/anthropic" in base_url or model.startswith("claude")

    if use_anthropic:
        req_body = json.dumps({
            "model": model,
            "max_tokens": 8192,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        base = base_url.rstrip("/")
        endpoint = base + "/v1/messages" if not base.endswith("/messages") else base
        headers = _api_headers(api_key)
    else:
        req_body = json.dumps({
            "model": model,
            "max_tokens": 8192,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        # Avoid double path if URL already includes /v1/chat/completions or /v1/messages
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/messages"):
            endpoint = base
        else:
            endpoint = base + "/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    req = urllib.request.Request(endpoint, data=req_body, headers=headers, method="POST")
    opener = _build_opener()

    try:
        print(f"[Structure] Calling {model}...", file=sys.stderr, flush=True)
        with opener.open(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if use_anthropic:
                content = data.get("content", [])
                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text.strip():
                                text = _extract_table(text)
                                print(f"[Structure] Done ({len(text)} chars)", file=sys.stderr, flush=True)
                                return text
            else:
                # OpenAI format
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text.strip():
                        text = _extract_table(text)
                        print(f"[Structure] Done ({len(text)} chars)", file=sys.stderr, flush=True)
                        return text
    except Exception as e:
        print(f"[Structure] Failed: {e}", file=sys.stderr, flush=True)
    return None


def vision_ocr_glm(img_bytes, model="glm-4v-flash", api_key=None, api_url=None, mode="text"):
    """Use vision model for OCR, with auto-split for long images."""
    if not api_key:
        api_key = os.environ.get("VISION_API_KEY", "")
    if not api_url:
        api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    if not api_key:
        return None

    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("[Vision-GLM] cv2.imdecode returned None", file=sys.stderr, flush=True)
            return None
        h, w = img.shape[:2]
        print(f"[Vision-GLM] Decoded: {w}x{h}", file=sys.stderr, flush=True)
        max_dim = 2000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    except Exception as e:
        print(f"[Vision-GLM] Decode failed: {e}", file=sys.stderr)
        return None

    MAX_H = 1800
    OVERLAP = 200
    if h <= MAX_H:
        sections = [img]
    else:
        sections = []
        y = 0
        while y < h:
            y2 = min(y + MAX_H, h)
            sections.append(img[y:y2, :, :])
            if y2 >= h:
                break
            y = y2 - OVERLAP

    # Build prompt based on mode
    if mode == "table" or mode == "table_raw":
        if mode == "table_raw":
            prompt_text = (
                "请识别图片中的表格内容，严格按照Markdown表格格式输出。规则：\n"
                "1. 每个独立表格原样输出，保持表头和数据行\n"
                "2. 表格中的非表格文字（如编号、单位、日期等）也作为数据行输出\n"
                "3. 第一行是表头，用|分隔每列\n"
                "4. 第二行是分隔行，如|---|---|---|\n"
                "5. 每一列的内容只放该列对应的信息\n"
                "6. 不要输出任何描述、解释或总结\n"
                "7. 如果有多个表格，用空行分隔\n"
                "8. 无法确定的字符用[?]标记"
            )
        else:
            prompt_text = (
                "请识别图片中的表格，严格按照Markdown表格格式输出。规则：\n"
                "1. 第一行必须是表头，用|分隔每列\n"
                "2. 第二行必须是分隔行，如|---|---|---|\n"
                "3. 从第三行开始是数据行，每行的列数必须与表头完全一致\n"
                "4. 每一列的内容只放该列对应的信息，不要将多列合并到一个单元格\n"
                "5. 保持与原图一致的行列结构，多个表格用空行分隔\n"
                "6. 不要输出任何描述、解释或总结\n"
                "7. 无法确定的字符用[?]标记"
            )
    else:
        prompt_text = (
            "请逐字逐句识别并转录图片中的所有文字。要求：\n"
            "1. 只输出图片中实际存在的文字，严格禁止添加任何描述、解释、总结或评论\n"
            "2. 不要输出类似这是一张、以下是、请注意等任何非原文内容\n"
            "3. 保持原始排版格式和换行\n"
            "4. 无法确定的字符用[?]标记"
        )

    # Pre-encode all sections
    encoded_sections = []
    for i, section in enumerate(sections):
        try:
            _, encoded = cv2.imencode('.jpg', section, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_b64 = base64.b64encode(encoded.tobytes()).decode()
            encoded_sections.append((i, img_b64))
        except Exception as e:
            print(f"[Vision-GLM] Encode section {i+1} failed: {e}", file=sys.stderr)

    def _call_section(idx, img_b64):
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }],
            "max_tokens": 1024,
            "temperature": 0
        }
        req = urllib.request.Request(api_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        opener = _build_opener()
        with opener.open(req, timeout=90) as resp:
            result = json.loads(resp.read().decode())
            if "error" in result:
                return idx, None, result['error']
            text = result["choices"][0]["message"]["content"]
            return idx, text.strip() if text and len(text.strip()) > 5 else None, None

    if len(encoded_sections) > 1:
        print(f"[Vision-GLM] Processing {len(encoded_sections)} sections in parallel...", file=sys.stderr, flush=True)
    all_text = [None] * len(encoded_sections)
    with ThreadPoolExecutor(max_workers=min(len(encoded_sections), 4)) as pool:
        futures = {pool.submit(_call_section, idx, b64): idx for idx, b64 in encoded_sections}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ridx, text, err = future.result()
                if err:
                    print(f"[Vision-GLM] Section {ridx+1} API error: {err}", file=sys.stderr, flush=True)
                elif text:
                    all_text[ridx] = text
                    print(f"[Vision-GLM] Section {ridx+1} done ({len(text)} chars)", file=sys.stderr, flush=True)
                else:
                    print(f"[Vision-GLM] Section {ridx+1} empty", file=sys.stderr, flush=True)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode()[:300] if e.fp else ''
                print(f"[Vision-GLM] Section {idx+1} HTTP {e.code}: {err_body}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Vision-GLM] Section {idx+1} failed: {e}", file=sys.stderr, flush=True)

    all_text = [t for t in all_text if t]

    if not all_text:
        return None

    if len(all_text) == 1:
        return all_text[0]

    # Merge sections: remove overlapping content
    merged = [all_text[0]]
    for section in all_text[1:]:
        prev_lines = merged[-1].split('\n')
        curr_lines = section.split('\n')
        best_overlap = 0
        for n in range(1, min(8, len(prev_lines), len(curr_lines) + 1)):
            prev_tail = [l.strip() for l in prev_lines[-n:] if l.strip()]
            curr_head = [l.strip() for l in curr_lines[:n] if l.strip()]
            if prev_tail and curr_head and prev_tail == curr_head:
                best_overlap = n
            elif prev_tail and curr_head:
                ratio = difflib.SequenceMatcher(None, '\n'.join(prev_tail), '\n'.join(curr_head)).ratio()
                if ratio < 0.6:
                    break
                best_overlap = n

        if best_overlap > 0:
            merged.append('\n'.join(curr_lines[best_overlap:]))
        else:
            merged.append(section)

    return '\n'.join(merged)


def vision_ocr_minimax(img_bytes, api_key=None, api_host=None, mode="text"):
    """Use MiniMax Token Plan VLM for OCR via /v1/coding_plan/vlm endpoint."""
    if not api_key:
        api_key = os.environ.get("VISION_API_KEY", "")
    if not api_host:
        api_host = "https://api.minimaxi.com"
    if not api_key:
        return None

    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("[Vision-MiniMax] cv2.imdecode returned None", file=sys.stderr, flush=True)
            return None
        h, w = img.shape[:2]
        print(f"[Vision-MiniMax] Decoded: {w}x{h}", file=sys.stderr, flush=True)
        max_dim = 2000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    except Exception as e:
        print(f"[Vision-MiniMax] Decode failed: {e}", file=sys.stderr)
        return None

    MAX_H = 1800
    OVERLAP = 200
    if h <= MAX_H:
        sections = [img]
    else:
        sections = []
        y = 0
        while y < h:
            y2 = min(y + MAX_H, h)
            sections.append(img[y:y2, :, :])
            if y2 >= h:
                break
            y = y2 - OVERLAP

    # Build prompt based on mode
    if mode == "table" or mode == "table_raw":
        prompt_text = (
            "请识别图片中的表格内容，严格按照Markdown表格格式输出。规则：\n"
            "1. 每个独立表格原样输出，保持表头和数据行\n"
            "2. 表格中的非表格文字（如编号、单位、日期等）也作为数据行输出\n"
            "3. 第一行是表头，用|分隔每列\n"
            "4. 第二行是分隔行，如|---|---|---|\n"
            "5. 每一列的内容只放该列对应的信息\n"
            "6. 不要输出任何描述、解释或总结\n"
            "7. 如果有多个表格，用空行分隔"
        )
    else:
        prompt_text = (
            "请逐字逐句识别并转录图片中的所有文字。要求：\n"
            "1. 只输出图片中实际存在的文字，严格禁止添加任何描述、解释、总结或评论\n"
            "2. 不要输出类似这是一张、以下是、请注意等任何非原文内容\n"
            "3. 保持原始排版格式和换行\n"
            "4. 无法确定的字符用[?]标记"
        )

    endpoint = api_host.rstrip("/") + "/v1/coding_plan/vlm"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "MM-API-Source": "Minimax-MCP",
    }

    # Pre-encode all sections
    encoded_sections = []
    for i, section in enumerate(sections):
        try:
            _, encoded = cv2.imencode('.jpg', section, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_b64 = base64.b64encode(encoded.tobytes()).decode()
            encoded_sections.append((i, img_b64))
        except Exception as e:
            print(f"[Vision-MiniMax] Encode section {i+1} failed: {e}", file=sys.stderr)

    def _call_section(idx, img_b64):
        data_url = f"data:image/jpeg;base64,{img_b64}"
        payload = json.dumps({"prompt": prompt_text, "image_url": data_url}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        opener = _build_opener()
        with opener.open(req, timeout=90) as resp:
            result = json.loads(resp.read().decode())
            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", -1)
            if status_code != 0:
                return idx, None, f"{base_resp.get('status_msg')} (code={status_code})"
            text = result.get("content", "")
            return idx, text.strip() if text and len(text.strip()) > 5 else None, None

    if len(encoded_sections) > 1:
        print(f"[Vision-MiniMax] Processing {len(encoded_sections)} sections in parallel...", file=sys.stderr, flush=True)
    all_text = [None] * len(encoded_sections)
    with ThreadPoolExecutor(max_workers=min(len(encoded_sections), 4)) as pool:
        futures = {pool.submit(_call_section, idx, b64): idx for idx, b64 in encoded_sections}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ridx, text, err = future.result()
                if err:
                    print(f"[Vision-MiniMax] Section {ridx+1} API error: {err}", file=sys.stderr, flush=True)
                elif text:
                    all_text[ridx] = text
                    print(f"[Vision-MiniMax] Section {ridx+1} done ({len(text)} chars)", file=sys.stderr, flush=True)
                else:
                    print(f"[Vision-MiniMax] Section {ridx+1} empty", file=sys.stderr, flush=True)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode()[:300] if e.fp else ''
                print(f"[Vision-MiniMax] Section {idx+1} HTTP {e.code}: {err_body}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Vision-MiniMax] Section {idx+1} failed: {e}", file=sys.stderr, flush=True)

    all_text = [t for t in all_text if t]

    if not all_text:
        return None

    if len(all_text) == 1:
        return all_text[0]

    # Merge sections: remove overlapping content (same logic as vision_ocr_glm)
    merged = [all_text[0]]
    for section in all_text[1:]:
        prev_lines = merged[-1].split('\n')
        curr_lines = section.split('\n')
        best_overlap = 0
        for n in range(1, min(8, len(prev_lines), len(curr_lines) + 1)):
            prev_tail = [l.strip() for l in prev_lines[-n:] if l.strip()]
            curr_head = [l.strip() for l in curr_lines[:n] if l.strip()]
            if prev_tail and curr_head and prev_tail == curr_head:
                best_overlap = n
            elif prev_tail and curr_head:
                ratio = difflib.SequenceMatcher(None, '\n'.join(prev_tail), '\n'.join(curr_head)).ratio()
                if ratio < 0.6:
                    break
                best_overlap = n
        if best_overlap > 0:
            merged.append('\n'.join(curr_lines[best_overlap:]))
        else:
            merged.append(section)

    return '\n'.join(merged)


class OCRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/ocr":
            self.handle_ocr()
        else:
            self.send_error(404)

    def handle_ocr(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            img_b64 = data.get("image_base64", "")
            if not img_b64:
                self.send_json({"error": "image_base64 required"}, 400)
                return

            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)

            # AI mode
            ai_config = data.get("ai_config", {})
            model = ai_config.get("model", "glm-4v-flash")
            ocr_mode = data.get("recog_type", "text")
            print(f"[Vision] AI mode, model={model}, recog_type={ocr_mode}, image size: {len(img_bytes)} bytes", file=sys.stderr, flush=True)

            is_minimax = model and model.startswith("minimax")

            if ocr_mode == "table":
                # Stage 1: vision model outputs raw tables
                if is_minimax:
                    raw_text = vision_ocr_minimax(img_bytes,
                        api_key=ai_config.get("api_key") or None,
                        api_host=ai_config.get("api_url") or None,
                        mode="table_raw")
                else:
                    raw_text = vision_ocr_glm(img_bytes,
                        model=model,
                        api_key=ai_config.get("api_key") or None,
                        api_url=ai_config.get("api_url") or None,
                        mode="table_raw")
                if not raw_text:
                    text = None
                else:
                    # Check if Stage 1 output is already a complete table
                    if _is_complete_markdown_table(raw_text):
                        print(f"[Vision] Stage 1 output is a complete table ({len(raw_text)} chars), skipping Stage 2", file=sys.stderr, flush=True)
                        text = raw_text
                    else:
                        print(f"[Vision] Stage 1 OCR done ({len(raw_text)} chars), structuring...", file=sys.stderr, flush=True)
                        # Stage 2: Claude structures raw text into table
                        text = ai_structure_table(raw_text,
                        api_key=ai_config.get("struct_api_key") or None,
                        base_url=ai_config.get("struct_api_url") or None,
                        model=ai_config.get("struct_model") or None)
                    if text is None:
                        text = raw_text  # fallback to raw OCR
                        print("[Vision] Structure failed, using raw OCR", file=sys.stderr, flush=True)
            else:
                if is_minimax:
                    text = vision_ocr_minimax(img_bytes,
                        api_key=ai_config.get("api_key") or None,
                        api_host=ai_config.get("api_url") or None,
                        mode=ocr_mode)
                else:
                    text = vision_ocr_glm(img_bytes,
                        model=model,
                        api_key=ai_config.get("api_key") or None,
                        api_url=ai_config.get("api_url") or None,
                        mode=ocr_mode)
            if text:
                self.send_json({"text": text, "status": "done", "strategy": model})
            else:
                self.send_json({"text": "(AI识别失败，请检查API配置)", "status": "error", "strategy": model})

        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e), "status": "error"}, 500)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_error(404)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[OCR] {args[0]}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    server = HTTPServer(("127.0.0.1", port), OCRHandler)
    print(f"OCR service running on http://127.0.0.1:{port}")
    print(f"  POST /ocr     - AI vision OCR")
    print(f"  GET  /health   - health check")
    server.serve_forever()


if __name__ == "__main__":
    main()
