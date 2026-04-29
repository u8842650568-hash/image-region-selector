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
from http.server import HTTPServer, BaseHTTPRequestHandler


def _build_opener():
    proxy = os.environ.get("http_proxy", os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890"))
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()


def _api_headers():
    return {
        "Content-Type": "application/json",
        "x-api-key": os.environ.get("VISION_API_KEY", ""),
        "anthropic-version": "2023-06-01",
    }


def ai_review(ocr_text):
    """Call AI to review and fix OCR text errors. Returns reviewed text or original on failure."""
    api_key = os.environ.get("VISION_API_KEY", "")
    if not api_key:
        return ocr_text

    base_url = os.environ.get("VISION_API_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
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
        headers=_api_headers(),
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


def vision_ocr_glm(img_bytes, model="glm-4v-flash", api_key=None, api_url=None):
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
            return None
        h, w = img.shape[:2]
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

    api_url = api_url
    all_text = []

    for i, section in enumerate(sections):
        if len(sections) > 1:
            print(f"[Vision-GLM] Section {i+1}/{len(sections)}...", file=sys.stderr, flush=True)
        try:
            _, encoded = cv2.imencode('.jpg', section, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(encoded.tobytes()).decode()
        except Exception as e:
            print(f"[Vision-GLM] Encode failed: {e}", file=sys.stderr)
            continue

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": "请逐字逐句识别并转录图片中的所有文字。要求：\n1. 只输出图片中实际存在的文字，严格禁止添加任何描述、解释、总结或评论\n2. 不要输出类似\"这是一张...\"、\"以下是...\"、\"请注意...\"等任何非原文内容\n3. 保持原始排版格式和换行\n4. 无法确定的字符用[?]标记"}
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
        # Use direct connection for zhipu API (proxy may cause 1210 error)
        direct_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

        try:
            with direct_opener.open(req, timeout=90) as resp:
                result = json.loads(resp.read().decode())
                if "error" in result:
                    print(f"[Vision-GLM] API error: {result['error']}", file=sys.stderr, flush=True)
                    continue
                text = result["choices"][0]["message"]["content"]
                if text and len(text.strip()) > 5:
                    all_text.append(text.strip())
                    print(f"[Vision-GLM] Section {i+1} done ({len(text)} chars)", file=sys.stderr, flush=True)
                else:
                    print(f"[Vision-GLM] Section {i+1} empty", file=sys.stderr, flush=True)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()[:300] if e.fp else ''
            print(f"[Vision-GLM] Section {i+1} HTTP {e.code}: {err_body}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Vision-GLM] Section {i+1} failed: {e}", file=sys.stderr, flush=True)

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
            print(f"[Vision] AI mode, model={model}, image size: {len(img_bytes)} bytes", file=sys.stderr, flush=True)
            text = vision_ocr_glm(img_bytes,
                model=model,
                api_key=ai_config.get("api_key") or None,
                api_url=ai_config.get("api_url") or None)
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
