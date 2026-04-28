#!/usr/bin/env python3
"""
OCR 识别服务
主力: PaddleOCR (lang='ch') 行级识别
增强: PaddleOCR-en (英文优化) + RapidOCR + Tesseract
融合: 字符级投票 + 英文间距修复
"""

import json
import base64
import os
import sys
import urllib.request
import urllib.error
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
import re
import difflib

from paddleocr import PaddleOCR
from rapidocr_onnxruntime import RapidOCR

_engine_ch = None
_engine_en = None
_rapid_ocr = None

def get_engine_ch():
    global _engine_ch
    if _engine_ch is None:
        print("Loading PaddleOCR-ch model...")
        _engine_ch = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        print("PaddleOCR-ch model loaded.")
    return _engine_ch

def get_engine_en():
    global _engine_en
    if _engine_en is None:
        print("Loading PaddleOCR-en model...")
        _engine_en = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("PaddleOCR-en model loaded.")
    return _engine_en

def get_rapid():
    global _rapid_ocr
    if _rapid_ocr is None:
        print("Loading RapidOCR model...")
        _rapid_ocr = RapidOCR()
        print("RapidOCR model loaded.")
    return _rapid_ocr


def tesseract_ocr(img_bytes):
    """Tesseract OCR for supplementary text detection."""
    import cv2
    import numpy as np
    import subprocess
    import tempfile

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    # Preprocess: grayscale + CLAHE + Otsu binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save temp file for tesseract
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(tmp.name, binary)
    tmp.close()

    try:
        result = subprocess.run(
            ['tesseract', tmp.name, 'stdout', '-l', 'eng+chi_sim', '--psm', '6'],
            capture_output=True, text=True, timeout=30
        )
        text = result.stdout.strip()
        if not text:
            return []
        # Split into lines, filter empty
        lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped:
                # Filter Tesseract noise: lines with too many special chars or gibberish
                alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
                if alpha_ratio < 0.3:
                    continue
                lines.append(stripped)
        return lines
    except Exception as e:
        print(f"[Tesseract] Failed: {e}", file=sys.stderr)
        return []
    finally:
        os.unlink(tmp.name)


def ocr_image(img_bytes):
    """Run PaddleOCR-ch, return list of (text, score, y_pos, x_pos) sorted by position"""
    engine = get_engine_ch()
    import cv2
    import numpy as np
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    result = engine.ocr(img, cls=True)
    if not result or not result[0]:
        return []
    lines = []
    for item in result[0]:
        bbox, (text, score) = item[0], item[1]
        y_pos = bbox[0][1]
        x_pos = bbox[0][0]
        lines.append((text, float(score), y_pos, x_pos))
    lines.sort(key=lambda x: (round(x[2] / 30) * 30, x[3]))
    return lines


def paddleocr_en(img_bytes):
    """PaddleOCR English mode - better word spacing for English text."""
    engine = get_engine_en()
    import cv2
    import numpy as np
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    result = engine.ocr(img, cls=True)
    if not result or not result[0]:
        return []
    lines = []
    for item in result[0]:
        bbox, (text, score) = item[0], item[1]
        y_pos = bbox[0][1]
        x_pos = bbox[0][0]
        lines.append((text, float(score), y_pos, x_pos))
    lines.sort(key=lambda x: (round(x[2] / 30) * 30, x[3]))
    return lines


def rapidocr_run(img_bytes):
    """RapidOCR - fast ONNX-based OCR with good English support."""
    engine = get_rapid()
    import cv2
    import numpy as np
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    result, _ = engine(img)
    if not result:
        return []
    lines = []
    for item in result:
        bbox, text, score = item
        y_pos = bbox[0][1]
        x_pos = bbox[0][0]
        lines.append((text, float(score), y_pos, x_pos))
    lines.sort(key=lambda x: (round(x[2] / 30) * 30, x[3]))
    return lines


def is_garbled(text):
    """Detect garbled OCR output - lines with too many short nonsensical fragments."""
    words = re.findall(r'[a-zA-Z]+', text)
    if len(words) < 3:
        return False  # too short to judge
    # Count short fragments (1-3 chars) - garbled text has lots of these
    short_count = sum(1 for w in words if len(w) <= 3)
    short_ratio = short_count / len(words)
    if short_ratio > 0.45:
        return True
    # Check if line has alternating very short words (pattern of garbled OCR)
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 3.5:
        return True
    return False


def filter_noise_lines(lines):
    """Filter out noise: low confidence, single chars, pure punctuation/numbers, very short nonsense"""
    filtered = []
    for text, score, y, x in lines:
        stripped = text.strip()
        if score < 0.6:
            continue
        if len(stripped) < 2:
            continue
        if len(stripped) <= 2 and re.match(r'^[\u4e00-\u9fff]+$', stripped):
            continue
        # Filter very short lines that are likely noise (single word fragments)
        alpha_len = sum(c.isalpha() for c in stripped)
        if alpha_len <= 2 and len(stripped) <= 4:
            continue
        # Filter pure numbers/number labels
        if re.match(r'^\d{1,3}\.?$', stripped):
            continue
        filtered.append((text, score, y, x))
    return filtered


def filter_red_ink(img_bytes):
    """Remove red ink from image before OCR"""
    import cv2
    import numpy as np
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_mask = mask1 | mask2
    kernel = np.ones((2, 2), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    result = img.copy()
    result[red_mask > 0] = [255, 255, 255]
    success, encoded = cv2.imencode('.png', result)
    if success:
        return encoded.tobytes()
    return img_bytes


def fix_english_spacing(text):
    """Fix common OCR spacing issues in English text."""
    lines = text.split('\n')
    fixed = []
    for line in lines:
        # Fix missing space between lowercase and uppercase (word boundary)
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        # Fix missing space after punctuation followed by letter
        line = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', line)
        # Fix missing space before common English words
        for word in ['the','is','a','an','to','of','in','and','was','for','it','he','she','they','we','you','I','that','this','not','but','his','her','my','your','our','are','be','at','on','or','do','if','so','no','as','by','an','with','have','has','had','from','been','will','would','can','could','should','all','what','when','where','how','who','which','there','their','about']:
            line = re.sub(r'(?<=[a-zA-Z])(' + word + r')(?=[a-z])', r' \1', line)
        # Remove double spaces
        line = re.sub(r'  +', ' ', line)
        line = line.strip()
        fixed.append(line)
    return '\n'.join(fixed)


def smart_merge(paddle_lines, tesseract_lines):
    """
    Smart merge of PaddleOCR and Tesseract with character-level voting.
    For overlapping lines detected by both engines, pick the better version.
    Tesseract-only lines are placed based on matched neighbors' positions.
    """
    filtered = filter_noise_lines(paddle_lines)

    if not filtered and not tesseract_lines:
        return [], 0.0
    if not filtered:
        lines = [(t.strip(), 0.7, i * 30, 0) for i, t in enumerate(tesseract_lines) if t.strip()]
        return lines, 0.7
    if not tesseract_lines:
        avg = sum(s for _, s, _, _ in filtered) / len(filtered)
        return list(filtered), avg

    # Index PaddleOCR lines (allow duplicates with list)
    paddle_entries = []  # [(key, text, score, y, x, used)]
    for text, score, y, x in filtered:
        paddle_entries.append([text.strip().lower(), text, score, y, x, False])

    # Pre-compute Tesseract matches
    tess_matches = []  # [(t_idx, t_stripped, best_p_idx, ratio)]
    for t_idx, t_line in enumerate(tesseract_lines):
        t_stripped = t_line.strip()
        t_key = t_stripped.lower()
        if len(t_stripped) < 3:
            continue

        best_ratio = 0
        best_idx = -1
        for pidx, (pkey, p_text, p_score, p_y, p_x, p_used) in enumerate(paddle_entries):
            if p_used:
                continue
            ratio = difflib.SequenceMatcher(None, t_key, pkey).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = pidx

        tess_matches.append((t_idx, t_stripped, best_idx, best_ratio))

    merged = []

    # First pass: add matched lines (PaddleOCR or better Tesseract version)
    for t_idx, t_stripped, best_idx, ratio in tess_matches:
        if best_idx >= 0 and ratio > 0.4:
            p_text = paddle_entries[best_idx][1]
            p_score = paddle_entries[best_idx][2]
            p_y = paddle_entries[best_idx][3]
            p_x = paddle_entries[best_idx][4]
            paddle_entries[best_idx][5] = True

            t_alpha = sum(c.isalpha() for c in t_stripped) / max(len(t_stripped), 1)
            p_alpha = sum(c.isalpha() for c in p_text) / max(len(p_text), 1)

            if t_alpha > p_alpha + 0.05 and len(t_stripped) > len(p_text.strip()):
                merged.append((t_stripped, max(p_score, 0.7), p_y, p_x))
            else:
                merged.append((p_text, p_score, p_y, p_x))

    # Second pass: place unmatched Tesseract lines based on context
    # If a Tesseract line falls between two matched lines, estimate its Y position
    paddle_y_min = min(e[3] for e in paddle_entries)
    paddle_y_max = max(e[3] for e in paddle_entries)

    for t_idx, t_stripped, best_idx, ratio in tess_matches:
        if best_idx >= 0 and ratio > 0.4:
            continue  # already matched

        # Find nearest matched Tesseract lines to estimate position
        prev_y = None
        next_y = None
        for t2_idx, t2_stripped, t2_best_idx, t2_ratio in tess_matches:
            if t2_best_idx < 0 or t2_ratio <= 0.4:
                continue
            p_y = paddle_entries[t2_best_idx][3]
            if t2_idx < t_idx and (prev_y is None or p_y > prev_y):
                prev_y = p_y
            if t2_idx > t_idx and (next_y is None or p_y < next_y):
                next_y = p_y

        if prev_y is not None and next_y is not None:
            # Between two matched lines: interpolate
            est_y = (prev_y + next_y) / 2
        elif prev_y is not None:
            est_y = prev_y + 60
        elif next_y is not None:
            est_y = max(0, next_y - 60)
        else:
            # No context at all: place based on PaddleOCR range
            est_y = 0

        merged.append((t_stripped, 0.65, est_y, 0))

    # Add unmatched PaddleOCR lines
    for pkey, p_text, p_score, p_y, p_x, p_used in paddle_entries:
        if not p_used:
            merged.append((p_text, p_score, p_y, p_x))

    merged.sort(key=lambda x: (round(x[2] / 30) * 30, x[3]))

    # Deduplicate: remove exact and near-duplicate lines, filter noise
    deduped = []
    for item in merged:
        item_core = re.sub(r'[\s\W]', '', item[0].strip().lower())
        if len(item_core) < 3:
            continue  # skip noise
        # Filter garbled OCR lines (short nonsensical fragments)
        if is_garbled(item[0]):
            continue
        # Skip Tesseract lines that look like edge/border artifacts
        if item[1] < 0.7:  # Tesseract supplement line
            # Filter lines ending with | (page border artifacts)
            if item[0].rstrip().endswith('|') or item[0].lstrip().startswith('|'):
                continue
            # Filter lines with mostly very short words (≤2 chars) relative to total
            segments = re.findall(r'[a-zA-Z]+', item[0])
            if segments and len(segments) >= 5:
                short_count = sum(1 for s in segments if len(s) <= 2)
                if short_count / len(segments) > 0.5:
                    continue  # more than half the words are 1-2 chars
        is_dup = False
        for prev in deduped:
            prev_core = re.sub(r'[\s\W]', '', prev[0].strip().lower())
            # Exact same text (after normalization)
            if item_core == prev_core:
                if item[1] > prev[1]:
                    deduped[deduped.index(prev)] = item
                is_dup = True
                break
            # Near-duplicate: high similarity within same Y band (±30px)
            if abs(prev[2] - item[2]) < 30:
                r = difflib.SequenceMatcher(None, item_core, prev_core).ratio()
                if r > 0.85:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(item)

    avg = sum(s for _, s, _, _ in deduped) / len(deduped) if deduped else 0
    return deduped, avg


def validate_vision(vision_text, ocr_text):
    """Validate Vision result against OCR to reject hallucinations."""
    if not vision_text or not ocr_text:
        return False

    v_lines = [l.strip() for l in vision_text.split('\n') if l.strip()]
    o_lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    if not v_lines or not o_lines:
        return False

    # Vision should have roughly similar number of lines (±50%)
    if len(v_lines) < len(o_lines) * 0.3 or len(v_lines) > len(o_lines) * 2.5:
        return False

    # Check character overlap
    v_core = re.sub(r'[\s\W]', '', vision_text.lower())
    o_core = re.sub(r'[\s\W]', '', ocr_text.lower())
    if not v_core or not o_core:
        return False

    # Use sequence matching for content similarity
    ratio = difflib.SequenceMatcher(None, v_core, o_core).ratio()
    if ratio < 0.2:
        return False

    return True


def find_supplements(ocr_lines, vision_text):
    """Find OCR lines not covered by Vision result."""
    if not vision_text:
        return [text for text, _, _, _ in ocr_lines if text.strip()]

    vision_core = re.sub(r'[\s\W]', '', vision_text.lower())
    supplements = []
    for text, score, y, x in ocr_lines:
        stripped = text.strip()
        if len(stripped) < 3:
            continue
        core = re.sub(r'[\s\W]', '', stripped.lower())
        if len(core) < 3:
            continue
        # Check if this line's core content appears in Vision
        if core not in vision_core:
            supplements.append(stripped)

    return supplements


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


def vision_transcribe(img_b64):
    """Best-effort Vision transcription. Returns text or None on any failure."""
    api_key = os.environ.get("VISION_API_KEY", "")
    if not api_key:
        return None

    try:
        import cv2
        import numpy as np
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            max_dim = 3000
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_b64 = base64.b64encode(encoded.tobytes()).decode()
            print(f"[Vision] Resized to {img.shape[1]}x{img.shape[0]}", file=sys.stderr)
    except Exception as e:
        print(f"[Vision] Resize skipped: {e}", file=sys.stderr)

    base_url = os.environ.get("VISION_API_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    req_body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 16384,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": "这是一张考试试卷的图片。请仔细辨认并逐字转录所有印刷体文字。\n规则：\n- 保持原始排版，每行对应图片中的一行文字\n- 忽略所有手写内容和红色批注\n- 对于无法确定的字符，用【?】标记\n- 不要添加任何解释或说明\n- 不要翻译或改写任何内容\n- 直接输出文字"}
            ]
        }]
    }).encode("utf-8")

    req = urllib.request.Request(base_url + "/v1/messages", data=req_body, headers=_api_headers(), method="POST")

    try:
        print("[Vision] Calling API...", file=sys.stderr)
        with _build_opener().open(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text and len(text) > 20:  # Sanity check: ignore too-short results
                        # Quick sanity: if >30% are 【?】, it's a failed read
                        q_count = text.count("【?】")
                        if q_count > 5 and q_count / max(text.count('\n'), 1) > 0.3:
                            print(f"[Vision] Rejected: too many unreadable ({q_count})", file=sys.stderr)
                            return None
                        print(f"[Vision] Done ({len(text)} chars)", file=sys.stderr)
                        return text
            return None
    except Exception as e:
        print(f"[Vision] Failed: {e}", file=sys.stderr)
        return None


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
        # Find overlap between tail of previous and head of current
        best_overlap = 0
        for n in range(1, min(8, len(prev_lines), len(curr_lines) + 1)):
            prev_tail = [l.strip() for l in prev_lines[-n:] if l.strip()]
            curr_head = [l.strip() for l in curr_lines[:n] if l.strip()]
            if prev_tail and curr_head and prev_tail == curr_head:
                best_overlap = n
            elif prev_tail and curr_head:
                # Partial overlap: check similarity
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

            # Mode selection: local (4-engine OCR) or ai (glm-4v-flash)
            mode = data.get("mode", "local")
            if mode == "ai":
                ai_config = data.get("ai_config", {})
                print(f"[Vision] AI mode requested, model={ai_config.get('model','default')}, image size: {len(img_bytes)} bytes", file=sys.stderr, flush=True)
                text = vision_ocr_glm(img_bytes,
                    model=ai_config.get("model", "glm-4v-flash"),
                    api_key=ai_config.get("api_key") or None,
                    api_url=ai_config.get("api_url") or None)
                if text:
                    self.send_json({"text": text, "status": "done", "strategy": "glm-4v-flash"})
                else:
                    self.send_json({"text": "(AI识别失败)", "status": "error", "strategy": "glm-4v-flash"})
                return

            # Run 4 OCR engines concurrently
            paddle_result = None
            paddle_en_result = None
            rapid_result = None
            tesseract_lines = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                def run_paddle():
                    filtered = filter_red_ink(img_bytes)
                    result = ocr_image(filtered)
                    print(f"[PaddleOCR-ch] Got {len(result)} lines", file=sys.stderr, flush=True)
                    return result

                def run_paddle_en():
                    filtered = filter_red_ink(img_bytes)
                    result = paddleocr_en(filtered)
                    print(f"[PaddleOCR-en] Got {len(result)} lines", file=sys.stderr, flush=True)
                    return result

                def run_rapid():
                    filtered = filter_red_ink(img_bytes)
                    result = rapidocr_run(filtered)
                    print(f"[RapidOCR] Got {len(result)} lines", file=sys.stderr, flush=True)
                    return result

                def run_tesseract():
                    lines = tesseract_ocr(img_bytes)
                    print(f"[Tesseract] Got {len(lines)} lines", file=sys.stderr, flush=True)
                    return lines

                f1 = executor.submit(run_paddle)
                f2 = executor.submit(run_paddle_en)
                f3 = executor.submit(run_rapid)
                f4 = executor.submit(run_tesseract)

                try:
                    paddle_result = f1.result(timeout=60)
                except Exception as e:
                    print(f"[PaddleOCR-ch] Exception: {e}", file=sys.stderr)
                try:
                    paddle_en_result = f2.result(timeout=60)
                except Exception as e:
                    print(f"[PaddleOCR-en] Exception: {e}", file=sys.stderr)
                try:
                    rapid_result = f3.result(timeout=60)
                except Exception as e:
                    print(f"[RapidOCR] Exception: {e}", file=sys.stderr)
                try:
                    tesseract_lines = f4.result(timeout=30)
                except Exception as e:
                    print(f"[Tesseract] Exception: {e}", file=sys.stderr)

            # Merge all engines: paddle-ch as primary, others as supplements
            all_supplements = (paddle_en_result or []) + (rapid_result or [])
            merged_lines, avg_score = smart_merge(
                paddle_result or [], tesseract_lines or []
            )

            # Add paddle-en and rapid lines as supplements
            if all_supplements:
                merged_text = "\n".join(t.lower() for t, _, _, _ in merged_lines)
                for text, score, y, x in all_supplements:
                    # Filter garbled text
                    if is_garbled(text):
                        continue
                    core = re.sub(r'[\s\W]', '', text.lower())
                    if len(core) < 3:
                        continue
                    # Check if already covered
                    if core in re.sub(r'[\s\W]', '', merged_text):
                        continue
                    # Check similarity with existing lines
                    is_dup = False
                    for mt, ms, my, mx in merged_lines:
                        mt_core = re.sub(r'[\s\W]', '', mt.lower())
                        r = difflib.SequenceMatcher(None, core, mt_core).ratio()
                        if r > 0.7:
                            is_dup = True
                            break
                    if not is_dup:
                        merged_lines.append((text, max(score, 0.65), y, x))

                merged_lines.sort(key=lambda x: (round(x[2] / 30) * 30, x[3]))

            # English spacing post-processing
            ocr_text = fix_english_spacing("\n".join(text for text, _, _, _ in merged_lines))

            if ocr_text:
                final_text = ocr_text
                strategy = "4-engine-merge"
            else:
                final_text = "(未识别到文字)"
                strategy = "none"

            if not final_text or final_text.strip() == "":
                self.send_json({"text": "(未识别到文字)", "status": "done", "strategy": strategy})
                return

            self.send_json({
                "text": final_text,
                "status": "done",
                "confidence": round(avg_score, 3),
                "strategy": strategy
            })

        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e), "status": "error"}, 500)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "model_loaded": _engine_ch is not None})
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
    print(f"  POST /ocr     - PaddleOCR + AI review (Vision best-effort)")
    print(f"  GET  /health   - health check")
    server.serve_forever()


if __name__ == "__main__":
    # Pre-load all OCR models in main thread
    print("Pre-loading OCR models...")
    get_engine_ch()
    get_engine_en()
    get_rapid()
    print("All models loaded.")
    main()
