package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"strings"
	"unicode"
	"time"
)

// ==================== HTTP Client (proxy bypass) ====================

// newHTTPClient creates an HTTP client that bypasses system proxy settings.
// This prevents Go's HTTP client from hanging when system http_proxy is set
// but the proxy is not reachable or not desired for API calls.
func newHTTPClient(timeout time.Duration) *http.Client {
	transport := &http.Transport{
		// Disable proxy for direct API calls
		Proxy: func(*http.Request) (*url.URL, error) {
			return nil, nil
		},
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
	}
}

// ==================== Image Preprocessing ====================

// decodeImage decodes base64 data URL to image.Image
func decodeImage(b64Data string) (image.Image, string, error) {
	if strings.HasPrefix(b64Data, "data:") {
		if idx := strings.Index(b64Data, ","); idx >= 0 {
			b64Data = b64Data[idx+1:]
		}
	}
	raw, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		raw, err = base64.RawStdEncoding.DecodeString(b64Data)
		if err != nil {
			return nil, "", err
		}
	}
	img, format, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, "", err
	}
	return img, format, nil
}

// encodeImageJPEG encodes image to JPEG base64 data URL
func encodeImageJPEG(img image.Image, quality int) string {
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality})
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// bilinearSample samples pixel at (x, y) using bilinear interpolation
func bilinearSample(img image.Image, x, y float64) (float64, float64, float64, float64) {
	bounds := img.Bounds()
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	fx := x - float64(x0)
	fy := y - float64(y0)

	sample := func(px, py int) (float64, float64, float64, float64) {
		if px < bounds.Min.X || px >= bounds.Max.X || py < bounds.Min.Y || py >= bounds.Max.Y {
			return 0, 0, 0, 0
		}
		r, g, b, a := img.At(px, py).RGBA()
		return float64(r), float64(g), float64(b), float64(a)
	}

	r00, g00, b00, a00 := sample(x0, y0)
	r10, g10, b10, a10 := sample(x1, y0)
	r01, g01, b01, a01 := sample(x0, y1)
	r11, g11, b11, a11 := sample(x1, y1)

	mix := func(a, b, t float64) float64 {
		return a + (b-a)*t
	}

	r := mix(mix(r00, r10, fx), mix(r01, r11, fx), fy)
	g := mix(mix(g00, g10, fx), mix(g01, g11, fx), fy)
	b := mix(mix(b00, b10, fx), mix(b01, b11, fx), fy)
	a := mix(mix(a00, a10, fx), mix(a01, a11, fx), fy)
	return r, g, b, a
}

// perspectiveTransform applies 4-point perspective correction.
// pts: 4 corner points in TL, TR, BR, BL order.
// Returns a new image with the region corrected to a rectangle.
func perspectiveTransform(img image.Image, pts [4]image.Point, maxDim int) image.Image {
	widthTop := math.Hypot(float64(pts[1].X-pts[0].X), float64(pts[1].Y-pts[0].Y))
	widthBottom := math.Hypot(float64(pts[2].X-pts[3].X), float64(pts[2].Y-pts[3].Y))
	maxW := int(math.Max(widthTop, widthBottom))

	heightLeft := math.Hypot(float64(pts[3].X-pts[0].X), float64(pts[3].Y-pts[0].Y))
	heightRight := math.Hypot(float64(pts[2].X-pts[1].X), float64(pts[2].Y-pts[1].Y))
	maxH := int(math.Max(heightLeft, heightRight))

	if maxW < 10 || maxH < 10 {
		return nil
	}

	if maxDim > 0 && math.Max(float64(maxW), float64(maxH)) > float64(maxDim) {
		scale := float64(maxDim) / math.Max(float64(maxW), float64(maxH))
		maxW = int(float64(maxW) * scale)
		maxH = int(float64(maxH) * scale)
	}

	dst := image.NewRGBA(image.Rect(0, 0, maxW, maxH))

	for dy := 0; dy < maxH; dy++ {
		v := float64(dy) / float64(maxH-1)

		bx0 := float64(pts[0].X)*(1-v) + float64(pts[3].X)*v
		by0 := float64(pts[0].Y)*(1-v) + float64(pts[3].Y)*v
		bx1 := float64(pts[1].X)*(1-v) + float64(pts[2].X)*v
		by1 := float64(pts[1].Y)*(1-v) + float64(pts[2].Y)*v

		for dx := 0; dx < maxW; dx++ {
			u := float64(dx) / float64(maxW-1)

			sx := bx0*(1-u) + bx1*u
			sy := by0*(1-u) + by1*u

			r, g, b, a := bilinearSample(img, sx, sy)
			dst.Set(dx, dy, color.RGBA64{R: uint16(r), G: uint16(g), B: uint16(b), A: uint16(a)})
		}
	}

	return dst
}

// cropRegionFromImage decodes the image, applies perspective correction on the region,
// and returns a base64 JPEG data URL of the cropped region.
func cropRegionFromImage(imageBase64 string, points []struct{ X float64 `json:"x"`; Y float64 `json:"y"` }) (string, error) {
	img, _, err := decodeImage(imageBase64)
	if err != nil {
		return "", fmt.Errorf("decode image: %v", err)
	}

	if len(points) < 4 {
		return "", fmt.Errorf("need 4 points")
	}

	pts := [4]image.Point{
		{X: int(points[0].X), Y: int(points[0].Y)},
		{X: int(points[1].X), Y: int(points[1].Y)},
		{X: int(points[2].X), Y: int(points[2].Y)},
		{X: int(points[3].X), Y: int(points[3].Y)},
	}

	warped := perspectiveTransform(img, pts, 2000)
	if warped == nil {
		return "", fmt.Errorf("perspective transform failed")
	}

	return encodeImageJPEG(warped, 85), nil
}

// ==================== Image Compression ====================

// maxImageDim is the maximum dimension (width or height) for images sent to vision APIs.
const maxImageDim = 2000

// compressImage decodes the image, resizes if needed (max maxImageDim), and returns
// a JPEG base64 data URL. This prevents sending huge images (e.g. 3072x4096) that
// cause API errors or poor OCR quality.
func compressImage(imageBase64 string) (string, error) {
	img, _, err := decodeImage(imageBase64)
	if err != nil {
		return "", fmt.Errorf("decode: %v", err)
	}

	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	// No resize needed
	if w <= maxImageDim && h <= maxImageDim {
		return encodeImageJPEG(img, 85), nil
	}

	// Compute new dimensions keeping aspect ratio
	scale := float64(maxImageDim) / math.Max(float64(w), float64(h))
	newW := int(float64(w) * scale)
	newH := int(float64(h) * scale)

	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))

	// Simple nearest-neighbor resize (fast, good enough for OCR)
	for dy := 0; dy < newH; dy++ {
		sy := int(float64(dy) / float64(newH-1) * float64(h-1))
		for dx := 0; dx < newW; dx++ {
			sx := int(float64(dx) / float64(newW-1) * float64(w-1))
			dst.Set(dx, dy, img.At(sx, sy))
		}
	}

	return encodeImageJPEG(dst, 85), nil
}

// ==================== Vision API ====================

// stripDataURL removes the data URL prefix from a base64 string
func stripDataURL(b64Data string) string {
	if strings.HasPrefix(b64Data, "data:") {
		if idx := strings.Index(b64Data, ","); idx >= 0 {
			return b64Data[idx+1:]
		}
	}
	return b64Data
}

// detectMediaType detects image type from base64 data
func detectMediaType(b64Data string) string {
	if len(b64Data) < 8 {
		return "image/jpeg"
	}
	raw, err := base64.StdEncoding.DecodeString(b64Data[:16])
	if err != nil {
		raw, err = base64.RawStdEncoding.DecodeString(b64Data[:16])
		if err != nil {
			return "image/jpeg"
		}
	}
	if len(raw) > 2 && raw[0] == 0xFF && raw[1] == 0xD8 {
		return "image/jpeg"
	}
	if len(raw) > 4 && raw[1] == 'P' && raw[2] == 'N' && raw[3] == 'G' {
		return "image/png"
	}
	if len(raw) > 2 && raw[0] == 0x47 && raw[1] == 0x49 && raw[2] == 0x46 {
		return "image/gif"
	}
	if len(raw) > 3 && raw[0] == 0x52 && raw[1] == 0x49 && raw[2] == 0x46 && raw[3] == 0x46 {
		return "image/webp"
	}
	return "image/jpeg"
}


// isCJK reports whether r is a CJK character.
func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

// isSentenceEnd reports whether the line ends with sentence-ending punctuation.
func isSentenceEnd(s string) bool {
	runes := []rune(s)
	if len(runes) == 0 {
		return false
	}
	switch runes[len(runes)-1] {
	case '。', '！', '？', '.', '!', '?',
		'）', '」', '：', ';',
		')', '"':
		return true
	}
	return false
}

// isStructuralLine reports whether the line starts with a structural marker.
func isStructuralLine(line string) bool {
	line = strings.TrimLeft(line, " ")
	if strings.HasPrefix(line, "【") || strings.HasPrefix(line, "〔") ||
		strings.HasPrefix(line, "20") || strings.HasPrefix(line, "19") {
		return true
	}
	if len(line) > 0 && line[0] >= '0' && line[0] <= '9' {
		return true
	}
	return false
}

// mergeBrokenLines merges lines that were hard-wrapped by the model.
func mergeBrokenLines(s string) string {
	lines := strings.Split(s, "\n")
	var merged []string
	buf := ""
	for _, line := range lines {
		line = strings.TrimRight(line, " ")
		if line == "" {
			if buf != "" {
				merged = append(merged, buf)
				buf = ""
			}
			merged = append(merged, "")
			continue
		}
		if buf == "" {
			buf = line
			continue
		}
		if isSentenceEnd(buf) || isStructuralLine(line) {
			merged = append(merged, buf)
			buf = line
		} else {
			buf += strings.Join(strings.Fields(line), " ")
		}
	}
	if buf != "" {
		merged = append(merged, buf)
	}
	return strings.Join(merged, "\n")
}

// normalizePunctuation fixes mismatched punctuation around CJK characters.
// e.g. English comma between Chinese chars → Chinese comma.
func normalizePunctuation(s string) string {
	// Map of ASCII punctuation → full-width CJK punctuation
	asciiToFull := map[rune]rune{
		',': '，',
		'.': '。',
		':': '：',
		';': '；',
		'!': '！',
		'?': '？',
		'(': '（',
		')': '）',
	}

	runes := []rune(s)
	for i, r := range runes {
		replacement, ok := asciiToFull[r]
		if !ok {
			continue
		}
		prevIsCJK := i > 0 && isCJK(runes[i-1])
		nextIsCJK := i+1 < len(runes) && isCJK(runes[i+1])
		// Skip: digit after (, digit before ), or letter adjacent
		prevIsAlnum := i > 0 && (unicode.IsDigit(runes[i-1]) || unicode.IsLetter(runes[i-1]) && !isCJK(runes[i-1]))
		nextIsAlnum := i+1 < len(runes) && (unicode.IsDigit(runes[i+1]) || unicode.IsLetter(runes[i+1]) && !isCJK(runes[i+1]))

		// Normalize if between CJK characters, or between CJK and ASCII letter/digit
		// But skip parentheses that wrap pure English/number content like "(2024)" or "(eligible"
		if r == '(' && nextIsAlnum && !prevIsCJK {
			continue
		}
		if r == ')' && prevIsAlnum && !nextIsCJK {
			continue
		}
		// Period between digits: "21. D" → keep ASCII
		if r == '.' && i > 0 && unicode.IsDigit(runes[i-1]) {
			continue
		}

		if prevIsCJK || nextIsCJK {
			runes[i] = replacement
		}
	}
	return string(runes)
}

// callVisionAPI is the main entry point for vision/OCR API calls.
// It routes Claude models to the Anthropic-compatible API, and other models
// to the OpenAI-compatible API. All config (including API keys) comes from
// aiConfig which is populated from the database.
func callVisionAPI(imageBase64 string, aiConfig map[string]interface{}) (string, error) {
	model, _ := aiConfig["model"].(string)

	if model == "" {
		model = "glm-4v-flash"
	}

	// Compress image before sending to reduce payload size
	compressed, err := compressImage(imageBase64)
	if err != nil {
		return "", fmt.Errorf("compress image: %v", err)
	}

	origSize := len(stripDataURL(imageBase64))
	newSize := len(stripDataURL(compressed))
	fmt.Printf("[Vision] model=%s, orig_b64=%d, compressed_b64=%d\n", model, origSize, newSize)

	// Route Claude models to Anthropic-compatible endpoint
	if strings.HasPrefix(model, "claude") {
		return callClaudeVision(compressed, aiConfig)
	}

	// Route MiniMax models to their custom coding_plan/vlm endpoint
	if strings.HasPrefix(model, "minimax") {
		return callMiniMaxVision(compressed, aiConfig)
	}

	// OpenAI-compatible API call
	apiURL, _ := aiConfig["api_url"].(string)
	apiKey, _ := aiConfig["api_key"].(string)

	if apiURL == "" {
		apiURL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
	}

	fmt.Printf("[Vision] OpenAI format: url=%s, has_key=%v\n", apiURL, apiKey != "")

	b64Data := stripDataURL(compressed)
	ct := detectMediaType(b64Data)
	dataURL := "data:" + ct + ";base64," + b64Data

	reqBody := map[string]interface{}{
		"model": model,
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]interface{}{
					{"type": "text", "text": "请识别图片中的所有文字内容，保持原始格式和排版。注意标点符号与原文一致：英文内容使用英文标点，中文内容使用中文标点。直接输出识别结果，不要加任何前缀或说明。"},
					{"type": "image_url", "image_url": map[string]string{"url": dataURL}},
				},
			},
		},
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", apiURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := newHTTPClient(120 * time.Second)
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API call failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				if text, ok := msg["content"].(string); ok {
					return text, nil
				}
			}
		}
	}

	if errMsg, ok := result["error"].(map[string]interface{}); ok {
		msg, _ := errMsg["message"].(string)
		return "", fmt.Errorf("API error: %s", msg)
	}

	return "", fmt.Errorf("unexpected response format")
}

// callClaudeVision calls the Anthropic-compatible API (e.g. Zhipu's proxy).
// API key and URL are loaded from aiConfig (populated from database).
// Falls back to env var ANTHROPIC_AUTH_TOKEN only if not in config.
func callClaudeVision(imageBase64 string, aiConfig map[string]interface{}) (string, error) {
	b64Data := stripDataURL(imageBase64)
	mediaType := detectMediaType(b64Data)

	apiURL, _ := aiConfig["claude_api_url"].(string)
	apiKey, _ := aiConfig["claude_api_key"].(string)
	fmt.Printf("[Vision] Claude format: url=%s, has_key=%v\n", apiURL, apiKey != "")

	// Map friendly model names to Anthropic model IDs
	model, _ := aiConfig["model"].(string)
	claudeModel := "claude-haiku-4-5-20251001"
	switch model {
	case "claude-sonnet", "sonnet":
		claudeModel = "claude-sonnet-4-20250514"
	case "claude-opus", "opus":
		claudeModel = "claude-opus-4-20250514"
	}

	// Fallback to env var

	if apiURL == "" {
		apiURL = os.Getenv("ANTHROPIC_BASE_URL")
		if apiURL != "" {
			apiURL = apiURL + "/v1/messages"
		}
	}
	if apiURL == "" {
		apiURL = "https://open.bigmodel.cn/api/anthropic/v1/messages"
	}

	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_AUTH_TOKEN")
	}

	if apiKey == "" {
		return "", fmt.Errorf("Claude API key not configured. Set claude_api_key in settings or ANTHROPIC_AUTH_TOKEN env var")
	}

	reqBody := map[string]interface{}{
		"model":      claudeModel,
		"max_tokens": 4096,
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]interface{}{
					{"type": "text", "text": "请识别图片中的所有文字内容，保持原始格式和排版。直接输出识别结果，注意标点符号与原文一致：英文内容使用英文标点，中文内容使用中文标点。直接输出识别结果，不要加任何前缀、说明或markdown格式。"},
					{"type": "image", "source": map[string]string{
						"type":       "base64",
						"media_type": mediaType,
						"data":       b64Data,
					}},
				},
			},
		},
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", apiURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := newHTTPClient(120 * time.Second)
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API call failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	if content, ok := result["content"].([]interface{}); ok && len(content) > 0 {
		if block, ok := content[0].(map[string]interface{}); ok {
			if text, ok := block["text"].(string); ok {
				return normalizePunctuation(strings.TrimSpace(text)), nil
			}
		}
	}

	if errMsg, ok := result["error"].(map[string]interface{}); ok {
		msg, _ := errMsg["message"].(string)
		return "", fmt.Errorf("API error: %s", msg)
	}

	return "", fmt.Errorf("unexpected response: %s", string(respBody[:min(len(respBody), 200)]))
}


// callMiniMaxVision calls MiniMax's custom VLM endpoint.
// API format: POST {prompt, image_url} -> {content}
func callMiniMaxVision(imageBase64 string, aiConfig map[string]interface{}) (string, error) {
	apiURL, _ := aiConfig["api_url"].(string)
	apiKey, _ := aiConfig["api_key"].(string)

	if apiURL == "" {
		apiURL = "https://api.minimaxi.com"
	}
	apiURL = strings.TrimRight(apiURL, "/") + "/v1/coding_plan/vlm"

	fmt.Printf("[Vision] MiniMax format: url=%s, has_key=%v\n", apiURL, apiKey != "")

	b64Data := stripDataURL(imageBase64)
	ct := detectMediaType(b64Data)
	dataURL := "data:" + ct + ";base64," + b64Data

	reqBody := map[string]interface{}{
		"prompt":    "请识别图片中的所有文字内容，保持原始格式和排版。注意标点符号与原文一致：英文内容使用英文标点，中文内容使用中文标点。直接输出识别结果，不要加任何前缀或说明。",
		"image_url": dataURL,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", apiURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := newHTTPClient(120 * time.Second)
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API call failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	if text, ok := result["content"].(string); ok {
		return normalizePunctuation(mergeBrokenLines(text)), nil
	}

	if errMsg, ok := result["error"].(map[string]interface{}); ok {
		msg, _ := errMsg["message"].(string)
		return "", fmt.Errorf("API error: %s", msg)
	}

	return "", fmt.Errorf("unexpected response: %s", string(respBody[:min(len(respBody), 200)]))
}
