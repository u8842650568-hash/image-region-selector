package main

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"

	_ "github.com/mattn/go-sqlite3"
	"github.com/xuri/excelize/v2"
	_ "golang.org/x/image/webp"
)


// ==================== Thumbnail ====================

const thumbMaxSize = 400

func makeThumbnail(data []byte, ct string) string {
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return ""
	}
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= thumbMaxSize && h <= thumbMaxSize {
		return ""
	}
	ratio := float64(thumbMaxSize) / float64(max(w, h))
	nw := int(float64(w) * ratio)
	nh := int(float64(h) * ratio)
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	// Bilinear resize
	for y := 0; y < nh; y++ {
		fy := float64(y) / ratio
		y0 := int(fy)
		y1 := y0 + 1
		if y1 >= h {
			y1 = h - 1
		}
		fyF := fy - float64(y0)
		for x := 0; x < nw; x++ {
			fx := float64(x) / ratio
			x0 := int(fx)
			x1 := x0 + 1
			if x1 >= w {
				x1 = w - 1
			}
			fxF := fx - float64(x0)
			r00, g00, b00, a00 := src.At(x0, y0).RGBA()
			r10, g10, b10, a10 := src.At(x1, y0).RGBA()
			r01, g01, b01, a01 := src.At(x0, y1).RGBA()
			r11, g11, b11, a11 := src.At(x1, y1).RGBA()
			interpolate := func(c00, c10, c01, c11 uint32) uint32 {
				return uint32(float64(c00)*(1-fxF)*(1-fyF) + float64(c10)*fxF*(1-fyF) + float64(c01)*(1-fxF)*fyF + float64(c11)*fxF*fyF)
			}
			dst.Set(x, y, color.RGBA64{
				R: uint16(interpolate(r00, r10, r01, r11)),
				G: uint16(interpolate(g00, g10, g01, g11)),
				B: uint16(interpolate(b00, b10, b01, b11)),
				A: uint16(interpolate(a00, a10, a01, a11)),
			})
		}
	}
	var buf bytes.Buffer
	switch ct {
	case "image/png":
		png.Encode(&buf, dst)
	default:
		jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 80})
	}
	return "data:" + ct + ";base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// ==================== Data Model ====================

type ImageItem struct {
	ID         int    `json:"id"`
	Name       string `json:"name"`
	Data       string `json:"data"`       // base64 data URL
	Thumb      string `json:"thumb"`      // small thumbnail base64
	Text       string `json:"text"`       // text recognition result
	Status     string `json:"status"`     // recognition status
	Order      int    `json:"order"`
	Regions    string `json:"regions,omitempty"`
}

var (
	images  []ImageItem
	nextID  int
	mu      sync.Mutex
	db      *sql.DB
)

// ==================== SQLite ====================

func initDB() {
	var err error
	db, err = sql.Open("sqlite3", "/root/Desktop/image-region-selector/multi_ocr.db")
	if err != nil {
		log.Fatal("DB open:", err)
	}
	db.Exec(`CREATE TABLE IF NOT EXISTS images (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		data TEXT NOT NULL,
		text TEXT DEFAULT '',
		status TEXT DEFAULT 'ready',
		table_text TEXT DEFAULT '',
		table_status TEXT DEFAULT 'ready',
		recog_type TEXT DEFAULT 'text',
		sort_order INTEGER DEFAULT 0
	)`)
	db.Exec("ALTER TABLE images ADD COLUMN regions TEXT DEFAULT ''")
	loadFromDB()
}

func loadFromDB() {
	rows, err := db.Query("SELECT id, name, data, text, status, sort_order, regions FROM images ORDER BY sort_order")
	if err != nil {
		log.Println("DB load:", err)
		return
	}
	defer rows.Close()
	images = []ImageItem{}
	maxID := 0
	for rows.Next() {
		var item ImageItem
		rows.Scan(&item.ID, &item.Name, &item.Data, &item.Text, &item.Status, &item.Order, &item.Regions)
		item.Thumb = item.Data
		images = append(images, item)
		if item.ID > maxID {
			maxID = item.ID
		}
	}
	nextID = maxID
	log.Printf("Loaded %d images from DB", len(images))
}

func dbInsert(item ImageItem) {
	db.Exec("INSERT INTO images (name, data, text, status, sort_order) VALUES (?, ?, ?, ?, ?)",
		item.Name, item.Data, item.Text, item.Status, item.Order)
	var id int
	db.QueryRow("SELECT last_insert_rowid()").Scan(&id)
	item.ID = id
}

func dbUpdateText(id int, text string) {
	db.Exec("UPDATE images SET text = ? WHERE id = ?", text, id)
}
func dbUpdateRegions(id int, regions string) {
	db.Exec("UPDATE images SET regions = ? WHERE id = ?", regions, id)
}
func dbUpdateStatus(id int, status string) {
	db.Exec("UPDATE images SET status = ? WHERE id = ?", status, id)
}

func dbDelete(id int) {
	db.Exec("DELETE FROM images WHERE id = ?", id)
}

func dbReorder(items []ImageItem) {
	for _, item := range items {
		db.Exec("UPDATE images SET sort_order = ? WHERE id = ?", item.Order, item.ID)
	}
}

// ==================== Handlers ====================

func handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(htmlPage))
}

func handleUpload(w http.ResponseWriter, r *http.Request) {
	file, header, err := r.FormFile("image")
	if err != nil {
		jsonErr(w, "上传失败: "+err.Error())
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		jsonErr(w, "读取失败: "+err.Error())
		return
	}

	b64 := base64.StdEncoding.EncodeToString(data)
	ct := http.DetectContentType(data)
	dataURL := "data:" + ct + ";base64," + b64

	thumb := makeThumbnail(data, ct)
	if thumb == "" {
		thumb = dataURL
	}

	mu.Lock()
	nextID++
	item := ImageItem{
		ID:        nextID,
		Name:      header.Filename,
		Data:      dataURL,
		Thumb:     thumb,
		Status:    "ready",
		Order:     len(images),
	}
	images = append(images, item)
	dbInsert(item)
	item.ID = nextID // dbInsert sets actual ID
	// Re-read to get the real auto-increment ID
	var realID int
	db.QueryRow("SELECT last_insert_rowid()").Scan(&realID)
	if realID > nextID {
		nextID = realID
	}
	mu.Unlock()

	jsonOK(w, item)
}

func handleListImages(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	jsonOK(w, images)
}

func handleReorder(w http.ResponseWriter, r *http.Request) {
	var req struct {
		IDs []int `json:"ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonErr(w, err.Error())
		return
	}

	mu.Lock()
	defer mu.Unlock()

	idMap := make(map[int]*ImageItem)
	for i := range images {
		idMap[images[i].ID] = &images[i]
	}
	newOrder := make([]ImageItem, 0, len(req.IDs))
	for i, id := range req.IDs {
		if item, ok := idMap[id]; ok {
			item.Order = i
			newOrder = append(newOrder, *item)
		}
	}
	images = newOrder
	dbReorder(images)
	jsonOK(w, map[string]string{"ok": "1"})
}

func handleDeleteImage(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/images/")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		jsonErr(w, "invalid id")
		return
	}

	mu.Lock()
	defer mu.Unlock()

	found := -1
	for i, item := range images {
		if item.ID == id {
			found = i
			break
		}
	}
	if found < 0 {
		jsonErr(w, "not found")
		return
	}
	images = append(images[:found], images[found+1:]...)
	for i := range images {
		images[i].Order = i
	}
	dbDelete(id)
	dbReorder(images)
	jsonOK(w, map[string]string{"ok": "1"})
}

func handleClearImages(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	images = []ImageItem{}
	mu.Unlock()
	db.Exec("DELETE FROM images")
	jsonOK(w, map[string]string{"ok": "1"})
}

func handleOCR(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/ocr/")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		jsonErr(w, "invalid id")
		return
	}

	var req struct {
		ImageBase64 string                 `json:"image_base64"`
		AIConfig    map[string]interface{} `json:"ai_config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonErr(w, err.Error())
		return
	}

	aiConfig := req.AIConfig
	if aiConfig == nil {
		cfg := loadServerConfig()
		aiConfig = map[string]interface{}{
			"model":          cfg["model"],
			"api_url":        cfg["api_url"],
			"api_key":        cfg["api_key"],

		}
	}

	mu.Lock()
	for i := range images {
		if images[i].ID == id {
			images[i].Status = "recognizing"
		}
	}
	mu.Unlock()
	dbUpdateStatus(id, "recognizing")

	text, err := callVisionAPI(req.ImageBase64, aiConfig)
	if err != nil {
		mu.Lock()
		for i := range images {
			if images[i].ID == id {
				images[i].Status = "error"
			}
		}
		mu.Unlock()
		dbUpdateStatus(id, "error")
		jsonErr(w, "识别失败: "+err.Error())
		return
	}

	mu.Lock()
	for i := range images {
		if images[i].ID == id {
			images[i].Text = text
			images[i].Status = "done"
		}
	}
	mu.Unlock()
	dbUpdateText(id, text)
	dbUpdateStatus(id, "done")

	jsonOK(w, map[string]interface{}{"text": text, "status": "done"})
}

func handleOCRRegion(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ImageBase64 string                 `json:"image_base64"`
		Regions     []struct {
			Points []struct {
				X float64 `json:"x"`
				Y float64 `json:"y"`
			} `json:"points"`
		} `json:"regions"`
		AIConfig  map[string]interface{} `json:"ai_config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonErr(w, err.Error())
		return
	}

	aiConfig := req.AIConfig
	if aiConfig == nil {
		cfg := loadServerConfig()
		aiConfig = map[string]interface{}{
			"model":          cfg["model"],
			"api_url":        cfg["api_url"],
			"api_key":        cfg["api_key"],

		}
	}

	var allTexts []string
	for i, region := range req.Regions {
		pts := region.Points
		if len(pts) < 4 {
			continue
		}
		croppedBase64, err := cropRegionFromImage(req.ImageBase64, pts)
		if err != nil {
			log.Printf("crop region %d: %v", i, err)
			allTexts = append(allTexts, fmt.Sprintf("[区域%d: 裁剪失败]", i+1))
			continue
		}
		text, err := callVisionAPI(croppedBase64, aiConfig)
		if err != nil {
			allTexts = append(allTexts, fmt.Sprintf("[区域%d: 识别失败]", i+1))
			continue
		}
		allTexts = append(allTexts, text)
	}

	jsonOK(w, map[string]interface{}{"text": strings.Join(allTexts, "\n\n")})
}

func handleSaveText(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/text/")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		jsonErr(w, "invalid id")
		return
	}

	var req struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonErr(w, err.Error())
		return
	}

	mu.Lock()
	for i := range images {
		if images[i].ID == id {
			images[i].Text = req.Text
		}
	}
	mu.Unlock()
	dbUpdateText(id, req.Text)

	jsonOK(w, map[string]string{"ok": "1"})
}

func handleSaveRegions(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/regions/")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		jsonErr(w, "invalid id")
		return
	}
	var req struct {
		Regions string `json:"regions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonErr(w, err.Error())
		return
	}
	mu.Lock()
	for i := range images {
		if images[i].ID == id {
			images[i].Regions = req.Regions
		}
	}
	mu.Unlock()
	dbUpdateRegions(id, req.Regions)
	jsonOK(w, map[string]string{"ok": "1"})
}

func handleExport(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	var buf strings.Builder
	for i, item := range images {
		if i > 0 {
			buf.WriteString("\n\n" + strings.Repeat("=", 60) + "\n\n")
		}
		buf.WriteString(item.Text)
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Write([]byte(buf.String()))
}





func handleExportXlsx(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	f := excelize.NewFile()
	defaultSheet := f.GetSheetName(0)
	for i, item := range images {
		sheetName := fmt.Sprintf("图片%d", i+1)
		if i == 0 {
			f.SetSheetName(defaultSheet, sheetName)
		} else {
			f.NewSheet(sheetName)
		}
		lines := strings.Split(item.Text, "\n")
		for li, line := range lines {
			f.SetCellValue(sheetName, fmt.Sprintf("A%d", li+1), line)
		}
	}

	w.Header().Set("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
	w.Header().Set("Content-Disposition", "attachment; filename=识别结果.xlsx")
	f.Write(w)
}

// ==================== JSON Helpers ====================

func jsonOK(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(data)
}

func jsonErr(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(400)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// ==================== HTML Frontend ====================

const htmlPage = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>多图识别</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"WenQuanYi Micro Hei","Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#f5f5f5;height:100vh;display:flex;flex-direction:column;overflow:hidden;color:#333}

/* Toolbar */
.toolbar{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#fff;border-bottom:1px solid #e0e0e0;flex-shrink:0}
.btn{padding:7px 16px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;font-size:13px;font-family:inherit;background:#fff;transition:all .15s;display:inline-flex;align-items:center;gap:5px}
.btn:hover{border-color:#4096ff;color:#4096ff}
.btn-primary{background:#4096ff;color:#fff;border-color:#4096ff}
.btn-primary:hover{background:#1677ff;border-color:#1677ff;color:#fff}
.btn-danger{color:#ff4d4f;border-color:#ff4d4f}
.btn-danger:hover{background:#ff4d4f;color:#fff}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-icon{width:32px;height:32px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;font-size:16px;display:inline-flex;align-items:center;justify-content:center;background:#fff;transition:all .15s;padding:0}
.btn-icon:hover{border-color:#4096ff;color:#4096ff}
.toolbar-right{margin-left:auto;display:flex;align-items:center;gap:8px}

/* Main layout */
.main{display:flex;flex:1;overflow:hidden}

/* Left sidebar */
.sidebar{width:220px;background:#fff;border-right:1px solid #e0e0e0;display:flex;flex-direction:column;flex-shrink:0}
.sidebar-header{padding:12px;font-size:13px;color:#999;border-bottom:1px solid #f0f0f0;display:flex;justify-content:space-between;align-items:center}
.sidebar-list{flex:1;overflow-y:auto;padding:8px}
.sidebar-list::-webkit-scrollbar{width:4px}
.sidebar-list::-webkit-scrollbar-thumb{background:#ddd;border-radius:2px}
.img-card{position:relative;margin-bottom:8px;border:2px solid transparent;border-radius:8px;transition:transform .3s cubic-bezier(.22,1,.36,1), border-color .15s, box-shadow .15s, background .15s;background:#fafafa;will-change:transform;display:flex;align-items:stretch}
.img-card:hover{border-color:#4096ff;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.img-card.active{border-color:#4096ff;background:#e6f4ff}
.img-card.dragging{box-shadow:0 12px 32px rgba(0,0,0,.22);z-index:100;border-color:#1677ff;background:#fff;transition:none;transform-origin:center center}
.img-card.dragging .thumb-wrap{display:none}
.drag-handle{width:20px;min-height:100%;display:flex;align-items:center;justify-content:center;cursor:grab;background:#f0f0f0;border-radius:6px 0 0 6px;flex-shrink:0;touch-action:none;transition:background .15s}
.drag-handle:hover{background:#e4e4e4}
.drag-handle:active{cursor:grabbing}
.card-body{flex:1;min-width:0;position:relative}
.drop-preview{position:absolute;left:4px;right:4px;height:0;border:2px solid #1677ff;border-radius:6px;box-shadow:0 0 8px rgba(22,119,255,.3);z-index:50;pointer-events:none;opacity:0;transition:top .2s cubic-bezier(.22,1,.36,1), opacity .15s}
.img-card .thumb{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:6px;display:block;background:#f0f0f0}
.img-card .order-badge{position:absolute;top:4px;left:4px;min-width:20px;height:20px;border-radius:10px;background:rgba(0,0,0,.55);color:#fff;font-size:11px;font-weight:bold;text-align:center;line-height:20px;padding:0 5px;pointer-events:none}
.img-card .info{padding:6px 8px;font-size:12px;color:#666;display:flex;justify-content:space-between;align-items:center}
.img-card .info .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:120px}
.img-card .status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.status-dot.ready{background:#d9d9d9}
.status-dot.recognizing{background:#fa8c16;animation:pulse 1s infinite}
.status-dot.done{background:#52c41a}
.status-dot.error{background:#ff4d4f}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.img-card .del-btn{position:absolute;top:4px;right:4px;width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,.5);color:#fff;border:none;cursor:pointer;font-size:14px;line-height:22px;text-align:center;display:none;align-items:center;justify-content:center}
.img-card:hover .del-btn{display:flex}
.img-card .ocr-btn{position:absolute;bottom:40px;left:50%;transform:translateX(-50%);padding:3px 12px;border-radius:4px;background:rgba(0,0,0,.6);color:#fff;border:none;cursor:pointer;font-size:12px;display:none}
.img-card:hover .ocr-btn{display:block}

.add-card{border:2px dashed #d9d9d9;border-radius:8px;padding:20px;text-align:center;cursor:pointer;color:#999;font-size:13px;transition:all .15s;margin-top:4px}
.add-card:hover{border-color:#4096ff;color:#4096ff}

/* Right content area */
.content{flex:1;display:flex;flex-direction:column;overflow:hidden}
.content-header{padding:12px 16px;border-bottom:1px solid #f0f0f0;font-size:13px;color:#999;background:#fff}
.text-list{flex:1;overflow-y:auto;padding:16px}
.text-list::-webkit-scrollbar{width:6px}
.text-list::-webkit-scrollbar-thumb{background:#ddd;border-radius:3px}
.text-block{background:#fff;margin-bottom:0;border-bottom:1px dashed #d9d9d9;transition:all .15s;overflow:hidden}
.text-block:last-child{border-bottom:none}
.text-block:hover{background:#fafafa}
.text-block-header{display:flex;align-items:center;gap:8px;padding:8px 12px;font-size:12px;color:#999}
.text-block-header img{width:32px;height:32px;object-fit:cover;border-radius:4px}
.text-block-header .idx{font-weight:bold;color:#333;font-size:13px}
.text-block-header .status-tag{margin-left:auto;padding:2px 8px;border-radius:4px;font-size:11px}
.status-tag.done{background:#f6ffed;color:#52c41a}
.status-tag.recognizing{background:#fff7e6;color:#fa8c16}
.status-tag.error{background:#fff2f0;color:#ff4d4f}
.status-tag.ready{background:#f0f0f0;color:#999}
.text-content{padding:12px 16px;min-height:60px;outline:none;font-size:14px;line-height:1.8;white-space:pre-wrap;word-break:break-word;color:#333}
.text-content:empty:before{content:attr(data-placeholder);color:#ccc}
.text-content:focus{background:#fafafa}

.empty-hint{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#ccc}
.empty-hint .icon{font-size:48px;margin-bottom:16px}
.empty-hint p{font-size:14px}

/* Region select modal */
.modal-overlay{display:none;position:fixed;inset:0;background:#1a1a2e;z-index:1000;align-items:center;justify-content:center}
.modal-overlay.show{display:flex}
.modal{background:#fff;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.15);width:90vw;height:85vh;display:flex;flex-direction:column;overflow:hidden}
.modal-overlay#regionModal.show .modal{width:100vw;height:100vh;border-radius:0}
.modal-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #e8e8e8}
.modal-header h3{font-size:15px}
.modal-header .close-btn{width:30px;height:30px;border:none;background:none;font-size:20px;cursor:pointer;border-radius:4px}
.modal-header .close-btn:hover{background:#f0f0f0}
.modal-body{flex:1;display:flex;overflow:hidden}
.modal-canvas-wrap{flex:1;position:relative;background:#1a1a2e;overflow:hidden}
.modal-canvas-wrap canvas{cursor:crosshair;position:relative;z-index:1}
.modal-hint{position:absolute;top:8px;left:12px;background:rgba(0,0,0,.55);color:#fff;padding:4px 14px;border-radius:12px;font-size:11px;pointer-events:none;z-index:5}
.modal-footer{display:flex;justify-content:flex-end;gap:8px;padding:12px 16px;border-top:1px solid #e8e8e8}
.modal-actions{display:flex;gap:8px;margin-left:auto}
.config-form{padding:20px 24px;flex:1;overflow-y:auto}
.config-form label{display:block;font-size:13px;color:#666;margin-bottom:4px;margin-top:16px}
.config-form label:first-child{margin-top:0}
.config-form input{width:100%;padding:8px 12px;border:1px solid #d9d9d9;border-radius:6px;font-size:13px;font-family:inherit;outline:none;transition:border-color .15s}
.config-form input:focus{border-color:#4096ff}
.config-form .hint{font-size:11px;color:#999;margin-top:4px}
.preset-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.preset-tag{padding:4px 12px;border:1px solid #d9d9d9;border-radius:14px;font-size:12px;cursor:pointer;transition:all .15s;color:#666;background:#fff}
.preset-tag:hover{border-color:#4096ff;color:#4096ff}
.preset-tag.active{background:#4096ff;color:#fff;border-color:#4096ff}
</style>
</head>
<body>

<div class="toolbar">
  <button class="btn" id="btnPreview" onclick="previewText()">预览文本</button>
  <button class="btn" id="btnExport" onclick="exportTxt()">导出txt</button>
  <button class="btn btn-danger" onclick="clearAll()">清空</button>
  <input type="file" id="fileInput" accept="image/*" multiple style="display:none" onchange="handleFiles(event)">
  <div class="toolbar-right">
    <button class="btn-icon" onclick="openConfigModal()" title="AI识别配置">&#9881;</button>
  </div>
</div>

<div class="main">
  <!-- Left sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <span>图片列表 (<span id="imgCount">0</span>)</span>
    </div>
    <div class="sidebar-list" id="sidebarList">
      <div class="add-card" onclick="addImages()">+ 添加图片</div>
    </div>
  </div>

  <!-- Right content -->
  <div class="content">
    <div class="content-header" id="contentHeader">上传图片后点击"开始识别"</div>
    <div class="text-list" id="textList">
      <div class="empty-hint" id="emptyHint">
        <div class="icon">&#128196;</div>
        <p>上传图片开始识别</p>
      </div>
    </div>
  </div>
</div>

<!-- Region select modal -->
<div class="modal-overlay" id="regionModal">
  <div class="modal">
    <div class="modal-header">
      <h3>框选识别区域</h3>
      <button class="close-btn" onclick="closeRegionModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="modal-canvas-wrap" id="modalCanvasWrap">
        <div class="modal-hint">拖拽框选识别区域，右键删除，滚轮缩放，空格+拖拽平移</div>
        <canvas id="modalCanvas"></canvas>
      </div>
    </div>
    <div class="modal-footer">
      <div class="modal-actions">
        <button class="btn" onclick="closeRegionModal()">取消</button>
        <button class="btn btn-primary" onclick="confirmRegionOCR()" id="btnConfirmRegion">识别选定区域</button>
      </div>
    </div>
  </div>
</div>

<!-- Config modal -->
<div class="modal-overlay" id="configModal">
  <div class="modal" style="width:520px;height:auto;max-height:80vh">
    <div class="modal-header">
      <h3>AI识别配置</h3>
      <button class="close-btn" onclick="closeConfigModal()">&times;</button>
    </div>
    <div class="config-form">
      <label>API 地址</label>
      <input type="text" id="cfgUrl" placeholder="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions">
      <div class="hint">AI 识别 API 地址（MiniMax VLM 仅需填写 API Host）</div>

      <label>模型名称</label>
      <input type="text" id="cfgModel" placeholder="glm-4v-flash">
      <div class="preset-list" id="presetList"></div>

      <label>API 密钥</label>
      <input type="password" id="cfgKey" placeholder="留空使用默认密钥">
    </div>
    <div class="modal-footer">
      <div class="modal-actions">
        <button class="btn" onclick="closeConfigModal()">取消</button>
        <button class="btn btn-primary" onclick="saveConfig()">保存</button>
      </div>
    </div>
  </div>
</div>

<script>
// ========== State ==========
let imgItems = [];
let activeImgId = null;
let ocrRunning = false;
let ocrMode = 'ai';
// ========== Field helpers ==========
function getText(item) { return item.text || ''; }
function setText(item, val) { item.text = val; }
function getStatus(item) { return item.status || 'ready'; }
function setStatus(item, val) { item.status = val; }


// Region modal state
let regionImgId = null;
let regionImg = null;
let regionCanvas, regionCtx;
let regionView = {x:0,y:0,scale:1,base:1};
let regions = [];
let regionDrag = {mode:'none', regionIdx:-1, cornerIdx:-1};
let regionSpace = false;



// ========== API ==========
async function api(path, opts={}) {
  const resp = await fetch(path, opts);
  return resp.json();
}

// ========== Init ==========
function genThumb(dataURL) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const maxSz = 400;
      if (img.width <= maxSz && img.height <= maxSz) { resolve(dataURL); return; }
      const ratio = Math.min(maxSz/img.width, maxSz/img.height);
      const c = document.createElement('canvas');
      c.width = Math.round(img.width*ratio);
      c.height = Math.round(img.height*ratio);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL('image/jpeg', 0.8));
    };
    img.onerror = () => resolve(dataURL);
    img.src = dataURL;
  });
}

async function loadImages() {
  try {
    const data = await api('/api/images');
    if (data && !data.error) {
      imgItems = (data || []).map(item => {
        if (item.regions && typeof item.regions === 'string') item.regions = JSON.parse(item.regions);
        return item;
      });
      activeImgId = imgItems.length > 0 ? imgItems[0].id : null;
      render();
      // Async generate thumbnails for items that use full data as thumb
      for (const item of imgItems) {
        if (item.thumb === item.data && item.data.length > 50000) {
          genThumb(item.data).then(thumb => { item.thumb = thumb; renderSidebar(); });
        }
      }
    }
  } catch(e) {}
}
loadImages();

// ========== Upload ==========
function addImages() {
  document.getElementById('fileInput').click();
}

async function handleFiles(e) {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  e.target.value = '';

  for (const file of files) {
    const fd = new FormData();
    fd.append('image', file);
    try {
      const item = await api('/api/upload', {method:'POST', body:fd});
      if (item && item.id) {
        imgItems.push(item);
        render();
      }
    } catch(err) {
      console.error('Upload error:', err);
    }
  }
}

async function autoOCR(id, dataURL) {
  const item = imgItems.find(i => i.id === id);
  if (!item || item.aborted) return;
  updateItemStatus(id, 'recognizing');
  render();
  try {
    const result = await api('/api/ocr/' + id, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({image_base64: dataURL, ...getAIConfigPayload()})
    });
    if (item.aborted) return;
    if (result && !result.error) {
      setText(item, result.text || '');
      setStatus(item, 'done');
    } else {
      updateItemStatus(id, 'error');
    }
  } catch(e) {
    if (!item.aborted) updateItemStatus(id, 'error');
  }
  render();
}

function updateItemStatus(id, status) {
  for (let i = 0; i < imgItems.length; i++) {
    if (imgItems[i].id === id) {
      setStatus(imgItems[i], status);
      break;
    }
  }
}

// ========== Render ==========
function render() {
  renderSidebar();
  renderTextList();
  document.getElementById('imgCount').textContent = imgItems.length;
}

function renderSidebar() {
  const list = document.getElementById('sidebarList');
  let html = '';
  imgItems.forEach((item, idx) => {
    const cls = item.id === activeImgId ? 'img-card active' : 'img-card';
    const dotCls = getStatus(item);
    const shortName = item.name.length > 15 ? item.name.substring(0,12)+'...' : item.name;
    html += '<div class="'+cls+'" data-id="'+item.id+'" data-idx="'+idx+'"'
      + ' onclick="selectImage('+item.id+')">'
      + '<div class="drag-handle" onpointerdown="onPointerDown(event)">'
      + '<svg viewBox="0 0 10 20" width="10" height="20"><circle cx="3" cy="4" r="1.5" fill="#bbb"/><circle cx="3" cy="10" r="1.5" fill="#bbb"/><circle cx="3" cy="16" r="1.5" fill="#bbb"/><circle cx="7" cy="4" r="1.5" fill="#bbb"/><circle cx="7" cy="10" r="1.5" fill="#bbb"/><circle cx="7" cy="16" r="1.5" fill="#bbb"/></svg>'
      + '</div>'
      + '<div class="card-body"><div class="thumb-wrap" style="position:relative"><img class="thumb" src="'+item.thumb+'" alt="'+item.name+'">'
      + '<span class="order-badge">'+(idx+1)+'</span></div>'
      + '<div class="info"><span class="name">'+shortName+'</span>'
      + '<span class="status-dot '+dotCls+'"></span></div>'
      + '<button class="del-btn" onclick="event.stopPropagation();deleteImage('+item.id+')">&times;</button>'
      + (getStatus(item) === 'recognizing'
        ? '<button class="ocr-btn" style="bottom:64px;background:rgba(255,0,0,.7)" onclick="event.stopPropagation();stopOCR('+item.id+')">停止</button>'
        : '<button class="ocr-btn" style="bottom:64px" onclick="event.stopPropagation();reOCR('+item.id+')">'+(getStatus(item)==='done'?'重新识别':'开始识别')+'</button>')
      + '<button class="ocr-btn" onclick="event.stopPropagation();openRegionModal('+item.id+')">框选识别</button>'
      + '</div></div>';
  });
  html += '<div class="add-card" onclick="addImages()">+ 添加图片</div>';
  list.innerHTML = html;
}

function renderTextList() {
  const list = document.getElementById('textList');
  if (imgItems.length === 0) {
    list.innerHTML = '<div class="empty-hint"><div class="icon">&#128196;</div><p>上传图片后点击"开始识别"</p></div>';
    return;
  }
  let html = '';
  imgItems.forEach((item, idx) => {
    const status = getStatus(item);
    const statusTag = getStatusTag(status);
    const contentHtml = escapeHtml(item.text || '');
    const contentClass = 'text-content';
    const editable = ' contenteditable="true" onblur="saveText('+item.id+', this.textContent)"';
    const placeholder = '等待识别...';
    const editBtn = '';
    const ocrBtn = (status === 'ready')
      ? '<button style="margin-top:8px;padding:6px 16px;border:1px solid #1890ff;background:#1890ff;color:#fff;border-radius:4px;cursor:pointer;font-size:13px" onclick="startOcr('+item.id+')">开始识别</button>'
      : '';
    html += '<div class="text-block" id="textBlock-'+item.id+'">'
      + '<div class="text-block-header">'
      + '<img src="'+item.data+'" alt="">'
      + '<span class="idx">图片 '+(idx+1)+'</span>'
      + '<span>'+item.name+'</span>'
      + statusTag + editBtn
      + '</div>'
      + '<div class="'+contentClass+'" data-id="'+item.id+'"'
      + ' data-placeholder="'+placeholder+'"'+editable+'>'+contentHtml+'</div>'
      + ocrBtn
      + '</div>';
  });
  list.innerHTML = html;
}



function cancelEditMarkdown(id) { renderTextList(); }

function getStatusTag(status) {
  const map = {
    'ready': '<span class="status-tag ready">待识别</span>',
    'recognizing': '<span class="status-tag recognizing">识别中...</span>',
    'done': '<span class="status-tag done">已识别</span>',
    'error': '<span class="status-tag error">识别失败</span>',
  };
  return map[status] || '<span class="status-tag ready">待识别</span>';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ========== Image Selection ==========
function selectImage(id) {
  activeImgId = id;
  renderSidebar();
  const el = document.getElementById('textBlock-'+id);
  if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
}

// ========== Stop single OCR ==========
function stopOCR(id) {
  const item = imgItems.find(i => i.id === id);
  if (item) {
    item.aborted = true;
    updateItemStatus(id, 'ready');
    render();
  }
}

// ========== Delete ==========
async function deleteImage(id) {
  if (!confirm('确定删除这张图片？')) return;
  await api('/api/images/'+id, {method:'DELETE'});
  imgItems = imgItems.filter(i => i.id !== id);
  render();
}

// ========== Re-OCR ==========
async function reOCR(id) {
  const item = imgItems.find(i => i.id === id);
  if (!item) return;

  // If image has saved regions, use region OCR instead
  if (item.regions && item.regions.length > 0) {
    await regionOCRById(id);
    return;
  }

  updateItemStatus(id, 'recognizing');
  render();
  try {
    const result = await api('/api/ocr/' + id, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({image_base64: item.data, ...getAIConfigPayload()})
    });
    if (result && !result.error) {
      setText(item, result.text || '');
      setStatus(item, 'done');
    } else {
      updateItemStatus(id, 'error');
    }
  } catch(e) {
    updateItemStatus(id, 'error');
  }
  render();
}

// Region OCR using saved regions (without opening modal)

async function regionOCRById(id) {
  const item = imgItems.find(i => i.id === id);
  if (!item || !item.regions || item.regions.length === 0) return;

  updateItemStatus(id, 'recognizing');
  render();

  try {
    const payload = {image_base64: item.data, regions: item.regions, ...getAIConfigPayload()};
    const result = await api('/api/ocr/region', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (!item.aborted && result && !result.error && result.text) {
      item.text = result.text;
      item.status = 'done';
      saveText(id, result.text);
    }
  } catch(e) {}

  if (item.aborted) { item.status = 'ready'; }
  delete item.aborted;
  render();
}

// ========== Save Text ==========
let saveTimer = null;
function saveText(id, text) {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    await api('/api/text/'+id, {
      method:'PUT',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    // Also update local
    for (let i = 0; i < imgItems.length; i++) {
      if (imgItems[i].id === id) { imgItems[i].text = text; break; }
    }
  }, 800);
}


// ========== CapCut-style Pointer Drag Reorder ==========
const _drag = {active:false, id:null, el:null, startY:0, origIdx:0, targetIdx:-1, moved:false, cardH:0};
let _dropPreview = null;

function _getDropPreview() {
  if (!_dropPreview) {
    _dropPreview = document.createElement('div');
    _dropPreview.className = 'drop-preview';
    document.getElementById('sidebarList').appendChild(_dropPreview);
  }
  return _dropPreview;
}

function onPointerDown(e) {
  if (e.button !== 0 || _drag.active) return;
  const card = e.currentTarget.closest('.img-card');
  if (!card) return;
  _drag.id = parseInt(card.dataset.id);
  _drag.el = card;
  _drag.origIdx = _drag.targetIdx = parseInt(card.dataset.idx);
  _drag.startY = e.clientY;
  _drag.moved = false;
  card.setPointerCapture(e.pointerId);
  card.addEventListener('pointermove', onPointerMove);
  card.addEventListener('pointerup', onPointerUp);
  card.addEventListener('pointercancel', onPointerUp);
}

function onPointerMove(e) {
  if (!_drag.active) {
    if (Math.abs(e.clientY - _drag.startY) < 5) return;
    _drag.active = true;
    _drag.moved = true;
    _drag.el.classList.add('dragging');
    _drag.cardH = _drag.el.getBoundingClientRect().height + 8;
  }

  // Move dragged card vertically
  _drag.el.style.transform = 'translateY(' + (e.clientY - _drag.startY) + 'px)';

  // Calculate target index: count how many non-dragged cards the pointer has passed
  const list = document.getElementById('sidebarList');
  const cards = list.querySelectorAll('.img-card');
  let newTarget = 0;
  for (const c of cards) {
    if (parseInt(c.dataset.id) === _drag.id) continue;
    const rect = c.getBoundingClientRect();
    if (e.clientY >= rect.top + rect.height / 2) {
      newTarget++;
    } else {
      break;
    }
  }

  if (newTarget !== _drag.targetIdx) {
    _drag.targetIdx = newTarget;
    _applyShifts();
    _showPreview(newTarget);
  }
}

function _applyShifts() {
  const list = document.getElementById('sidebarList');
  const cards = list.querySelectorAll('.img-card');
  const h = _drag.cardH;
  const from = _drag.origIdx;
  const to = _drag.targetIdx;

  cards.forEach(card => {
    const idx = parseInt(card.dataset.idx);
    if (idx === _drag.origIdx) return;
    let shift = 0;
    if (from < to) {
      if (idx > from && idx <= to) shift = -h;
    } else if (from > to) {
      if (idx >= to && idx < from) shift = h;
    }
    card.style.transform = shift ? 'translateY(' + shift + 'px)' : '';
  });
}

function _showPreview(targetIdx) {
  const list = document.getElementById('sidebarList');
  const cards = list.querySelectorAll('.img-card:not(.dragging)');
  const preview = _getDropPreview();
  const listRect = list.getBoundingClientRect();

  if (targetIdx === _drag.origIdx) {
    preview.style.opacity = '0';
    return;
  }

  // Position the preview box at the gap between cards
  let gapTop, gapBottom;
  if (targetIdx >= cards.length) {
    const lastCard = cards[cards.length - 1];
    if (!lastCard) { preview.style.opacity = '0'; return; }
    const r = lastCard.getBoundingClientRect();
    gapTop = r.bottom - listRect.top + list.scrollTop;
    gapBottom = gapTop + 8;
  } else if (targetIdx === 0) {
    const firstCard = cards[0];
    if (!firstCard) { preview.style.opacity = '0'; return; }
    const r = firstCard.getBoundingClientRect();
    gapBottom = r.top - listRect.top + list.scrollTop;
    gapTop = gapBottom - 8;
  } else {
    const above = cards[targetIdx - 1];
    const below = cards[targetIdx];
    if (!above || !below) { preview.style.opacity = '0'; return; }
    const rAbove = above.getBoundingClientRect();
    const rBelow = below.getBoundingClientRect();
    gapTop = rAbove.bottom - listRect.top + list.scrollTop;
    gapBottom = rBelow.top - listRect.top + list.scrollTop;
  }

  preview.style.top = gapTop + 'px';
  preview.style.height = Math.max(gapBottom - gapTop, 4) + 'px';
  preview.style.opacity = '1';
}

function onPointerUp(e) {
  const card = _drag.el;
  card.removeEventListener('pointermove', onPointerMove);
  card.removeEventListener('pointerup', onPointerUp);
  card.removeEventListener('pointercancel', onPointerUp);

  if (!_drag.active) {
    _drag.active = false;
    _drag.moved = false;
    return;
  }

  // Hide preview
  _getDropPreview().style.opacity = '0';

  const from = _drag.origIdx;
  const to = _drag.targetIdx;

  if (to !== from) {
    // Commit reorder
    const [moved] = imgItems.splice(from, 1);
    imgItems.splice(to, 0, moved);
    api('/api/reorder', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ids: imgItems.map(i => i.id)})
    });

    // FLIP: snapshot current visual positions (including shifted transforms)
    const list = document.getElementById('sidebarList');
    const cards = list.querySelectorAll('.img-card');
    const snapshots = [];
    cards.forEach(c => snapshots.push({id:parseInt(c.dataset.id), top:c.getBoundingClientRect().top}));

    // Re-render with new order
    render();

    // Invert: lock cards at their old visual positions
    const newCards = list.querySelectorAll('.img-card');
    for (const nc of newCards) {
      const id = parseInt(nc.dataset.id);
      const snap = snapshots.find(s => s.id === id);
      if (!snap) continue;
      const newTop = nc.getBoundingClientRect().top;
      const dy = snap.top - newTop;
      if (Math.abs(dy) < 1) continue;
      nc.style.transition = 'none';
      nc.style.transform = 'translateY(' + dy + 'px)';
    }

    // Play: animate to final positions
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        newCards.forEach(c => {
          if (!c.style.transform) return;
          c.style.transition = 'transform .3s cubic-bezier(.22,1,.36,1)';
          c.style.transform = '';
          c.addEventListener('transitionend', function h() {
            c.removeEventListener('transitionend', h);
            c.style.transition = '';
            c.style.transform = '';
          }, {once:true});
        });
      });
    });
  } else {
    // No reorder — animate back to origin
    // Clear all other cards' transforms first
    document.querySelectorAll('.img-card').forEach(c => {
      if (parseInt(c.dataset.id) !== _drag.id) { c.style.transform = ''; c.style.transition = ''; }
    });
    card.classList.remove('dragging');
    card.style.transition = 'transform .25s cubic-bezier(.22,1,.36,1)';
    card.style.transform = '';
    card.addEventListener('transitionend', function h() {
      card.removeEventListener('transitionend', h);
      card.style.transition = '';
    }, {once:true});
    setTimeout(() => { card.style.transition = ''; }, 300);
  }

  // Prevent click after drag
  if (_drag.moved) {
    card.addEventListener('click', function prevent(ev) { ev.stopPropagation(); card.removeEventListener('click', prevent); }, {once:true});
  }

  _drag.active = false;
  _drag.moved = false;
}

// ========== Recognize All ==========
async function recognizeAll() {
  if (ocrRunning || imgItems.length === 0) return;
  ocrRunning = true;
  document.getElementById('btnRecognizeAll').disabled = true;
  document.getElementById('btnRecognizeAll').textContent = '停止识别';
  document.getElementById('btnRecognizeAll').onclick = stopAllOCR;
  for (const item of imgItems) {
    item.aborted = false;
    updateItemStatus(item.id, 'recognizing');
  }
  render();

  const promises = imgItems.map(item =>
    api('/api/ocr/' + item.id, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image_base64: item.data, ...getAIConfigPayload()})
    }).then(r => {
      if (item.aborted) return;
      if (r && !r.error) {
        item.text = r.text || '';
        item.status = 'done';
      } else {
        updateItemStatus(item.id, 'error');
      }
      render();
      return r;
    }).catch(() => {
      if (!item.aborted) { updateItemStatus(item.id, 'error'); render(); }
    })
  );
  await Promise.all(promises);
  resetOCRButton();
}

function stopAllOCR() {
  for (const item of imgItems) {
    if (getStatus(item) === 'recognizing') {
      item.aborted = true;
      updateItemStatus(item.id, 'ready');
    }
  }
  resetOCRButton();
  render();
}

function resetOCRButton() {
  ocrRunning = false;
  const btn = document.getElementById('btnRecognizeAll');
  btn.disabled = false;
  btn.textContent = '全部识别';
  btn.onclick = recognizeAll;
}

// ========== Export ==========
function previewText() {
  if (imgItems.length === 0) { alert('没有内容可预览'); return; }
  window.open('/api/export', '_blank');
}

function exportTxt() {
  if (imgItems.length === 0) { alert('没有内容可导出'); return; }

  let text = '';
  imgItems.forEach((item) => {
    if (item.text) text += item.text + '\n\n';
  });
  const blob = new Blob([text], {type:'text/plain;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '多图文识别_' + new Date().toISOString().slice(0,10) + '.txt';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ========== Clear ==========
async function clearAll() {
  if (imgItems.length === 0) return;
  if (!confirm('确定清空当前模式的所有图片和文本？')) return;
  await fetch('/api/images/clear', {method: 'DELETE'});
  imgItems = [];
  activeImgId = null;
  render();
}

// ========== Region Select Modal ==========
function openRegionModal(id) {
  regionImgId = id;
  const item = imgItems.find(i => i.id === id);
  if (!item) return;
  const modal = document.getElementById('regionModal');
  modal.classList.add('show');
  // Reset button state
  const btn = document.getElementById('btnConfirmRegion');
  btn.disabled = false;
  btn.textContent = '识别选定区域';
  regionCanvas = document.getElementById('modalCanvas');
  regionCtx = regionCanvas.getContext('2d');
  // Restore saved regions (migrate old rect format to quad)
  regions = (item.regions || []).map(r => {
    if (r.points) return JSON.parse(JSON.stringify(r));
    return {points:[{x:r.x1,y:r.y1},{x:r.x2,y:r.y1},{x:r.x2,y:r.y2},{x:r.x1,y:r.y2}]};
  });

  regionImg = new Image();
  regionImg.onload = () => {
    resizeRegionCanvas();
  };
  regionImg.src = item.data;
}

function closeRegionModal() {
  // Save regions to item before closing
  if (regionImgId) {
    const item = imgItems.find(i => i.id === regionImgId);
    if (item) {
      const r = JSON.stringify(regions);
      item.regions = JSON.parse(r);
      api('/api/regions/' + regionImgId, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({regions: r})});
    }
  }
  document.getElementById('regionModal').classList.remove('show');
  regionImgId = null;
  regionImg = null;
  regions = [];
}

function resizeRegionCanvas() {
  const wrap = document.getElementById('modalCanvasWrap');
  const rect = wrap.getBoundingClientRect();
  regionCanvas.width = rect.width;
  regionCanvas.height = rect.height;
  if (regionImg) {
    const sx = (regionCanvas.width - 20) / regionImg.naturalWidth;
    const sy = (regionCanvas.height - 20) / regionImg.naturalHeight;
    regionView.base = Math.min(sx, sy, 1);
    // For large images: use height-fit if width-fit would make it too small
    if (sx < sy && sx < 0.3) {
      regionView.base = Math.min(sy, 1);
    }
    regionView.scale = 1;
    regionView.x = (regionCanvas.width - regionImg.naturalWidth * regionView.base) / 2;
    regionView.y = (regionCanvas.height - regionImg.naturalHeight * regionView.base) / 2;
  }
  redrawRegion();
}

function redrawRegion() {
  if (!regionCtx || !regionImg) return;
  const c = regionCtx, cv = regionCanvas;
  const s = regionView.base * regionView.scale;
  c.clearRect(0, 0, cv.width, cv.height);
  c.fillStyle = '#1a1a2e';
  c.fillRect(0, 0, cv.width, cv.height);
  c.drawImage(regionImg, regionView.x, regionView.y, regionImg.naturalWidth*s, regionImg.naturalHeight*s);

  const colors = ['#00ff88','#ff4466','#44aaff','#ffaa00','#cc66ff'];
  regions.forEach((r, i) => {
    const pts = r.points.map(p => rToC(p.x, p.y));
    const color = colors[i % colors.length];
    // Filled polygon
    c.beginPath();
    c.moveTo(pts[0].x, pts[0].y);
    for (let j = 1; j < 4; j++) c.lineTo(pts[j].x, pts[j].y);
    c.closePath();
    c.fillStyle = color + '25';
    c.fill();
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.stroke();
    // Corner handles
    for (let j = 0; j < 4; j++) {
      c.beginPath();
      c.arc(pts[j].x, pts[j].y, 5, 0, Math.PI*2);
      c.fillStyle = '#fff';
      c.fill();
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.stroke();
    }
    // Number badge at first point
    c.font = 'bold 14px sans-serif';
    c.fillStyle = color;
    c.fillRect(pts[0].x, pts[0].y-20, 24, 18);
    c.fillStyle = '#fff';
    c.fillText('#'+(i+1), pts[0].x+4, pts[0].y-6);
  });

  // Draw preview
  if (regionDrag.mode === 'draw' && regionDrag.start && regionDrag.current) {
    const x1 = Math.min(regionDrag.start.x, regionDrag.current.x);
    const y1 = Math.min(regionDrag.start.y, regionDrag.current.y);
    const w = Math.abs(regionDrag.current.x - regionDrag.start.x);
    const h = Math.abs(regionDrag.current.y - regionDrag.start.y);
    c.strokeStyle = '#00ff88';
    c.lineWidth = 2;
    c.setLineDash([6,3]);
    c.strokeRect(x1, y1, w, h);
    c.setLineDash([]);
    c.fillStyle = 'rgba(0,255,136,0.12)';
    c.fillRect(x1, y1, w, h);
  }

  c.font = '11px sans-serif';
  c.fillStyle = '#666';
  c.fillText(Math.round(regionView.scale*100)+'%', 8, cv.height - 8);
}

function rToC(ox, oy) {
  const s = regionView.base * regionView.scale;
  return {x: ox*s + regionView.x, y: oy*s + regionView.y};
}
function cToR(cx, cy) {
  const s = regionView.base * regionView.scale;
  return {x: (cx - regionView.x)/s, y: (cy - regionView.y)/s};
}

function findCornerAt(cx, cy, threshold) {
  for (let ri = regions.length-1; ri >= 0; ri--) {
    for (let ci = 0; ci < 4; ci++) {
      const p = rToC(regions[ri].points[ci].x, regions[ri].points[ci].y);
      const dx = p.x-cx, dy = p.y-cy;
      if (dx*dx+dy*dy <= threshold*threshold) return {regionIdx:ri, cornerIdx:ci};
    }
  }
  return null;
}

function pointInQuad(px, py, pts) {
  let inside = false;
  for (let i = 0, j = 3; i < 4; j = i++) {
    const xi=pts[i].x, yi=pts[i].y, xj=pts[j].x, yj=pts[j].y;
    if (((yi>py)!==(yj>py)) && (px<(xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}

// Modal mouse events
document.getElementById('modalCanvas').addEventListener('mousedown', e => {
  if (!regionImg) return;
  const rect = regionCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;

  if (e.button === 1 || (e.button === 0 && regionSpace)) {
    regionDrag = {mode:'pan', start:{x:cx, y:cy}, ox:regionView.x, oy:regionView.y, regionIdx:-1, cornerIdx:-1};
    regionCanvas.style.cursor = 'grabbing';
    return;
  }
  if (e.button === 2) {
    const orig = cToR(cx, cy);
    for (let i = regions.length-1; i >= 0; i--) {
      if (pointInQuad(orig.x, orig.y, regions[i].points)) {
        regions.splice(i, 1);
        redrawRegion();
        return;
      }
    }
    return;
  }
  // Left click: check corner first, then draw
  const hit = findCornerAt(cx, cy, 8);
  if (hit) {
    regionDrag = {mode:'corner', regionIdx:hit.regionIdx, cornerIdx:hit.cornerIdx, start:{x:cx, y:cy}};
    regionCanvas.style.cursor = 'move';
  } else {
    regionDrag = {mode:'draw', start:{x:cx, y:cy}, current:{x:cx, y:cy}, regionIdx:-1, cornerIdx:-1};
  }
});

document.getElementById('modalCanvas').addEventListener('mousemove', e => {
  const rect = regionCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  if (regionDrag.mode === 'pan') {
    regionView.x = regionDrag.ox + (cx - regionDrag.start.x);
    regionView.y = regionDrag.oy + (cy - regionDrag.start.y);
    redrawRegion();
  } else if (regionDrag.mode === 'corner') {
    const orig = cToR(cx, cy);
    regions[regionDrag.regionIdx].points[regionDrag.cornerIdx] = {
      x: Math.max(0, Math.min(orig.x, regionImg.naturalWidth)),
      y: Math.max(0, Math.min(orig.y, regionImg.naturalHeight))
    };
    redrawRegion();
  } else if (regionDrag.mode === 'draw') {
    regionDrag.current = {x:cx, y:cy};
    redrawRegion();
  } else {
    // Idle: update cursor based on corner proximity
    const hit = findCornerAt(cx, cy, 8);
    regionCanvas.style.cursor = hit ? 'move' : 'crosshair';
  }
});

document.getElementById('modalCanvas').addEventListener('mouseup', e => {
  if (regionDrag.mode === 'draw' && regionDrag.start && regionDrag.current) {
    const o1 = cToR(Math.min(regionDrag.start.x, regionDrag.current.x), Math.min(regionDrag.start.y, regionDrag.current.y));
    const o2 = cToR(Math.max(regionDrag.start.x, regionDrag.current.x), Math.max(regionDrag.start.y, regionDrag.current.y));
    const rw = o2.x - o1.x, rh = o2.y - o1.y;
    if (rw > 5 && rh > 5) {
      regions.push({points:[
        {x:Math.max(0,o1.x), y:Math.max(0,o1.y)},
        {x:Math.min(regionImg.naturalWidth,o2.x), y:Math.max(0,o1.y)},
        {x:Math.min(regionImg.naturalWidth,o2.x), y:Math.min(regionImg.naturalHeight,o2.y)},
        {x:Math.max(0,o1.x), y:Math.min(regionImg.naturalHeight,o2.y)}
      ]});
    }
  }
  regionDrag = {mode:'none', regionIdx:-1, cornerIdx:-1};
  redrawRegion();
});

document.getElementById('modalCanvas').addEventListener('wheel', e => {
  e.preventDefault();
  if (!regionImg) return;
  const rect = regionCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
  const ns = Math.max(0.1, Math.min(regionView.scale * factor, 20));
  regionView.x = cx - (cx - regionView.x) * (ns / regionView.scale);
  regionView.y = cy - (cy - regionView.y) * (ns / regionView.scale);
  regionView.scale = ns;
  redrawRegion();
}, {passive:false});

document.getElementById('modalCanvas').addEventListener('contextmenu', e => e.preventDefault());

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !e.repeat) { regionSpace = true; if (regionCanvas) regionCanvas.style.cursor = 'grab'; }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space') { regionSpace = false; if (regionCanvas) regionCanvas.style.cursor = 'crosshair'; }
});

// Confirm region OCR
async function confirmRegionOCR() {
  if (regions.length === 0) { alert('请先框选区域'); return; }
  if (!regionImgId || !regionImg) return;

  const btn = document.getElementById('btnConfirmRegion');

  // Save regions, set status, close modal immediately
  const savedRegions = JSON.parse(JSON.stringify(regions));
  const savedImgId = regionImgId;
  const savedImg = regionImg;
  const savedItem = imgItems.find(i => i.id === regionImgId);
  closeRegionModal();
  if (savedItem) savedItem.aborted = false;
  updateItemStatus(savedImgId, 'recognizing');
  render();

  let allText = '';
  try {
    const payload = {image_base64: savedItem.data, regions: savedRegions, ...getAIConfigPayload()};
    const result = await api('/api/ocr/region', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (savedItem && savedItem.aborted) { savedItem.status = 'ready'; }
    else if (result && !result.error && result.text) {
      savedItem.text = result.text;
      savedItem.status = 'done';
      saveText(savedImgId, result.text);
    }
  } catch(e) {}

  if (savedItem) delete savedItem.aborted;

  render();
}

// Resize observer for modal
new ResizeObserver(() => { if (regionImg) resizeRegionCanvas(); }).observe(document.getElementById('modalCanvasWrap'));

// ========== Config Modal ==========
let aiConfig = {
  model: localStorage.getItem('aiModel') || 'glm-4v-flash',
  api_url: localStorage.getItem('aiApiUrl') || 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions',
  api_key: localStorage.getItem('aiApiKey') || ''
};

// Per-model key storage
function getPerModelKey(modelName) {
  const keys = JSON.parse(localStorage.getItem('perModelKeys') || '{}');
  return keys[modelName] || '';
}
function setPerModelKey(modelName, apiKey, apiUrl) {
  const keys = JSON.parse(localStorage.getItem('perModelKeys') || '{}');
  if (apiKey) keys[modelName] = {key: apiKey, url: apiUrl};
  else delete keys[modelName];
  localStorage.setItem('perModelKeys', JSON.stringify(keys));
}

// Load persistent config from server
async function loadServerConfig() {
  try {
    const cfg = await api('/api/config');
    if (cfg.api_key) aiConfig.api_key = cfg.api_key;
    if (cfg.api_url) aiConfig.api_url = cfg.api_url;
    if (cfg.model) aiConfig.model = cfg.model;
  } catch(e) {}
}
loadServerConfig();

const presets = [
  {name:'glm-4v-flash',   label:'GLM-4V Flash (免费)', url:'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions'},
  {name:'glm-4.6v',       label:'GLM-4.6V (智谱专业OCR)', url:'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions'},
  {name:'minimax-vlm',    label:'MiniMax VLM (Token Plan)', url:'https://api.minimaxi.com'},
];



function openConfigModal() {
  document.getElementById('cfgUrl').value = aiConfig.api_url;
  document.getElementById('cfgModel').value = aiConfig.model;
  const saved = getPerModelKey(aiConfig.model);
  document.getElementById('cfgKey').value = saved.key || aiConfig.api_key;
  document.getElementById('configModal').classList.add('show');
  renderPresets();
}

function closeConfigModal() {
  document.getElementById('configModal').classList.remove('show');
}

function selectPreset(p) {
  document.getElementById('cfgUrl').value = p.url;
  document.getElementById('cfgModel').value = p.name;
  const saved = getPerModelKey(p.name);
  if (saved) {
    document.getElementById('cfgKey').value = saved.key || '';
    if (saved.url) document.getElementById('cfgUrl').value = saved.url;
    aiConfig.api_key = saved.key;
    aiConfig.api_url = saved.url;
  } else {
    document.getElementById('cfgKey').value = aiConfig.api_key;
  }
  aiConfig.model = p.name;
  renderPresets();
}


function renderPresets() {
  const cur = document.getElementById('cfgModel').value;
  const container = document.getElementById('presetList');
  container.innerHTML = presets.map(p =>
    '<span class="preset-tag'+(p.name===cur?' active':'')+'" onclick="selectPreset(presets['+presets.indexOf(p)+'])">'+p.label+'</span>'
  ).join('');
}


function saveConfig() {
  aiConfig.api_url = document.getElementById('cfgUrl').value.trim();
  aiConfig.model = document.getElementById('cfgModel').value.trim();
  aiConfig.api_key = document.getElementById('cfgKey').value.trim();
  // Save per-model key
  setPerModelKey(aiConfig.model, aiConfig.api_key, aiConfig.api_url);
  localStorage.setItem('aiModel', aiConfig.model);
  localStorage.setItem('aiApiUrl', aiConfig.api_url);
  localStorage.setItem('aiApiKey', aiConfig.api_key);
  // Persist to server
  api('/api/config', {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
    api_key: aiConfig.api_key, api_url: aiConfig.api_url, model: aiConfig.model,

  })});
  closeConfigModal();
}

function getAIConfigPayload() {
  if (ocrMode !== 'ai') return {};
  return {
    ai_config: {
      model: aiConfig.model,
      api_url: aiConfig.api_url,
      api_key: aiConfig.api_key,

    }
  };
}
</script>
</body>
</html>
`

// ==================== AI Config Persistence ====================

const configFile = "config.json"

func loadServerConfig() map[string]string {
	data, err := os.ReadFile(configFile)
	if err != nil {
		return map[string]string{}
	}
	var cfg map[string]string
	if json.Unmarshal(data, &cfg) != nil {
		return map[string]string{}
	}
	return cfg
}

func saveServerConfig(cfg map[string]string) {
	data, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(configFile, data, 0644)
}

func handleGetConfig(w http.ResponseWriter, r *http.Request) {
	cfg := loadServerConfig()
	jsonOK(w, cfg)
}

func handleSaveConfig(w http.ResponseWriter, r *http.Request) {
	var cfg map[string]string
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		jsonErr(w, "invalid json")
		return
	}
	saveServerConfig(cfg)
	jsonOK(w, map[string]string{"ok": "1"})
}

// ==================== Main ====================

func main() {
	initDB()
	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/api/upload", handleUpload)
	http.HandleFunc("/api/images", handleListImages)
	http.HandleFunc("/api/reorder", handleReorder)
	http.HandleFunc("/api/images/clear", handleClearImages)
	http.HandleFunc("/api/images/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			handleDeleteImage(w, r)
		}
	})
	http.HandleFunc("/api/ocr/region", handleOCRRegion)
	http.HandleFunc("/api/ocr/", handleOCR)
	http.HandleFunc("/api/text/", handleSaveText)
	http.HandleFunc("/api/regions/", handleSaveRegions)
	http.HandleFunc("/api/export", handleExport)
	http.HandleFunc("/api/export/xlsx", handleExportXlsx)
	http.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			handleGetConfig(w, r)
		} else if r.Method == http.MethodPut {
			handleSaveConfig(w, r)
		}
	})

	port := "29817"
	fmt.Printf("多图识别服务: http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
