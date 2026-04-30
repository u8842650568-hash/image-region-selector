package main

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"

	_ "github.com/mattn/go-sqlite3"
	"github.com/xuri/excelize/v2"
)

const ocrURL = "http://127.0.0.1:8081"

// ==================== Data Model ====================

type ImageItem struct {
	ID         int    `json:"id"`
	Name       string `json:"name"`
	Data       string `json:"data"`       // base64 data URL
	Thumb      string `json:"thumb"`      // small thumbnail base64
	Text       string `json:"text"`       // text recognition result
	TableText  string `json:"table_text"` // table recognition result
	Status     string `json:"status"`     // text recognition status
	TableStatus string `json:"table_status"` // table recognition status
	RecogType  string `json:"recog_type"` // "text" or "table"
	Order      int    `json:"order"`
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
	db.Exec("ALTER TABLE images ADD COLUMN table_text TEXT DEFAULT ''")
	db.Exec("ALTER TABLE images ADD COLUMN table_status TEXT DEFAULT 'ready'")
	db.Exec("ALTER TABLE images ADD COLUMN recog_type TEXT DEFAULT 'text'")
	loadFromDB()
}

func loadFromDB() {
	rows, err := db.Query("SELECT id, name, data, text, table_text, status, table_status, recog_type, sort_order FROM images ORDER BY sort_order")
	if err != nil {
		log.Println("DB load:", err)
		return
	}
	defer rows.Close()
	images = []ImageItem{}
	maxID := 0
	for rows.Next() {
		var item ImageItem
		rows.Scan(&item.ID, &item.Name, &item.Data, &item.Text, &item.TableText, &item.Status, &item.TableStatus, &item.RecogType, &item.Order)
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
	db.Exec("INSERT INTO images (name, data, text, status, sort_order, recog_type) VALUES (?, ?, ?, ?, ?, ?)",
		item.Name, item.Data, item.Text, item.Status, item.Order, item.RecogType)
	var id int
	db.QueryRow("SELECT last_insert_rowid()").Scan(&id)
	item.ID = id
}

func dbUpdateText(id int, text string) {
	db.Exec("UPDATE images SET text = ? WHERE id = ?", text, id)
}
func dbUpdateTableText(id int, text string) {
	db.Exec("UPDATE images SET table_text = ? WHERE id = ?", text, id)
}

func dbUpdateStatus(id int, status string) {
	db.Exec("UPDATE images SET status = ? WHERE id = ?", status, id)
}
func dbUpdateTableStatus(id int, status string) {
	db.Exec("UPDATE images SET table_status = ? WHERE id = ?", status, id)
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

	mu.Lock()
	nextID++
	recogType := r.FormValue("recog_type")
	if recogType == "" {
		recogType = "text"
	}
	item := ImageItem{
		ID:        nextID,
		Name:      header.Filename,
		Data:      dataURL,
		Thumb:     dataURL,
		Status:    "ready",
		RecogType: recogType,
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
	filter := r.URL.Query().Get("type")
	if filter != "" {
		filtered := []ImageItem{}
		for _, img := range images {
			if img.RecogType == filter {
				filtered = append(filtered, img)
			}
		}
		jsonOK(w, filtered)
	} else {
		jsonOK(w, images)
	}
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

func handleOCR(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/ocr/")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		jsonErr(w, "invalid id")
		return
	}

	var req struct {
		ImageBase64 string                 `json:"image_base64"`
		Mode        string                 `json:"mode"`
		RecogType   string                 `json:"recog_type"`
		AIConfig    map[string]interface{} `json:"ai_config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonErr(w, err.Error())
		return
	}

	isTable := req.RecogType == "table"
	mu.Lock()
	for i := range images {
		if images[i].ID == id {
			if isTable {
				images[i].TableStatus = "recognizing"
			} else {
				images[i].Status = "recognizing"
			}
		}
	}
	mu.Unlock()
	if isTable {
		dbUpdateTableStatus(id, "recognizing")
	} else {
		dbUpdateStatus(id, "recognizing")
	}

	// Call OCR service
	ocrReq := map[string]interface{}{"image_base64": req.ImageBase64}
	if req.Mode != "" {
		ocrReq["mode"] = req.Mode
	}
	if req.RecogType != "" {
		ocrReq["recog_type"] = req.RecogType
	}
	if req.AIConfig != nil {
		ocrReq["ai_config"] = req.AIConfig
	}
	bodyBytes, _ := json.Marshal(ocrReq)

	resp, err := http.Post(ocrURL+"/ocr", "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		mu.Lock()
		for i := range images {
			if images[i].ID == id {
				if isTable {
					images[i].TableStatus = "error"
				} else {
					images[i].Status = "error"
				}
			}
		}
		mu.Unlock()
		if isTable {
			dbUpdateTableStatus(id, "error")
		} else {
			dbUpdateStatus(id, "error")
		}
		jsonErr(w, "OCR服务不可用: "+err.Error())
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	text, _ := result["text"].(string)
	status, _ := result["status"].(string)

	mu.Lock()
	for i := range images {
		if images[i].ID == id {
			if isTable {
				images[i].TableText = text
				images[i].TableStatus = "done"
			} else {
				images[i].Text = text
				images[i].Status = "done"
			}
		}
	}
	mu.Unlock()
	if isTable {
		dbUpdateTableText(id, text)
		dbUpdateTableStatus(id, "done")
	} else {
		dbUpdateText(id, text)
		dbUpdateStatus(id, "done")
	}

	jsonOK(w, map[string]interface{}{"text": text, "status": status})
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

func handleSaveTableText(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/table_text/")
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
			images[i].TableText = req.Text
		}
	}
	mu.Unlock()
	dbUpdateTableText(id, req.Text)

	jsonOK(w, map[string]string{"ok": "1"})
}

func handleExport(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	exportType := r.URL.Query().Get("type")
	var buf strings.Builder
	for i, item := range images {
		if i > 0 {
			buf.WriteString("\n\n" + strings.Repeat("=", 60) + "\n\n")
		}
		if exportType == "table" {
			buf.WriteString(item.TableText)
		} else {
			buf.WriteString(item.Text)
		}
	}
	if exportType == "table" {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;margin:20px auto;padding:0 16px}.ocr-table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px;line-height:1.6}.ocr-table th,.ocr-table td{border:1px solid #d9d9d9;padding:8px 12px;text-align:left}.ocr-table th{background:#fafafa;font-weight:600}.ocr-table tbody tr:nth-child(even){background:#fafafa}.hr{border:none;border-top:1px solid #e8e8e8;margin:32px 0}</style></head><body>" + parseMarkdownTables(buf.String()) + "</body></html>"))
	} else {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Write([]byte(buf.String()))
	}
}

func parseMarkdownTables(md string) string {
	lines := strings.Split(md, "\n")
	var html strings.Builder
	inTable := false
	var tableRows [][]string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "|") && strings.HasSuffix(trimmed, "|") {
			if isSeparatorRow(trimmed) {
				continue
			}
			cells := strings.Split(trimmed, "|")
			if len(cells) > 2 {
				cells = cells[1 : len(cells)-1]
			}
			var row []string
			for _, c := range cells {
				row = append(row, strings.TrimSpace(c))
			}
			tableRows = append(tableRows, row)
			inTable = true
		} else {
			if inTable && len(tableRows) > 0 {
				html.WriteString(buildTableHTML(tableRows))
				tableRows = nil
				inTable = false
			}
			if trimmed != "" {
				html.WriteString("<p>" + escapeHTML(trimmed) + "</p>")
			}
		}
	}
	if inTable && len(tableRows) > 0 {
		html.WriteString(buildTableHTML(tableRows))
	}
	return html.String()
}

func isSeparatorRow(s string) bool {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "|") || !strings.HasSuffix(s, "|") {
		return false
	}
	s = s[1 : len(s)-1]
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return true
	}
	for _, c := range s {
		if c != '-' && c != ':' && c != '|' && c != ' ' {
			return false
		}
	}
	return true
}

func escapeHTML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

func buildTableHTML(rows [][]string) string {
	if len(rows) == 0 {
		return ""
	}
	cols := 0
	for _, r := range rows {
		if len(r) > cols {
			cols = len(r)
		}
	}
	var html strings.Builder
	html.WriteString("<table class=\"ocr-table\"><thead><tr>")
	for c := 0; c < cols; c++ {
		val := ""
		if c < len(rows[0]) {
			val = rows[0][c]
		}
		html.WriteString("<th>" + escapeHTML(val) + "</th>")
	}
	html.WriteString("</tr></thead>")
	if len(rows) > 1 {
		html.WriteString("<tbody>")
		for r := 1; r < len(rows); r++ {
			html.WriteString("<tr>")
			for c := 0; c < cols; c++ {
				val := ""
				if c < len(rows[r]) {
					val = rows[r][c]
				}
				html.WriteString("<td>" + escapeHTML(val) + "</td>")
			}
			html.WriteString("</tr>")
		}
		html.WriteString("</tbody>")
	}
	html.WriteString("</table>")
	return html.String()
}

func handleExportXlsx(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	f := excelize.NewFile()
	defaultSheet := f.GetSheetName(0)
	sheetIdx := 0
	for _, item := range images {
		if item.TableText == "" {
			continue
		}
		sheetIdx++
		sheetName := fmt.Sprintf("图片%d", sheetIdx)
		if sheetIdx == 1 {
			// Reuse default sheet, rename it
			f.SetSheetName(defaultSheet, sheetName)
		} else {
			f.NewSheet(sheetName)
		}
		lines := strings.Split(item.TableText, "\n")
		rowIdx := 1
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "|") && strings.HasSuffix(trimmed, "|") && !isSeparatorRow(trimmed) {
				cells := strings.Split(trimmed, "|")
				if len(cells) > 2 {
					cells = cells[1 : len(cells)-1]
				}
				for ci, c := range cells {
					colName, _ := excelize.ColumnNumberToName(ci + 1)
					cellRef := fmt.Sprintf("%s%d", colName, rowIdx)
					f.SetCellValue(sheetName, cellRef, strings.TrimSpace(c))
				}
				rowIdx++
			}
		}
	}

	w.Header().Set("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
	w.Header().Set("Content-Disposition", "attachment; filename=多表格识别.xlsx")
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
<title>多图文本识别</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"WenQuanYi Micro Hei","Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#f5f5f5;height:100vh;display:flex;flex-direction:column;overflow:hidden;color:#333}

/* Toolbar */
.toolbar{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#fff;border-bottom:1px solid #e0e0e0;flex-shrink:0}
.mode-select{position:relative;display:inline-flex;align-items:center;gap:4px;padding:6px 12px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;user-select:none;transition:all .15s;background:#fff;font-size:14px;font-weight:500;color:#333}
.mode-select:hover{border-color:#4096ff;color:#4096ff}
.mode-select.open{border-color:#4096ff;box-shadow:0 0 0 2px rgba(22,119,255,.15)}
.mode-select-arrow{font-size:10px;color:#999;transition:transform .2s}
.mode-select.open .mode-select-arrow{transform:rotate(180deg)}
.mode-select-dropdown{position:absolute;top:calc(100% + 4px);left:0;min-width:100%;background:#fff;border:1px solid #e8e8e8;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,.1);z-index:200;display:none;overflow:hidden}
.mode-select.open .mode-select-dropdown{display:block}
.mode-select-option{padding:8px 14px;font-size:13px;cursor:pointer;transition:background .1s;color:#333}
.mode-select-option:hover{background:#f0f7ff;color:#4096ff}
.mode-select-option.active{color:#1677ff;font-weight:500;background:#e6f4ff}
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
.text-content.table-content{white-space:normal;word-break:normal}
.ocr-table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px;line-height:1.6}
.ocr-table th,.ocr-table td{border:1px solid #d9d9d9;padding:8px 12px;text-align:left;outline:none}
.ocr-table td[contenteditable=true]:focus,.ocr-table th[contenteditable=true]:focus{box-shadow:inset 0 0 0 2px #4096ff;background:#fff}
.ocr-table th{background:#fafafa;font-weight:600;white-space:nowrap;position:sticky;top:0}
.ocr-table tbody tr:hover{background:#f0f7ff}
.ocr-table tbody tr:nth-child(even){background:#fafafa}
.ocr-table+.ocr-table{margin-top:16px}
.edit-md-btn{margin-left:auto;padding:2px 8px;font-size:11px;border:1px solid #d9d9d9;border-radius:4px;cursor:pointer;background:#fff;color:#666;transition:all .15s}
.edit-md-btn:hover{color:#4096ff;border-color:#4096ff}

.empty-hint{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#ccc}
.empty-hint .icon{font-size:48px;margin-bottom:16px}
.empty-hint p{font-size:14px}

/* Region select modal */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center}
.modal-overlay.show{display:flex}
.modal{background:#fff;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.15);width:90vw;height:85vh;display:flex;flex-direction:column;overflow:hidden}
.modal-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #e8e8e8}
.modal-header h3{font-size:15px}
.modal-header .close-btn{width:30px;height:30px;border:none;background:none;font-size:20px;cursor:pointer;border-radius:4px}
.modal-header .close-btn:hover{background:#f0f0f0}
.modal-body{flex:1;display:flex;overflow:hidden}
.modal-canvas-wrap{flex:1;position:relative;background:#1a1a2e;overflow:hidden}
.modal-canvas-wrap canvas{cursor:crosshair}
.modal-hint{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.6);color:#fff;padding:6px 16px;border-radius:20px;font-size:12px;pointer-events:none}
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
  <div class="mode-select" id="modeSelect">
    <span class="mode-select-label" id="modeLabel">多图文识别</span>
    <span class="mode-select-arrow">&#9662;</span>
    <div class="mode-select-dropdown" id="modeDropdown" onclick="event.stopPropagation()">
      <div class="mode-select-option active" data-type="text" onclick="setRecogType('text')">多图文识别</div>
      <div class="mode-select-option" data-type="table" onclick="setRecogType('table')">多表格识别</div>
    </div>
  </div>
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
    <div class="content-header" id="contentHeader">上传图片后自动识别文本</div>
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
        <canvas id="modalCanvas"></canvas>
        <div class="modal-hint">拖拽框选识别区域，右键删除，滚轮缩放，空格+拖拽平移</div>
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
      <input type="text" id="cfgUrl" placeholder="https://open.bigmodel.cn/api/paas/v4/chat/completions">
      <div class="hint">兼容 OpenAI Chat Completions 格式的 API 端点</div>

      <label>模型名称</label>
      <input type="text" id="cfgModel" placeholder="glm-4v-flash">
      <div class="preset-list" id="presetList"></div>

      <label>API 密钥</label>
      <input type="password" id="cfgKey" placeholder="留空使用默认密钥">
      <div class="hint">密钥仅存储在浏览器本地，不会上传到服务器</div>
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
let recogType = localStorage.getItem('recogType') || 'text';
// ========== Field helpers for dual mode ==========
function getText(item) { return recogType === 'table' ? (item.table_text || '') : (item.text || ''); }
function setText(item, val) { if (recogType === 'table') item.table_text = val; else item.text = val; }
function getStatus(item) { return recogType === 'table' ? (item.table_status || 'ready') : (item.status || 'ready'); }
function setStatus(item, val) { if (recogType === 'table') item.table_status = val; else item.status = val; }
function getSaveUrl() { return recogType === 'table' ? '/api/table_text/' : '/api/text/'; }


// Region modal state
let regionImgId = null;
let regionImg = null;
let regionCanvas, regionCtx;
let regionView = {x:0,y:0,scale:1,base:1};
let regions = [];
let regionDrag = {mode:'none'};
let regionSpace = false;

// ========== Mode Select ==========
function setRecogType(type) {
  recogType = type;
  ocrMode = 'ai';
  localStorage.setItem('recogType', type);
  localStorage.setItem('ocrMode', 'ai');
  const label = type === 'table' ? '多表格识别' : '多图文识别';
  document.getElementById('modeLabel').textContent = label;
  document.querySelectorAll('.mode-select-option').forEach(el => {
    el.classList.toggle('active', el.dataset.type === type);
  });
  document.getElementById('modeSelect').classList.remove('open');
  document.getElementById('btnPreview').style.display = '';
  document.getElementById('btnExport').style.display = '';
  document.getElementById('btnPreview').textContent = type === 'table' ? '预览表格' : '预览文本';
  document.getElementById('btnExport').textContent = type === 'table' ? '导出xlsx' : '导出txt';
  loadImages();
}
document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('modeSelect');
  sel.addEventListener('click', (e) => {
    e.stopPropagation();
    sel.classList.toggle('open');
  });
  document.addEventListener('click', () => sel.classList.remove('open'));
  setRecogType(recogType);
});

// ========== API ==========
async function api(path, opts={}) {
  const resp = await fetch(path, opts);
  return resp.json();
}

// ========== Init ==========
async function loadImages() {
  try {
    const data = await api('/api/images?type=' + recogType);
    if (data && !data.error) {
      imgItems = data;
      activeImgId = imgItems.length > 0 ? imgItems[0].id : null;
      render();
    }
  } catch(e) {}
}
loadImages();

// ========== Upload ==========
function addImages() {
  document.getElementById('fileInput').click();
}

async function handleFiles(e) {
  const files = e.target.files;
  if (!files.length) return;
  e.target.value = '';

  for (const file of files) {
    const fd = new FormData();
    fd.append('image', file);
    try {
      fd.append('recog_type', recogType);
      const item = await api('/api/upload', {method:'POST', body:fd});
      if (item && item.id) {
        imgItems.push(item);
        render();
        // Auto OCR after upload
        autoOCR(item.id, item.data);
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
      body: JSON.stringify({image_base64: dataURL, mode: ocrMode, recog_type: recogType, ...getAIConfigPayload()})
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
      + '<div class="card-body"><div class="thumb-wrap" style="position:relative"><img class="thumb" src="'+item.data+'" alt="'+item.name+'">'
      + '<span class="order-badge">'+(idx+1)+'</span></div>'
      + '<div class="info"><span class="name">'+shortName+'</span>'
      + '<span class="status-dot '+dotCls+'"></span></div>'
      + '<button class="del-btn" onclick="event.stopPropagation();deleteImage('+item.id+')">&times;</button>'
      + (getStatus(item) === 'recognizing'
        ? '<button class="ocr-btn" style="bottom:64px;background:rgba(255,0,0,.7)" onclick="event.stopPropagation();stopOCR('+item.id+')">停止</button>'
        : '<button class="ocr-btn" style="bottom:64px" onclick="event.stopPropagation();reOCR('+item.id+')">重新识别</button>')
      + '<button class="ocr-btn" onclick="event.stopPropagation();openRegionModal('+item.id+')">框选识别</button>'
      + '</div></div>';
  });
  html += '<div class="add-card" onclick="addImages()">+ 添加图片</div>';
  list.innerHTML = html;
}

function renderTextList() {
  const list = document.getElementById('textList');
  if (imgItems.length === 0) {
    list.innerHTML = '<div class="empty-hint"><div class="icon">&#128196;</div><p>上传图片开始识别</p></div>';
    return;
  }
  const isTable = recogType === 'table';
  let html = '';
  imgItems.forEach((item, idx) => {
    const statusTag = getStatusTag(getStatus(item));
    let contentHtml = isTable ? markdownTableToHtml(item.table_text || '', true) : escapeHtml(item.text || '');
    const noTableHint = isTable && contentHtml && !contentHtml.includes('<table') && item.table_text
      ? '<div style="padding:8px 12px;color:#fa8c16;font-size:13px;background:#fffbe6;border-radius:4px;margin:8px 0">该图片未检测到表格结构，已以文本形式展示</div>' : '';
    const contentClass = isTable ? 'text-content table-content' : 'text-content';
    const editable = isTable ? '' : ' contenteditable="true" onblur="saveText('+item.id+', this.textContent)"';
    const placeholder = isTable ? '等待表格识别...' : '等待识别...';
    const editBtn = isTable ? '' : '';
    html += '<div class="text-block" id="textBlock-'+item.id+'">'
      + '<div class="text-block-header">'
      + '<img src="'+item.data+'" alt="">'
      + '<span class="idx">图片 '+(idx+1)+'</span>'
      + '<span>'+item.name+'</span>'
      + statusTag + editBtn
      + '</div>'
      + '<div class="'+contentClass+'" data-id="'+item.id+'"'
      + ' data-placeholder="'+placeholder+'"'+editable+'>'+noTableHint+contentHtml+'</div>'
      + '</div>';
  });
  list.innerHTML = html;
}

function markdownTableToHtml(md, editable) {
  if (!md) return '';
  const lines = md.split('\n');
  let html = '';
  let tableRows = [];
  let inTable = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      if (/^\|[\s\-:|]+$/.test(trimmed)) continue;
      if (!inTable) inTable = true;
      const cells = trimmed.split('|').slice(1, -1).map(c => c.trim());
      tableRows.push(cells);
    } else {
      if (inTable) {
        html += buildTableHtml(tableRows, editable);
        tableRows = [];
        inTable = false;
      }
      if (trimmed) html += '<p style="margin:4px 0;color:#666">' + escapeHtml(trimmed) + '</p>';
    }
  }
  if (inTable && tableRows.length > 0) html += buildTableHtml(tableRows, editable);
  return html;
}

function buildTableHtml(rows, editable) {
  if (!rows.length) return '';
  const cols = Math.max(...rows.map(r => r.length));
  const ce = editable ? ' contenteditable="true"' : '';
  const blur = editable ? ' onblur="saveTableCellEdit(this)"' : '';
  let html = '<table class="ocr-table"><thead><tr>';
  for (let c = 0; c < cols; c++) html += '<th' + ce + blur + '>' + escapeHtml(rows[0][c] || '') + '</th>';
  html += '</tr></thead>';
  if (rows.length > 1) {
    html += '<tbody>';
    for (let r = 1; r < rows.length; r++) {
      html += '<tr>';
      for (let c = 0; c < cols; c++) html += '<td' + ce + blur + '>' + escapeHtml(rows[r][c] || '') + '</td>';
      html += '</tr>';
    }
    html += '</tbody>';
  }
  html += '</table>';
  return html;
}

function editRawMarkdown(id) {
  const item = imgItems.find(i => i.id === id);
  if (!item) return;
  const block = document.getElementById('textBlock-' + id);
  const contentDiv = block.querySelector('.text-content');
  contentDiv.style.display = 'none';
  const ta = document.createElement('textarea');
  ta.style.cssText = 'width:100%;min-height:200px;padding:12px;font-family:monospace;font-size:13px;border:1px solid #4096ff;border-radius:6px;resize:vertical;outline:none;line-height:1.6;box-sizing:border-box';
  ta.value = item.table_text || '';
  const actions = document.createElement('div');
  actions.style.cssText = 'display:flex;gap:8px;margin:8px 16px;justify-content:flex-end';
  actions.innerHTML = '<button class="btn" onclick="cancelEditMarkdown('+id+')">取消</button>'
    + '<button class="btn" style="background:#1677ff;color:#fff;border-color:#1677ff" onclick="saveEditMarkdown('+id+')">保存</button>';
  contentDiv.parentNode.insertBefore(ta, contentDiv.nextSibling);
  ta.parentNode.insertBefore(actions, ta.nextSibling);
  ta.focus();
}
function cancelEditMarkdown(id) { renderTextList(); }
function saveEditMarkdown(id) {
  const item = imgItems.find(i => i.id === id);
  if (!item) return;
  const block = document.getElementById('textBlock-' + id);
  const ta = block.querySelector('textarea');
  if (ta) {
    item.table_text = ta.value;
    api('/api/table_text/' + id, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text: item.table_text})});
  }
  renderTextList();
}

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
      body: JSON.stringify({image_base64: item.data, mode: ocrMode, recog_type: recogType, ...getAIConfigPayload()})
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

  const img = new Image();
  img.src = item.data;
  await new Promise(r => { img.onload = r; });

  let allText = '';
  for (const r of item.regions) {
    if (item.aborted) break;
    const sw = Math.round(r.x2 - r.x1), sh = Math.round(r.y2 - r.y1);
    if (sw <= 0 || sh <= 0) continue;
    const tmpC = document.createElement('canvas');
    tmpC.width = sw; tmpC.height = sh;
    tmpC.getContext('2d').drawImage(img, r.x1, r.y1, sw, sh, 0, 0, sw, sh);
    const b64 = tmpC.toDataURL('image/png');
    try {
      const result = await api('/api/ocr/' + id, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({image_base64: b64, mode: ocrMode, recog_type: recogType, ...getAIConfigPayload()})
      });
      if (item.aborted) break;
      if (result && !result.error && result.text) {
        if (allText) allText += '\n\n';
        allText += result.text;
      }
    } catch(e) { if (item.aborted) break; }
  }

  if (!item.aborted) {
    item.text = allText;
    item.status = 'done';
  }
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

// ========== Save Table Cell Edit ==========
function saveTableCellEdit(cell) {
  const block = cell.closest('.text-block');
  if (!block) return;
  const id = parseInt(block.id.replace('textBlock-', ''));
  // Rebuild markdown from all tables in this block
  const md = rebuildMarkdownFromDom(block);
  // Update local
  const item = imgItems.find(i => i.id === id);
  if (item) item.table_text = md;
  // Debounced save to server
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    await api('/api/table_text/' + id, {
      method:'PUT',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: md})
    });
  }, 600);
}

function rebuildMarkdownFromDom(block) {
  const tables = block.querySelectorAll('table.ocr-table');
  let md = '';
  tables.forEach((table, ti) => {
    if (ti > 0) md += '\n';
    const allRows = table.querySelectorAll('tr');
    allRows.forEach((tr, ri) => {
      const cells = tr.querySelectorAll('th, td');
      const line = '| ' + Array.from(cells).map(c => c.textContent.trim()).join(' | ') + ' |';
      md += line + '\n';
      if (ri === 0) {
        md += '| ' + Array.from(cells).map(() => '---').join(' | ') + ' |' + '\n';
      }
    });
  });
  return md.trim();
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
      body: JSON.stringify({image_base64: item.data, mode: ocrMode, recog_type: recogType, ...getAIConfigPayload()})
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
  window.open('/api/export?type=' + recogType, '_blank');
}

function exportTxt() {
  if (imgItems.length === 0) { alert('没有内容可导出'); return; }
  if (recogType === 'table') {
    window.location = '/api/export/xlsx';
    return;
  }
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
  if (!confirm('确定清空所有图片和文本？')) return;
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
  regionCanvas = document.getElementById('modalCanvas');
  regionCtx = regionCanvas.getContext('2d');
  // Restore saved regions
  regions = item.regions ? JSON.parse(JSON.stringify(item.regions)) : [];

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
    if (item) item.regions = JSON.parse(JSON.stringify(regions));
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
    const p1 = rToC(r.x1, r.y1);
    const p2 = rToC(r.x2, r.y2);
    const color = colors[i % colors.length];
    c.fillStyle = color + '25';
    c.fillRect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y);
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.strokeRect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y);
    c.font = 'bold 14px sans-serif';
    c.fillStyle = color;
    c.fillRect(p1.x, p1.y-20, 24, 18);
    c.fillStyle = '#fff';
    c.fillText('#'+(i+1), p1.x+4, p1.y-6);
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

// Modal mouse events
document.getElementById('modalCanvas').addEventListener('mousedown', e => {
  if (!regionImg) return;
  const rect = regionCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;

  if (e.button === 1 || (e.button === 0 && regionSpace)) {
    regionDrag = {mode:'pan', start:{x:cx, y:cy}, ox:regionView.x, oy:regionView.y};
    regionCanvas.style.cursor = 'grabbing';
    return;
  }
  if (e.button === 2) {
    const orig = cToR(cx, cy);
    for (let i = regions.length-1; i >= 0; i--) {
      const r = regions[i];
      if (orig.x >= r.x1 && orig.x <= r.x2 && orig.y >= r.y1 && orig.y <= r.y2) {
        regions.splice(i, 1);
        redrawRegion();
        return;
      }
    }
    return;
  }
  regionDrag = {mode:'draw', start:{x:cx, y:cy}, current:{x:cx, y:cy}};
});

document.getElementById('modalCanvas').addEventListener('mousemove', e => {
  const rect = regionCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  if (regionDrag.mode === 'pan') {
    regionView.x = regionDrag.ox + (cx - regionDrag.start.x);
    regionView.y = regionDrag.oy + (cy - regionDrag.start.y);
    redrawRegion();
  } else if (regionDrag.mode === 'draw') {
    regionDrag.current = {x:cx, y:cy};
    redrawRegion();
  }
});

document.getElementById('modalCanvas').addEventListener('mouseup', e => {
  if (regionDrag.mode === 'draw' && regionDrag.start && regionDrag.current) {
    const o1 = cToR(Math.min(regionDrag.start.x, regionDrag.current.x), Math.min(regionDrag.start.y, regionDrag.current.y));
    const o2 = cToR(Math.max(regionDrag.start.x, regionDrag.current.x), Math.max(regionDrag.start.y, regionDrag.current.y));
    const rw = o2.x - o1.x, rh = o2.y - o1.y;
    if (rw > 5 && rh > 5) {
      regions.push({x1:Math.max(0,o1.x), y1:Math.max(0,o1.y), x2:Math.min(regionImg.naturalWidth,o2.x), y2:Math.min(regionImg.naturalHeight,o2.y)});
    }
  }
  regionDrag = {mode:'none'};
  redrawRegion();
});

document.getElementById('modalCanvas').addEventListener('wheel', e => {
  e.preventDefault();
  if (!regionImg) return;
  const rect = regionCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
  const ns = Math.max(0.3, Math.min(regionView.scale * factor, 8));
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
  btn.disabled = true;
  btn.textContent = '识别中...';

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
  for (let ri = 0; ri < savedRegions.length; ri++) {
    if (savedItem && savedItem.aborted) break;
    const r = savedRegions[ri];
    const sw = Math.round(r.x2 - r.x1), sh = Math.round(r.y2 - r.y1);
    if (sw <= 0 || sh <= 0) continue;
    const tmpC = document.createElement('canvas');
    tmpC.width = sw; tmpC.height = sh;
    tmpC.getContext('2d').drawImage(savedImg, r.x1, r.y1, sw, sh, 0, 0, sw, sh);
    const b64 = tmpC.toDataURL('image/png');
    try {
      const result = await api('/api/ocr/' + savedImgId, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({image_base64: b64, mode: ocrMode, recog_type: recogType, ...getAIConfigPayload()})
      });
      if (savedItem && savedItem.aborted) break;
      if (result && !result.error && result.text) {
        if (allText) allText += '\n\n';
        allText += result.text;
      }
    } catch(e) { if (savedItem && savedItem.aborted) break; }
  }

  if (savedItem && !savedItem.aborted) {
    savedItem.text = allText;
    savedItem.status = 'done';
  }
  if (savedItem) delete savedItem.aborted;

  btn.disabled = false;
  btn.textContent = '识别选定区域';
  render();
}

// Resize observer for modal
new ResizeObserver(() => { if (regionImg) resizeRegionCanvas(); }).observe(document.getElementById('modalCanvasWrap'));

// ========== Config Modal ==========
let aiConfig = {
  model: localStorage.getItem('aiModel') || 'glm-4v-flash',
  api_url: localStorage.getItem('aiApiUrl') || 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
  api_key: localStorage.getItem('aiApiKey') || ''
};

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
  {name:'glm-4v-flash',   label:'GLM-4V Flash (免费)', url:'https://open.bigmodel.cn/api/paas/v4/chat/completions'},
  {name:'glm-4v-plus',    label:'GLM-4V Plus (付费)', url:'https://open.bigmodel.cn/api/paas/v4/chat/completions'},
  {name:'qwen-vl-max',    label:'Qwen-VL Max (通义千问)', url:'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'},
  {name:'qwen-vl-plus',   label:'Qwen-VL Plus (通义千问)', url:'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'},
  {name:'gpt-4o',         label:'GPT-4o (OpenAI)', url:'https://api.openai.com/v1/chat/completions'},
];

function openConfigModal() {
  document.getElementById('cfgUrl').value = aiConfig.api_url;
  document.getElementById('cfgModel').value = aiConfig.model;
  document.getElementById('cfgKey').value = aiConfig.api_key;
  document.getElementById('configModal').classList.add('show');
  renderPresets();
}

function closeConfigModal() {
  document.getElementById('configModal').classList.remove('show');
}

function selectPreset(p) {
  document.getElementById('cfgUrl').value = p.url;
  document.getElementById('cfgModel').value = p.name;
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
  localStorage.setItem('aiModel', aiConfig.model);
  localStorage.setItem('aiApiUrl', aiConfig.api_url);
  localStorage.setItem('aiApiKey', aiConfig.api_key);
  // Persist to server
  api('/api/config', {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
    api_key: aiConfig.api_key, api_url: aiConfig.api_url, model: aiConfig.model
  })});
  closeConfigModal();
}

function getAIConfigPayload() {
  if (ocrMode !== 'ai') return {};
  return {
    ai_config: {
      model: aiConfig.model,
      api_url: aiConfig.api_url,
      api_key: aiConfig.api_key
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
	http.HandleFunc("/api/images/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			handleDeleteImage(w, r)
		}
	})
	http.HandleFunc("/api/ocr/", handleOCR)
	http.HandleFunc("/api/text/", handleSaveText)
	http.HandleFunc("/api/table_text/", handleSaveTableText)
	http.HandleFunc("/api/export", handleExport)
	http.HandleFunc("/api/export/xlsx", handleExportXlsx)
	http.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			handleGetConfig(w, r)
		} else if r.Method == http.MethodPut {
			handleSaveConfig(w, r)
		}
	})

	port := "8082"
	fmt.Printf("多图文本识别服务: http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
