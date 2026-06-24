# Titanic Flask + SQLite + Ajax + Machine Learning 專案

## 專案簡介

本專案以老師提供的 Titanic Flask + SQLite + Fetch/Ajax 範例為基礎，先建立一個可以瀏覽 Titanic 乘客資料的網頁系統，接著逐步加入機器學習功能。

目前資料來源為 `titanic.csv`，使用 `init_db.py` 將資料匯入 SQLite，產生 `my_db.db`。  
網站後端使用 `app.py` 啟動 Flask Server，前端使用 HTML 與 JavaScript Fetch/Ajax 呼叫後端 API。

目前已開始新增機器學習相關功能，但模型訓練功能仍是簡單雛形，尚未進入正式訓練階段。

---

## 目前完成進度

### 1. 使用 `init_db.py` 建立 SQLite 資料庫

已執行老師提供的 `init_db.py`，讀取 `titanic.csv`，並使用 SQLite 產生 `my_db.db`。

執行指令：

```bash
python init_db.py
```

執行完成後，會產生：

```text
my_db.db
```

資料庫中會建立 `titanic` 資料表，作為網站資料顯示與後續機器學習模型訓練的資料來源。

---

### 2. 使用 `app.py` 啟動 Flask 網站

已執行老師提供的 `app.py`，觀察 Titanic 資料集在網頁中的呈現方式。

執行指令：

```bash
python app.py
```

啟動後，在瀏覽器開啟：

```text
http://127.0.0.1:5000
```

即可看到 Titanic 乘客資料管理頁面。

目前首頁可以觀察 Titanic 乘客資料，並支援基本的查詢、分頁、新增、編輯與刪除功能。

---

### 3. 新增機器學習訓練頁面雛形：`ml.html`

已新增機器學習相關頁面雛形：

```text
templates/ml.html
```

此頁面目前作為「機器學習模型訓練頁面」的初始版本，主要目標是先建立前端頁面結構。

目前此頁面預計用來放置：

| 區塊 | 說明 |
|---|---|
| 模型訓練按鈕 | 之後可一鍵啟動模型訓練 |
| 訓練狀態顯示 | 顯示模型是否已開始訓練或訓練完成 |
| 訓練結果區塊 | 之後顯示模型準確率、最佳超參數等資訊 |

目前狀態：已建立簡單雛形，尚未完成正式機器學習訓練流程。

---

### 4. 新增 `/api/ml/train` API 雛形

已在後端新增機器學習訓練 API 的簡單雛形：

```text
/api/ml/train
```

此 API 目前的目標是先建立前端與後端溝通的基礎流程。

目前狀態：

| 項目 | 狀態 |
|---|---|
| API 路徑 | 已新增 |
| 可由前端呼叫 | 規劃中 / 初步完成 |
| 讀取 Titanic 資料表 | 尚未正式完成 |
| 正式模型訓練 | 尚未完成 |
| 超參數調整 | 尚未完成 |
| 儲存模型 | 尚未完成 |
| 回傳最佳超參數 | 尚未完成 |

目前此 API 只是雛形，還沒有真正進行機器學習模型訓練。

---

## 目前網站功能

目前原本的 Titanic 乘客資料管理功能包含：

| 功能 | 說明 |
|---|---|
| 查看乘客列表 | 首頁顯示 Titanic 乘客資料 |
| 分頁瀏覽 | 每頁顯示固定筆數資料 |
| 姓名搜尋 | 可依乘客姓名查詢 |
| 新增乘客 | 可新增一筆乘客資料 |
| 編輯乘客 | 可修改既有乘客資料 |
| 刪除乘客 | 可刪除指定乘客資料 |

---

## 目前 RESTful API

目前後端 `app.py` 已提供以下乘客資料 CRUD API：

| HTTP Method | API 路徑 | 功能 |
|---|---|---|
| GET | `/api/passengers` | 取得乘客列表 |
| GET | `/api/passengers/<passenger_id>` | 取得單一乘客 |
| POST | `/api/passengers` | 新增乘客 |
| PUT | `/api/passengers/<passenger_id>` | 修改乘客 |
| DELETE | `/api/passengers/<passenger_id>` | 刪除乘客 |

目前新增中的機器學習 API：

| HTTP Method | API 路徑 | 功能 | 狀態 |
|---|---|---|---|
| POST | `/api/ml/train` | 啟動模型訓練 | 雛形，尚未正式訓練 |

---

## 專案結構

目前專案大致結構如下：

```text
titanic_project/
│
├── .titanic/              # Python 虛擬環境，不上傳到 Git
│
├── templates/             # HTML 頁面資料夾
│   ├── index.html         # 首頁：乘客列表、搜尋、分頁、刪除
│   ├── new.html           # 新增乘客頁面
│   ├── edit.html          # 編輯乘客頁面
│   └── ml.html            # 機器學習訓練頁面雛形
│
├── .gitignore             # Git 忽略設定
├── app.py                 # Flask 主程式
├── init_db.py             # 初始化 SQLite 資料庫
├── my_db.db               # SQLite 資料庫，由 init_db.py 產生
└── titanic.csv            # Titanic 原始資料來源
```

---

## 虛擬環境說明

`.titanic/` 是本專案使用的 Python 虛擬環境資料夾。

虛擬環境的用途是讓此專案使用獨立的 Python 套件環境，避免影響電腦上的其他 Python 專案。

建立虛擬環境範例：

```bash
python -m venv .titanic
```

啟動虛擬環境範例：

Windows PowerShell：

```bash
.\.titanic\Scripts\Activate.ps1
```

安裝目前專案需要的套件：

```bash
pip install flask pandas
```

後續加入機器學習模型訓練時，可能還需要安裝：

```bash
pip install scikit-learn joblib
```

---

## `.gitignore` 說明

`.gitignore` 用來設定不需要上傳到 Git 的檔案。

目前預計忽略的內容包含：

```text
*.db
*.csv
*.txt

.titanic/
```

意思是：

| 設定 | 說明 |
|---|---|
| `*.db` | 忽略 SQLite 資料庫檔案 |
| `*.csv` | 忽略 CSV 資料檔 |
| `*.txt` | 忽略文字檔 |
| `.titanic/` | 忽略 Python 虛擬環境資料夾 |

注意：如果作業繳交時老師需要檢查資料來源，雖然 `csv` 與 `db` 被 Git 忽略，仍要另外確認是否需要一併繳交。

---

## Titanic 資料欄位

目前主要使用的 Titanic 欄位包含：

| 欄位 | 說明 |
|---|---|
| PassengerId | 乘客編號 |
| Survived | 是否生還，0 代表未生還，1 代表生還 |
| Pclass | 艙等，1、2、3 |
| Name | 乘客姓名 |
| Sex | 性別 |
| Age | 年齡 |
| SibSp | 船上兄弟姊妹或配偶人數 |
| Parch | 船上父母或子女人數 |
| Ticket | 船票號碼 |
| Fare | 票價 |
| Cabin | 客艙 |
| Embarked | 登船港口 |

---

## 後續待完成項目

接下來會在目前的雛形基礎上，逐步完成老師要求的機器學習功能。

### 必做功能

| 項目 | 說明 | 狀態 |
|---|---|---|
| 新增 ML 頁面 | 建立機器學習相關頁面 | 已建立簡單雛形 |
| 一鍵訓練模型 | 從 `titanic` 資料表讀取資料並訓練模型 | 尚未完成 |
| 調整超參數 | 使用多組參數訓練並找出最佳參數 | 尚未完成 |
| 顯示最佳超參數 | 在頁面上顯示最佳模型參數 | 尚未完成 |
| 顯示訓練完成狀態 | 讓使用者知道模型是否訓練完成 | 尚未完成 |
| 儲存訓練模型 | 將訓練完成的模型存成檔案 | 尚未完成 |
| 單筆預測 | 讓使用者輸入乘客資料並預測是否生還 | 尚未完成 |
| 顯示生存機率 | 顯示預測結果與生還機率 | 尚未完成 |

---

## 後續可能新增的頁面

| 頁面 | 功能 |
|---|---|
| `/ml` | 機器學習模型訓練頁面 |
| `/ml/predict` | 單筆乘客生還預測頁面 |
| `/analysis` | Titanic 資料分析視覺化頁面 |

---

## 學習目標

本專案目前的學習重點：

1. 了解 CSV 資料如何匯入 SQLite。
2. 了解 Flask 如何連接 SQLite。
3. 了解 RESTful API 的基本設計方式。
4. 了解前端如何使用 Fetch/Ajax 呼叫後端 API。
5. 了解如何在網頁上呈現資料庫內容。
6. 開始建立機器學習頁面與 API 的雛形。
7. 為後續機器學習模型訓練、超參數調整、模型儲存與預測功能打基礎。

---

## 目前進度總結

目前已完成老師範例的基礎執行流程：

1. 執行 `init_db.py`，使用 SQLite 產生 `my_db.db`。
2. 執行 `app.py`，觀察 Titanic 資料集在網頁上的樣子。
3. 新增機器學習訓練頁面 `ml.html` 的簡單雛形。
4. 新增 `/api/ml/train` 的簡單 API 雛形，但尚未正式進行模型訓練。

下一步目標是讓 `/api/ml/train` 真正讀取 `titanic` 資料表，並開始進行機器學習模型訓練。
