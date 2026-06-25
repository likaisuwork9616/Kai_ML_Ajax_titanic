# Titanic Flask + SQLite + Ajax + Machine Learning 專案

## 專案簡介

本專案以老師提供的 Titanic Flask + SQLite + Fetch/Ajax 範例為基礎，先建立一個可以瀏覽 Titanic 乘客資料的網頁系統，接著逐步加入機器學習功能。

目前資料來源為 `titanic.csv`，使用 `init_db.py` 將資料匯入 SQLite，產生 `my_db.db`。網站後端使用 `app.py` 啟動 Flask Server，前端使用 HTML 與 JavaScript Fetch/Ajax 呼叫後端 API。

目前已完成機器學習一鍵訓練分支，包含：後端讀取 Titanic 資料、訓練 Random Forest 模型、將訓練結果回傳到網頁顯示，並把訓練好的模型儲存成檔案。接著也已新增模型狀態查詢 API，讓前端頁面可以知道目前模型是否已經訓練完成。

目前也已完成單筆乘客生還預測功能，使用者可以進入 `/ml/predict` 頁面輸入乘客資料，前端會呼叫 `/api/ml/predict`，後端會讀取已儲存的模型並回傳是否生還與生還機率。

---

## 目前完成進度

### 1. 執行 `init_db.py`，使用 SQLite 產生 `my_db.db`

已執行老師提供的 `init_db.py`，讀取 `titanic.csv`，並使用 SQLite 產生 `my_db.db`。

執行指令：

```bash
python init_db.py
```

執行完成後會產生：

```text
my_db.db
```

資料庫中會建立 `titanic` 資料表，作為網站資料顯示與後續機器學習模型訓練的資料來源。

---

### 2. 執行 `app.py`，觀察鐵達尼號資料集在網頁的樣子

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

已新增機器學習相關頁面：

```text
templates/ml.html
```

此頁面作為「機器學習模型訓練頁面」，目前包含：

| 區塊 | 說明 |
|---|---|
| 一鍵訓練按鈕 | 按下後呼叫後端 `/api/ml/train` |
| 目前模型狀態 | 頁面開啟時自動呼叫 `/api/ml/status`，直接顯示目前是否已有訓練好的模型 |
| 訓練狀態顯示 | 顯示尚未訓練、訓練中、訓練完成或訓練失敗 |
| 訓練結果區塊 | 顯示模型名稱、資料筆數、Accuracy、使用特徵欄位 |

此頁面使用 JavaScript `fetch()` 以 Ajax 的方式呼叫後端 API，不需要重新整理整個頁面即可顯示訓練結果。頁面一打開也會自動查詢目前模型狀態，因此不需要手動輸入 `/api/ml/status` 才能確認是否已有模型。

---

### 4. 新增 `/api/ml/train` API 雛形

已在後端新增機器學習訓練 API：

```text
POST /api/ml/train
```

此 API 一開始先作為前後端溝通的簡單雛形，讓 `ml.html` 可以按下按鈕後呼叫後端。

目前此 API 已從雛形進一步完成一鍵訓練功能。

---

### 5. 一鍵訓練分支

目前已完成一鍵訓練分支的 5-1 到 5-4。

#### 5-1 後端真的讀取 Titanic 資料

後端 `/api/ml/train` 會從 SQLite 的 `titanic` 資料表讀取 Titanic 資料，並選擇適合訓練的欄位。

目前使用的目標欄位：

| 欄位 | 說明 |
|---|---|
| `Survived` | 是否生還，作為模型要預測的答案 |

目前使用的特徵欄位包含：

| 欄位 | 說明 |
|---|---|
| `Pclass` | 艙等 |
| `Sex` | 性別 |
| `Age` | 年齡 |
| `SibSp` | 船上兄弟姊妹或配偶人數 |
| `Parch` | 船上父母或子女人數 |
| `Fare` | 票價 |
| `Embarked` | 登船港口 |

資料前處理包含：

| 處理項目 | 說明 |
|---|---|
| 缺失值處理 | 補齊 `Age`、`Fare`、`Embarked` 等缺失資料 |
| 類別欄位轉換 | 使用 one-hot encoding 將 `Sex`、`Embarked` 轉成模型可讀取的數字欄位 |
| 訓練 / 測試切分 | 將資料切分成訓練資料與測試資料 |

---

#### 5-2 後端訓練一個簡單模型

目前後端使用 Random Forest 進行 Titanic 生存預測模型訓練。

目前模型：

```text
RandomForestClassifier
```

目前流程：

```text
讀取 titanic 資料表
→ 選擇特徵欄位與目標欄位
→ 處理缺失值
→ 類別欄位轉換
→ 切分訓練集與測試集
→ 訓練 Random Forest 模型
→ 計算 Accuracy
```

---

#### 5-3 後端回傳訓練結果給網頁

目前 `/api/ml/train` 訓練完成後，會將結果以 JSON 格式回傳給前端。

前端 `ml.html` 會顯示：

| 項目 | 說明 |
|---|---|
| 訓練訊息 | 例如 `training completed` |
| 模型名稱 | 例如 `RandomForestClassifier` |
| 總資料筆數 | Titanic 資料總筆數 |
| 訓練資料筆數 | 訓練集資料筆數 |
| 測試資料筆數 | 測試集資料筆數 |
| Accuracy | 模型在測試集上的準確率 |
| 使用特徵欄位 | 模型實際使用的欄位 |

目前網頁已可成功顯示訓練結果，例如：

```text
模型：RandomForestClassifier
總資料筆數：891
訓練資料筆數：712
測試資料筆數：179
Accuracy：0.8212
```

---

#### 5-4 把模型儲存成檔案

目前訓練完成後，後端會使用 `joblib` 將訓練好的模型儲存成檔案。

模型儲存位置範例：

```text
models/titanic_model.joblib
```

此檔案可作為後續預測功能使用，不需要每次預測時都重新訓練模型。

---

### 6. 新增 `/api/ml/status`，讓頁面知道模型是否訓練完成

已新增模型狀態查詢 API：

```text
GET /api/ml/status
```

此 API 會檢查 `models/` 資料夾中是否存在已訓練完成的模型檔案。

如果模型檔案存在，代表模型已經訓練完成，會回傳：

```json
{
  "trained": true,
  "message": "模型已訓練完成",
  "accuracy": 0.82,
  "best_params": {
    "n_estimators": 100,
    "max_depth": 5
  }
}
```

如果模型檔案不存在，或檔名不正確，會回傳：

```json
{
  "trained": false,
  "message": "尚未訓練模型"
}
```

目前已測試確認：當模型檔名正確時會顯示 `trained: true`，當模型檔名錯誤或找不到模型檔案時會顯示 `trained: false`。

此功能已整合到 `ml.html`。使用者只要開啟機器學習頁面：

```text
http://127.0.0.1:5000/ml
```

頁面上方的「目前模型狀態」區塊就會自動呼叫 `/api/ml/status`，直接顯示目前是否已經有訓練好的模型。若模型檔案存在，頁面會顯示模型已訓練完成；若模型檔案不存在，頁面會顯示尚未訓練模型。

---

### 7. 新增 `/ml/predict` 預測頁面

已新增單筆乘客生還預測頁面：

```text
templates/predict.html
```

使用者可以在頁面輸入以下乘客資料：

| 欄位 | 說明 |
|---|---|
| `Pclass` | 艙等 |
| `Sex` | 性別 |
| `Age` | 年齡 |
| `SibSp` | 船上兄弟姊妹或配偶人數 |
| `Parch` | 船上父母或子女人數 |
| `Fare` | 票價 |
| `Embarked` | 登船港口 |

頁面網址：

```text
http://127.0.0.1:5000/ml/predict
```

此頁面使用 JavaScript `fetch()` 將表單資料送到後端 API，不需要重新整理整個頁面，就可以顯示預測結果。

---

### 8. 新增 `/api/ml/predict`，回傳是否生還與生還機率

已新增後端預測 API：

```text
POST /api/ml/predict
```

此 API 會先檢查模型檔案是否存在。如果尚未訓練模型，會回傳錯誤訊息，提醒使用者先到 `/ml` 頁面訓練模型。

如果模型已存在，API 會：

```text
讀取前端輸入的乘客資料
→ 載入 models/titanic_model.joblib
→ 將輸入資料轉成 DataFrame
→ 使用與訓練時一致的 one-hot encoding
→ 使用 model.feature_names_in_ 對齊訓練時的欄位
→ 預測是否生還
→ 回傳生還機率
```

回傳格式範例：

```json
{
  "message": "prediction completed",
  "prediction": 1,
  "prediction_label": "生還",
  "survival_probability": 0.78
}
```

目前已測試確認：預測頁面可以順利顯示是否生還與生還機率。

---

### 9. 修正預測時特徵欄位不一致問題

在開發預測 API 時，曾遇到模型錯誤訊息：

```text
Feature names unseen at fit time:
- Embarked
- Sex

Feature names seen at fit time, yet now missing:
- Embarked_C
- Embarked_Q
- Embarked_S
- Sex_female
- Sex_male
```

原因是模型訓練時已將 `Sex`、`Embarked` 透過 one-hot encoding 轉換成：

```text
Sex_female
Sex_male
Embarked_C
Embarked_Q
Embarked_S
```

但預測時一開始直接送入原始欄位 `Sex`、`Embarked`，導致模型看到的欄位與訓練時不同。

目前已修正：

| 修正項目 | 說明 |
|---|---|
| 預測時也執行 `pd.get_dummies()` | 讓 `Sex`、`Embarked` 轉成 one-hot encoding 欄位 |
| 使用 `model.feature_names_in_` | 取得模型訓練時使用的欄位名稱 |
| 使用 `reindex(..., fill_value=0)` | 自動補齊缺少的 one-hot 欄位，並確保欄位順序一致 |

修正後，預測功能已可正常運作。

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
| 機器學習訓練頁面 | 可進入 `/ml` 頁面進行模型訓練 |
| 一鍵訓練模型 | 可按下按鈕訓練 Titanic 生存預測模型 |
| 顯示訓練結果 | 可在網頁顯示 Accuracy、資料筆數與特徵欄位 |
| 儲存模型 | 可將訓練完成的模型儲存成 `.joblib` 檔案 |
| 檢查模型狀態 | 可透過 `/api/ml/status` 判斷模型是否已訓練完成 |
| ML 頁面直接顯示模型狀態 | 開啟 `/ml` 時，頁面會自動顯示目前是否已有訓練好的模型 |
| 生還預測頁面 | 可進入 `/ml/predict` 輸入乘客資料 |
| 單筆生還預測 | 可預測乘客是否生還 |
| 顯示生還機率 | 可顯示模型預測的生還機率百分比 |

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

目前新增的機器學習 API：

| HTTP Method | API 路徑 | 功能 | 狀態 |
|---|---|---|---|
| POST | `/api/ml/train` | 啟動模型訓練，回傳訓練結果，並儲存模型 | 已完成 5-1 到 5-4 |
| GET | `/api/ml/status` | 檢查模型檔案是否存在，回傳模型是否已訓練完成 | 已完成第 6 步 |
| POST | `/api/ml/predict` | 載入已訓練模型，預測單筆乘客是否生還與生還機率 | 已完成第 8 步 |

---

## 專案結構

目前專案大致結構如下：

```text
titanic_project/
│
├── .venv/                         # Python 虛擬環境，不上傳到 Git
│
├── models/                        # 儲存訓練完成的模型檔案
│   └── titanic_model.joblib        # 訓練後產生的模型檔案
│
├── templates/                      # HTML 頁面資料夾
│   ├── index.html                  # 首頁：乘客列表、搜尋、分頁、刪除
│   ├── new.html                    # 新增乘客頁面
│   ├── edit.html                   # 編輯乘客頁面
│   ├── ml.html                     # 機器學習訓練頁面
│   └── predict.html                # 單筆乘客生還預測頁面
│
├── .gitignore                      # Git 忽略設定
├── requirements.txt                # Python 套件清單
├── app.py                          # Flask 主程式
├── init_db.py                      # 初始化 SQLite 資料庫
├── my_db.db                        # SQLite 資料庫，由 init_db.py 產生
└── titanic.csv                     # Titanic 原始資料來源
```

---

## 環境建置

本專案使用的 Python 虛擬環境名稱為：

```text
.venv
```

### 1. 建立虛擬環境

```bash
python -m venv .venv
```

### 2. 啟動虛擬環境

Windows PowerShell：

```bash
.\.venv\Scripts\Activate.ps1
```

Windows CMD：

```bash
.\.venv\Scripts\activate.bat
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 3. 安裝套件

```bash
pip install -r requirements.txt
```

---

## `requirements.txt` 說明

本專案目前使用的主要套件如下：

| 套件 | 用途 |
|---|---|
| `Flask` | 建立後端網站與 API |
| `pandas` | 讀取 CSV、處理資料表、從 SQLite 讀取資料 |
| `scikit-learn` | 訓練 Random Forest 機器學習模型 |
| `joblib` | 儲存與讀取訓練完成的模型檔案 |

SQLite 使用 Python 內建的 `sqlite3`，不需要寫進 `requirements.txt`。

---

## `.gitignore` 建議

`.gitignore` 用來設定不需要上傳到 Git 的檔案。

建議忽略的內容包含：

```text
.venv/
__pycache__/
*.pyc

*.db
*.csv

models/*.joblib
```

意思是：

| 設定 | 說明 |
|---|---|
| `.venv/` | 忽略 Python 虛擬環境資料夾 |
| `__pycache__/` | 忽略 Python 快取資料夾 |
| `*.pyc` | 忽略 Python 編譯快取檔 |
| `*.db` | 忽略 SQLite 資料庫檔案 |
| `*.csv` | 忽略 CSV 資料檔 |
| `models/*.joblib` | 忽略訓練後產生的模型檔案 |

注意：如果作業繳交時老師需要檢查資料來源，雖然 `csv` 與 `db` 可被 Git 忽略，仍要另外確認是否需要一併繳交。

---

## Titanic 資料欄位

目前 Titanic 資料表主要欄位包含：

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

## 操作流程

### 第一次執行專案

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python init_db.py
python app.py
```

接著開啟：

```text
http://127.0.0.1:5000
```

### 執行機器學習訓練

1. 啟動 Flask：

```bash
python app.py
```

2. 開啟機器學習頁面：

```text
http://127.0.0.1:5000/ml
```

3. 按下「開始訓練模型」。
4. 網頁會呼叫 `POST /api/ml/train`。
5. 後端會讀取 SQLite 的 `titanic` 資料表，訓練模型，回傳 Accuracy 與特徵欄位。
6. 訓練完成後，模型會儲存成 `.joblib` 檔案。
7. `ml.html` 會在頁面開啟時自動呼叫 `GET /api/ml/status`，因此可以直接在機器學習頁面看到目前是否已有訓練好的模型。

也可以用以下 API 單獨測試模型狀態：

```text
http://127.0.0.1:5000/api/ml/status
```

如果模型檔案存在，會顯示 `trained: true`；如果模型檔案不存在或檔名錯誤，會顯示 `trained: false`。

### 執行單筆生還預測

1. 先確認已經訓練過模型，並且存在：

```text
models/titanic_model.joblib
```

2. 啟動 Flask：

```bash
python app.py
```

3. 開啟預測頁面：

```text
http://127.0.0.1:5000/ml/predict
```

4. 輸入乘客資料，例如艙等、性別、年齡、票價與登船港口。
5. 按下「開始預測」。
6. 網頁會呼叫 `POST /api/ml/predict`。
7. 後端會載入已訓練模型，並回傳是否生還與生還機率。
8. 頁面會顯示預測結果，例如：

```text
是否生還：生還
生還機率：78.00%
```

---



## 後續待完成項目

接下來會在目前的一鍵訓練基礎上，逐步完成老師要求的其他機器學習功能。

| 項目 | 說明 | 狀態 |
|---|---|---|
| 新增 ML 頁面 | 建立機器學習相關頁面 | 已完成 |
| 一鍵訓練模型 | 從 `titanic` 資料表讀取資料並訓練模型 | 已完成 |
| 顯示訓練結果 | 在網頁上顯示 Accuracy 與特徵欄位 | 已完成 |
| 儲存訓練模型 | 將訓練完成的模型存成 `.joblib` 檔案 | 已完成 |
| 顯示模型狀態 | 新增 `/api/ml/status`，讓頁面知道模型是否已訓練 | 已完成 |
| ML 頁面自動顯示模型狀態 | `/ml` 頁面開啟時自動呼叫 `/api/ml/status` 並顯示是否已有模型 | 已完成 |
| 調整超參數 | 使用多組參數訓練並找出最佳參數 | 已完成基本版 |
| 顯示最佳超參數 | 在頁面上顯示最佳模型參數 | 已完成基本版 |
| 單筆預測 | 讓使用者輸入乘客資料並預測是否生還 | 已完成 |
| 顯示生存機率 | 顯示預測結果與生還機率 | 已完成 |

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
6. 了解如何新增機器學習訓練頁面。
7. 了解後端如何讀取資料並訓練簡單模型。
8. 了解如何把訓練結果回傳給前端顯示。
9. 了解如何使用 `joblib` 儲存訓練完成的模型。
10. 了解如何透過 `/api/ml/status` 檢查模型檔案是否存在，並讓前端知道模型是否已訓練完成。
11. 了解如何讓 `ml.html` 在頁面開啟時自動呼叫 API，並直接顯示目前是否已有訓練好的模型。
12. 了解如何建立 `/ml/predict` 預測頁面，讓使用者輸入單筆乘客資料。
13. 了解如何使用 `joblib.load()` 載入已訓練完成的模型。
14. 了解訓練資料與預測資料必須使用一致的前處理流程。
15. 了解如何使用 `model.feature_names_in_` 對齊預測時的特徵欄位。

---

## 目前進度總結

目前已完成：

1. 執行 `init_db.py`，使用 SQLite 產生 `my_db.db`。
2. 執行 `app.py`，觀察 Titanic 資料集在網頁上的樣子。
3. 新增機器學習訓練頁面 `ml.html` 的簡單雛形。
4. 新增 `/api/ml/train` 的簡單 API 雛形。
5. 完成一鍵訓練分支：
   - 5-1 後端真的讀取 Titanic 資料。
   - 5-2 後端訓練一個簡單模型。
   - 5-3 後端回傳訓練結果給網頁。
   - 5-4 把模型儲存成檔案。
6. 新增 `/api/ml/status`，讓頁面知道模型是否訓練完成。
   - 模型檔案存在時回傳 `trained: true`。
   - 模型檔案不存在或檔名錯誤時回傳 `trained: false`。
   - 可同時回傳 Accuracy 與最佳參數資訊。
7. 將模型狀態整合到 `ml.html`。
   - 開啟 `/ml` 頁面時會自動呼叫 `/api/ml/status`。
   - 頁面上方會直接顯示目前是否已有訓練好的模型。
   - 不需要手動輸入 `/api/ml/status` 才能查看模型狀態。
8. 新增 `/ml/predict` 預測頁面。
   - 使用者可以輸入乘客艙等、性別、年齡、家庭人數、票價與登船港口。
   - 頁面會使用 Fetch/Ajax 呼叫後端預測 API。
9. 新增 `/api/ml/predict`。
   - 後端會載入已儲存的模型。
   - 預測乘客是否生還。
   - 回傳生還機率。
10. 修正預測時特徵欄位不一致問題。
   - 訓練時 `Sex`、`Embarked` 有使用 one-hot encoding。
   - 預測時也要使用相同的前處理。
   - 使用 `model.feature_names_in_` 與 `reindex()` 對齊欄位。
   - 目前已可順利預測是否存活與生還機率。

下一步目標是新增 `/analysis` 資料分析視覺化頁面，或整理目前功能準備 demo。
