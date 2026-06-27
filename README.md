# Titanic Survival Prediction Web App

一個結合 Flask、SQLite、Fetch/Ajax 與 Machine Learning 的 Titanic 生還預測專案。

本專案提供 Titanic 乘客資料管理、資料分析視覺化、模型訓練、模型狀態查詢與單筆乘客生還預測功能。使用者可以在網頁上操作資料、訓練 Random Forest 模型，並輸入乘客資訊預測是否生還與生還機率。

---

## 專案功能

### 乘客資料管理

- 查看 Titanic 乘客資料列表
- 依姓名搜尋乘客
- 分頁瀏覽資料
- 新增乘客資料
- 編輯乘客資料
- 刪除乘客資料

### 機器學習模型訓練

- 從 SQLite 的 `titanic` 資料表讀取資料
- 使用 Titanic 特徵訓練生還預測模型
- 支援在網頁調整 Random Forest 超參數
- 顯示 Accuracy、訓練筆數、測試筆數與特徵欄位
- 儲存正式模型為 `models/titanic_model.joblib`
- 儲存模型資訊為 `models/model_info.json`
- 只有當新模型 Accuracy 較高時，才會取代原本模型

### 單筆生還預測

- 使用者可輸入乘客資料
- 後端載入目前正式模型
- 回傳是否生還與生還機率
- 預測時會自動對齊訓練時的 one-hot encoding 欄位

### 資料分析視覺化

- 顯示整體生還率
- 依性別分析生還率
- 依艙等分析生還率
- 依登船港口分析生還率
- 使用表格與 Chart.js 長條圖呈現分析結果

---

## 技術使用

| 類別 | 技術 |
|---|---|
| 後端 | Flask |
| 資料庫 | SQLite |
| 前端 | HTML、CSS、JavaScript |
| API 呼叫 | Fetch / Ajax |
| 資料處理 | pandas |
| 機器學習 | scikit-learn |
| 模型儲存 | joblib |
| 視覺化 | Chart.js |

---

## 專案結構

```text
titanic_project/
│
├── models/
│   ├── titanic_model.joblib        # 訓練後儲存的正式模型
│   └── model_info.json             # 正式模型資訊
│
├── templates/
│   ├── index.html                  # 首頁：乘客列表、搜尋、分頁、刪除
│   ├── new.html                    # 新增乘客頁面
│   ├── edit.html                   # 編輯乘客頁面
│   ├── ml.html                     # 機器學習訓練頁面
│   ├── predict.html                # 單筆生還預測頁面
│   └── analysis.html               # 資料分析視覺化頁面
│
├── app.py                          # Flask 主程式與 API
├── init_db.py                      # 初始化 SQLite 資料庫
├── titanic.csv                     # Titanic 原始資料
├── my_db.db                        # SQLite 資料庫，由 init_db.py 產生
├── requirements.txt                # Python 套件清單
├── .gitignore
└── README.md
```

---

## 環境建置

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

### 4. 初始化資料庫

```bash
python init_db.py
```

執行後會產生：

```text
my_db.db
```

### 5. 啟動 Flask

```bash
python app.py
```

啟動後開啟：

```text
http://127.0.0.1:5000
```

---

## 主要頁面

| 頁面 | 功能 |
|---|---|
| `/` | Titanic 乘客資料管理首頁 |
| `/passengers/new` | 新增乘客 |
| `/passengers/<passenger_id>/edit` | 編輯乘客 |
| `/ml` | 模型訓練與模型狀態頁面 |
| `/ml/predict` | 單筆乘客生還預測頁面 |
| `/analysis` | Titanic 資料分析視覺化頁面 |

---

## RESTful API

### 乘客資料 API

| Method | API | 功能 |
|---|---|---|
| GET | `/api/passengers` | 取得乘客列表，支援搜尋與分頁 |
| GET | `/api/passengers/<passenger_id>` | 取得單一乘客資料 |
| POST | `/api/passengers` | 新增乘客 |
| PUT | `/api/passengers/<passenger_id>` | 修改乘客 |
| DELETE | `/api/passengers/<passenger_id>` | 刪除乘客 |

### 機器學習 API

| Method | API | 功能 |
|---|---|---|
| POST | `/api/ml/train` | 訓練 Random Forest 模型，並在 Accuracy 較高時更新正式模型 |
| GET | `/api/ml/status` | 讀取目前正式模型狀態與模型資訊 |
| POST | `/api/ml/predict` | 預測單筆乘客是否生還與生還機率 |

### 資料分析 API

| Method | API | 功能 |
|---|---|---|
| GET | `/api/analysis/summary` | 回傳整體、性別、艙等與登船港口的生還率分析 |

---

## 機器學習流程

模型訓練流程如下：

```text
讀取 SQLite titanic 資料表
→ 選擇特徵欄位與目標欄位
→ 處理缺失值
→ 對 Sex、Embarked 做 one-hot encoding
→ 切分訓練集與測試集
→ 訓練 RandomForestClassifier
→ 計算 Accuracy
→ 與目前正式模型 Accuracy 比較
→ 若新模型較好，更新模型檔案與模型資訊
```

目前使用的目標欄位：

| 欄位 | 說明 |
|---|---|
| `Survived` | 是否生還，0 代表未生還，1 代表生還 |

目前使用的特徵欄位：

| 欄位 | 說明 |
|---|---|
| `Pclass` | 艙等 |
| `Sex` | 性別 |
| `Age` | 年齡 |
| `SibSp` | 船上兄弟姊妹或配偶人數 |
| `Parch` | 船上父母或子女人數 |
| `Fare` | 票價 |
| `Embarked` | 登船港口 |

可在網頁調整的模型參數：

| 參數 | 說明 |
|---|---|
| `n_estimators` | 隨機森林中樹的數量 |
| `max_depth` | 每棵決策樹的最大深度 |
| `min_samples_split` | 節點繼續切分所需的最少資料筆數 |
| `test_size` | 測試資料比例 |
| `random_state` | 隨機種子，方便重現結果 |

---

## Demo 操作流程

### 1. 查看乘客資料

開啟首頁：

```text
http://127.0.0.1:5000
```

可展示乘客列表、搜尋、分頁、新增、編輯與刪除功能。

### 2. 訓練模型

開啟模型訓練頁面：

```text
http://127.0.0.1:5000/ml
```

操作流程：

1. 查看目前模型狀態。
2. 調整 Random Forest 超參數。
3. 按下「開始訓練模型」。
4. 查看本次 Accuracy、舊模型 Accuracy 與是否取代正式模型。

### 3. 單筆預測

開啟預測頁面：

```text
http://127.0.0.1:5000/ml/predict
```

輸入乘客資料後，系統會回傳：

```text
是否生還
生還機率
```

範例測試資料：

| 測試資料 | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked | 預期觀察 |
|---|---:|---|---:|---:|---:|---:|---|---|
| A | 1 | female | 30 | 0 | 0 | 100 | C | 通常生還機率較高 |
| B | 3 | male | 30 | 0 | 0 | 8 | S | 通常生還機率較低 |

實際結果會受到訓練參數、資料切分與目前正式模型影響。

### 4. 查看資料分析

開啟資料分析頁面：

```text
http://127.0.0.1:5000/analysis
```

可查看整體生還率，以及依性別、艙等、登船港口分組的生還率表格與圖表。

---

## 測試重點

| 測試項目 | 預期結果 |
|---|---|
| 首頁載入乘客資料 | 正常顯示資料表與分頁 |
| 搜尋姓名 | 顯示符合關鍵字的乘客 |
| 新增乘客 | 資料新增成功並回到首頁 |
| 編輯乘客 | 資料更新成功 |
| 刪除乘客 | 資料刪除成功並重新載入列表 |
| 開啟 `/ml` | 顯示目前模型狀態 |
| 訓練模型 | 顯示 Accuracy、參數、特徵欄位與模型比較結果 |
| 新模型較好 | 更新 `titanic_model.joblib` 與 `model_info.json` |
| 新模型沒有較好 | 保留原本正式模型 |
| 單筆預測 | 顯示是否生還與生還機率 |
| 開啟 `/analysis` | 顯示分析表格與 Chart.js 圖表 |

---

## `.gitignore` 建議

```text
.venv/
__pycache__/
*.pyc

*.db
*.csv

models/*.joblib
models/model_info.json
```

資料庫、CSV 與模型檔案通常可由程式重新產生，因此可以不放入 Git 版本控制。

---

## 學習重點

- Flask route 與 RESTful API 設計
- SQLite 資料庫 CRUD 操作
- JavaScript Fetch/Ajax 前後端資料交換
- pandas 資料讀取與前處理
- one-hot encoding 類別欄位轉換
- Random Forest 分類模型訓練
- Accuracy 評估與模型版本保護
- joblib 模型儲存與載入
- Chart.js 資料視覺化

