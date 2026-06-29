# Titanic Survival Prediction Web App

一個整合 **Flask、SQLite、RESTful API、Fetch/Ajax、資料分析與機器學習** 的 Titanic 生還預測系統。

本專案從 Titanic 乘客資料 CRUD 出發，延伸出資料分析、特徵工程、多模型訓練、模型狀態管理、單筆預測、CSV 批次預測，以及 Kaggle submission 下載功能。

GitHub：<https://github.com/likaisuwork9616/Kai_ML_Ajax_titanic>

---

## 專案亮點

- 使用 Flask 建立後端頁面與 RESTful API
- 使用 SQLite 儲存 Titanic 乘客資料
- 使用 JavaScript Fetch/Ajax 完成前後端互動
- 支援乘客資料 CRUD：查詢、新增、編輯、刪除、搜尋、分頁
- 提供資料分析頁面，包含缺失值分析與生還率視覺化
- 支援 Titanic 特徵工程：`Title`、`FamilySize`、`FamilyGroup`、`HasCabin`
- 支援多模型訓練：Logistic Regression、Random Forest、Gradient Boosting
- 可在網頁調整模型參數與勾選訓練特徵
- 訓練完成後儲存正式模型與模型資訊
- 只有新模型 Accuracy 較高時，才會取代正式模型
- 支援單筆乘客生還預測與生還機率顯示
- 支援 CSV 批次預測與結果下載
- 可輸出 Kaggle Titanic submission 格式 CSV

---

## 使用技術

| 類別 | 技術 |
|---|---|
| 後端 | Flask |
| 資料庫 | SQLite |
| 前端 | HTML、CSS、JavaScript |
| API 串接 | Fetch / Ajax |
| 資料處理 | pandas |
| 機器學習 | scikit-learn |
| 模型儲存 | joblib |
| 視覺化 | Chart.js |

---

## 專案結構

```text
titanic_project/
├── app.py                    # Flask 主程式與 API
├── init_db.py                # 初始化 SQLite 資料庫
├── titanic.csv               # Titanic 原始資料
├── my_db.db                  # SQLite 資料庫，由 init_db.py 產生
├── requirements.txt          # Python 套件清單
├── README.md
│
├── models/
│   ├── titanic_model.joblib  # 訓練後儲存的正式模型
│   └── model_info.json       # 正式模型資訊
│
└── templates/
    ├── index.html            # 乘客列表、搜尋、分頁、刪除
    ├── new.html              # 新增乘客
    ├── edit.html             # 編輯乘客
    ├── analysis.html         # 資料分析視覺化
    ├── ml.html               # 模型訓練與模型狀態
    └── predict.html          # 單筆預測與 CSV 批次預測
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

### 5. 啟動專案

```bash
python app.py
```

瀏覽器開啟：

```text
http://127.0.0.1:5000
```

---

## 主要頁面

| 頁面 | 功能 |
|---|---|
| `/` | Titanic 乘客資料列表、搜尋、分頁、刪除 |
| `/passengers/new` | 新增乘客資料 |
| `/passengers/<passenger_id>/edit` | 編輯乘客資料 |
| `/analysis` | 資料分析、缺失值統計、生還率圖表 |
| `/ml` | 模型訓練、模型狀態、模型比較 |
| `/ml/predict` | 單筆預測、CSV 批次預測、下載結果 |

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

### 資料分析 API

| Method | API | 功能 |
|---|---|---|
| GET | `/api/analysis/summary` | 回傳整體與分組生還率分析 |
| GET | `/api/analysis/missing-values` | 回傳各欄位缺失值數量與比例 |

### 機器學習 API

| Method | API | 功能 |
|---|---|---|
| POST | `/api/ml/train` | 訓練模型，支援模型選擇、參數調整與特徵選擇 |
| GET | `/api/ml/status` | 取得目前正式模型狀態 |
| POST | `/api/ml/predict` | 單筆乘客生還預測 |
| POST | `/api/ml/predict-csv` | CSV 批次預測 |

---

## 資料分析與特徵工程

本專案針對 Titanic 資料加入以下分析與前處理：

| 項目 | 說明 |
|---|---|
| 缺失值分析 | 統計各欄位缺失數量與缺失比例 |
| `Age` | 使用中位數補值 |
| `Embarked` | 使用眾數補值 |
| `Fare` | 使用中位數補值 |
| `Cabin` | 不直接補原始 Cabin，改萃取 `HasCabin` |
| `Title` | 從 `Name` 萃取 Mr、Mrs、Miss、Master、Rare 等頭銜 |
| `FamilySize` | `SibSp + Parch + 1` |
| `FamilyGroup` | 依家庭大小分成 Alone、SmallFamily、LargeFamily |
| `HasCabin` | Cabin 有值為 1，否則為 0 |

訓練、單筆預測與 CSV 批次預測共用同一套前處理邏輯，避免訓練與預測欄位不一致。

---

## 機器學習功能

### 支援模型

| model_type | 模型 |
|---|---|
| `logistic_regression` | Logistic Regression |
| `random_forest` | Random Forest Classifier |
| `gradient_boosting` | Gradient Boosting Classifier |

### 可調參數

| 模型 | 參數 |
|---|---|
| Logistic Regression | `C`、`max_iter`、`solver` |
| Random Forest | `n_estimators`、`max_depth`、`min_samples_split`、`min_samples_leaf` |
| Gradient Boosting | `n_estimators`、`learning_rate`、`max_depth` |
| 共用參數 | `test_size`、`random_state` |

### 可選模型特徵

```text
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked,
Title, FamilySize, FamilyGroup, HasCabin
```

訓練流程：

```text
讀取 SQLite titanic 資料
→ 缺失值處理
→ 特徵工程
→ 依使用者勾選的 selected_features 選取欄位
→ 類別欄位 one-hot encoding
→ 切分訓練集與測試集
→ 依 model_type 建立模型
→ 訓練模型並計算 Accuracy
→ 與舊模型 Accuracy 比較
→ 若新模型較好，更新正式模型與 model_info.json
```

---

## CSV 批次預測

`/ml/predict` 頁面支援 CSV 批次上傳。一般 CSV 至少需要包含：

```csv
Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
1,female,30,0,0,100,C
3,male,22,1,0,7.25,S
2,female,45,0,1,26,S
```

批次預測完成後，可下載一般預測結果：

```csv
row,prediction,prediction_label,survival_probability
1,1,生還,0.6369
2,0,未生還,0.0783
3,0,未生還,0.3755
```

若上傳檔案包含 `PassengerId`，可下載 Kaggle submission 格式：

```csv
PassengerId,Survived
892,0
893,1
894,0
```

---

## Demo 操作流程

1. 開啟首頁 `/`，展示乘客資料 CRUD、搜尋與分頁。
2. 進入 `/analysis`，展示缺失值分析與生還率圖表。
3. 進入 `/ml`，查看目前模型狀態。
4. 選擇模型、調整參數、勾選特徵。
5. 按下「開始訓練模型」。
6. 查看 Accuracy、模型參數、特徵欄位與是否更新正式模型。
7. 進入 `/ml/predict`，輸入單筆乘客資料進行預測。
8. 上傳 CSV 進行批次預測。
9. 下載一般預測結果或 Kaggle submission CSV。

---

## 測試重點

| 測試項目 | 預期結果 |
|---|---|
| 首頁載入 | 正常顯示 Titanic 乘客資料 |
| 搜尋乘客 | 顯示符合姓名關鍵字的資料 |
| 新增 / 編輯 / 刪除 | 資料可正常寫入、更新與刪除 |
| `/analysis` | 顯示缺失值與分組生還率 |
| `/ml` 模型狀態 | 顯示目前是否已有正式模型 |
| 訓練模型 | 回傳 Accuracy、參數、特徵與模型比較結果 |
| 多模型切換 | 不同模型顯示對應參數並可訓練 |
| 特徵勾選 | 可用不同特徵組合訓練模型 |
| 單筆預測 | 顯示生還 / 未生還與生還機率 |
| CSV 批次預測 | 顯示每筆預測結果與機率 |
| Kaggle submission | 上傳含 `PassengerId` 的 CSV 後可下載 `PassengerId,Survived` 格式 |

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

`my_db.db`、CSV 檔案與訓練後模型通常可由程式重新產生，因此不建議放入 Git 版本控制。

---

## 學習重點

- Flask route 與 RESTful API 設計
- SQLite CRUD 操作
- JavaScript Fetch/Ajax 前後端互動
- pandas 資料處理與 CSV 讀取
- Titanic 缺失值處理與特徵工程
- one-hot encoding 與訓練 / 預測欄位對齊
- Logistic Regression、Random Forest、Gradient Boosting 模型比較
- joblib 模型儲存與載入
- CSV 批次預測與 Kaggle submission 輸出
- Chart.js 資料視覺化
