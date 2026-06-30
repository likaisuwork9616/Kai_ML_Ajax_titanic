# Titanic Survival Prediction Web App

一個以 Titanic 資料集為主題的全端機器學習 Web 專案。系統整合 **Flask、SQLite、RESTful API、Fetch/Ajax、資料分析、特徵工程、模型訓練、單筆預測、CSV 批次預測與 Bootstrap 版面設計**，讓使用者可以在網頁中管理乘客資料、觀察資料分析結果、訓練模型，並預測乘客是否生還。

GitHub：<https://github.com/likaisuwork9616/Kai_ML_Ajax_titanic>

Youtube影片說明: <https://youtu.be/r_wN18dczGE>

---

## 專案特色

- 使用 Flask 建立後端頁面與 RESTful API
- 使用 SQLite 儲存 Titanic 乘客資料
- 使用 JavaScript Fetch/Ajax 完成前後端非同步互動
- 支援乘客資料 CRUD、搜尋與分頁
- 提供資料分析頁面，包含缺失值統計與生還率視覺化
- 加入 Titanic 特徵工程：`Title`、`FamilySize`、`FamilyGroup`、`HasCabin`
- 支援多模型訓練：Logistic Regression、Random Forest、Gradient Boosting
- 可在網頁調整模型參數與選擇訓練特徵
- 訓練後會儲存正式模型與模型資訊
- 只有新模型 Accuracy 較高時，才會取代目前正式模型
- 支援單筆乘客生還預測與生還機率顯示
- 支援 CSV 批次預測與結果下載
- 可輸出 Kaggle Titanic submission 格式 CSV
- 使用 `base.html` 統一導覽列、卡片、表格、按鈕與提示訊息樣式

---

## 使用技術

| 類別 | 技術 |
|---|---|
| 後端 | Flask |
| 資料庫 | SQLite |
| 前端 | HTML、CSS、JavaScript、Jinja2 Template |
| API 串接 | Fetch / Ajax |
| 資料處理 | pandas |
| 機器學習 | scikit-learn |
| 模型儲存 | joblib |
| 視覺化 | Chart.js |
| 版面設計 | Bootstrap、共用 `base.html`、Card Layout、Responsive Table |

---

## 專案結構

```text
titanic_project/
├── app.py                    # Flask 主程式、頁面 route 與 API
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
    ├── base.html             # 共用版型、導覽列與共用樣式
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

執行後會產生 `my_db.db`。

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

## 資料分析與前處理

本專案針對 Titanic 資料加入以下前處理與特徵工程：

| 項目 | 處理方式 |
|---|---|
| `Age` | 使用中位數補值 |
| `Embarked` | 使用眾數補值 |
| `Fare` | 使用中位數補值 |
| `Cabin` | 不直接補原始 Cabin，改萃取 `HasCabin` |
| `Title` | 從 `Name` 萃取 Mr、Mrs、Miss、Master、Rare 等頭銜 |
| `FamilySize` | `SibSp + Parch + 1` |
| `FamilyGroup` | 依家庭大小分成 Alone、SmallFamily、LargeFamily |
| `HasCabin` | Cabin 有值為 1，否則為 0 |



### 特徵工程欄位說明

本專案不是直接把所有原始欄位丟進模型，而是先從 Titanic 資料中萃取出較有解釋性的特徵，再讓模型進行訓練。

| 欄位名稱 | 來源欄位 | 產生方式 | 欄位類型 | 用途說明 | 是否放入模型 |
|---|---|---|---|---|---|
| `Title` | `Name` | 從姓名中擷取稱謂，例如 Mr、Mrs、Miss、Master；少見稱謂合併為 Rare | 類別特徵 | 稱謂可能反映性別、年齡、婚姻狀態或社會身份 | 是 |
| `FamilyName` | `Name` | 取姓名逗號前的姓氏，例如 `Braund, Mr. Owen Harris` 取出 `Braund` | 類別特徵 | 可用來觀察家族關係，但類別數量太多 | 否，目前保留作分析延伸 |
| `FamilySize` | `SibSp`、`Parch` | `FamilySize = SibSp + Parch + 1` | 數值特徵 | 表示乘客同行家庭人數，`+1` 代表乘客本人 | 是 |
| `FamilyGroup` | `FamilySize` | 依家庭大小分成 `Alone`、`SmallFamily`、`LargeFamily` | 類別特徵 | 將家庭大小轉成較容易解釋的分組 | 是 |
| `HasCabin` | `Cabin` | Cabin 有值為 `1`，缺失為 `0` | 數值特徵 | Cabin 缺失值很多，不直接補 Cabin 原值，而是改看是否有船艙紀錄 | 是 |

類別特徵在訓練前會使用 one-hot encoding 轉成模型可以讀取的數值欄位，例如 `Sex_female`、`Sex_male`、`Title_Mr`、`FamilyGroup_Alone` 等。數值特徵則會在缺失值處理後直接進入模型訓練。

訓練、單筆預測與 CSV 批次預測共用同一套前處理邏輯，避免訓練與預測時欄位不一致。

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

### 可選訓練特徵

```text
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked,
Title, FamilySize, FamilyGroup, HasCabin
```

### 訓練流程

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

若上傳檔案包含 `PassengerId`，也可以下載 Kaggle submission 格式：

```csv
PassengerId,Survived
892,0
893,1
894,0
```

---

## Demo 操作流程

1. 開啟首頁 `/`，查看乘客資料表格、搜尋與分頁。
2. 進入「新增乘客」，新增一筆 Titanic 乘客資料。
3. 回到首頁，編輯或刪除指定乘客資料。
4. 進入 `/analysis`，查看缺失值統計與生還率圖表。
5. 進入 `/ml`，查看目前正式模型狀態。
6. 選擇模型、調整參數、勾選特徵，按下「開始訓練模型」。
7. 查看本次 Accuracy、模型參數、使用特徵與是否更新正式模型。
8. 進入 `/ml/predict`，輸入單筆乘客資料並取得生還預測。
9. 上傳 CSV 進行批次預測。
10. 下載一般預測結果或 Kaggle submission CSV。

---

## 測試重點

| 測試項目 | 預期結果 |
|---|---|
| 乘客 CRUD | 可正常新增、查詢、編輯、刪除資料 |
| 搜尋與分頁 | 可依關鍵字搜尋，並正常切換頁面 |
| 資料分析 | 可顯示缺失值、摘要卡片與生還率圖表 |
| 模型訓練 | 可回傳 Accuracy、模型參數、特徵欄位與模型比較結果 |
| 模型狀態 | 可顯示目前是否已有正式模型 |
| 多模型切換 | 不同模型顯示對應參數並可完成訓練 |
| 單筆預測 | 可顯示生還 / 未生還與生還機率 |
| CSV 批次預測 | 可顯示每筆預測結果與機率 |
| CSV 下載 | 可下載一般預測結果與 Kaggle submission 格式 |
| RWD 版面 | 小視窗下表格可橫向捲動，畫面不嚴重跑版 |

---

## 版本控制建議

建議不要將虛擬環境、資料庫、CSV 測試檔與訓練後模型放入 Git 版本控制。

`.gitignore` 可加入：

```text
.venv/
__pycache__/
*.pyc
*.db
*.csv
models/*.joblib
models/model_info.json
```

---

## 專案學習重點

透過本專案可以練習：

- Flask route 與 RESTful API 設計
- SQLite CRUD 操作
- JavaScript Fetch/Ajax 前後端互動
- Jinja2 Template 繼承與共用版型設計
- Bootstrap 卡片、表格、表單與導覽列排版
- pandas 資料處理與 CSV 讀取
- Titanic 缺失值處理與特徵工程
- one-hot encoding 與訓練 / 預測欄位對齊
- Logistic Regression、Random Forest、Gradient Boosting 模型比較
- joblib 模型儲存與載入
- 單筆預測、CSV 批次預測與 Kaggle submission 輸出
- Chart.js 資料視覺化
