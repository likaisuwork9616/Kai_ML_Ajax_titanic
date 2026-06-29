# Titanic Survival Prediction Web App

一個結合 Flask、SQLite、Fetch/Ajax 與 Machine Learning 的 Titanic 生還預測專案。

本專案提供 Titanic 乘客資料管理、資料分析視覺化、特徵工程、多模型訓練、模型狀態查詢、單筆乘客生還預測與 CSV 批次預測功能。使用者可以在網頁上操作資料、觀察缺失值與特徵分析，並在 `/ml` 訓練頁面選擇 Logistic Regression、Random Forest 或 Gradient Boosting 進行模型比較，也可以勾選想使用的模型特徵，觀察不同模型、不同參數與不同特徵組合對 Accuracy 的影響。完成 CSV 批次預測後，可下載一般預測結果 CSV，也可以在上傳含有 `PassengerId` 的 Kaggle Titanic 測試資料時，下載符合 Kaggle submission 格式的 `submission.csv`。

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
- 使用 Titanic 原始欄位與特徵工程欄位訓練生還預測模型
- 新增 `Title`、`FamilySize`、`FamilyGroup`、`HasCabin` 等特徵
- 支援在 `/ml` 頁面選擇不同模型：Logistic Regression、Random Forest、Gradient Boosting
- 支援依照不同模型顯示對應的可調參數
- 支援在 `/ml` 頁面勾選模型訓練特徵
- 可比較不同模型、不同參數與不同特徵組合對 Accuracy 的影響
- 顯示模型類型、模型名稱、Accuracy、訓練筆數、測試筆數、原始勾選特徵與 one-hot encoding 後的實際模型欄位
- 儲存正式模型為 `models/titanic_model.joblib`
- 儲存模型資訊為 `models/model_info.json`
- 只有當新模型 Accuracy 較高時，才會取代原本模型

### 單筆生還預測

- 使用者可輸入乘客資料
- 後端載入目前正式模型
- 回傳是否生還與生還機率
- 預測時會自動對齊訓練時的 one-hot encoding 欄位

### CSV 批次預測與結果下載

- 可在 `/ml/predict` 頁面上傳 CSV 檔案進行批次預測
- 後端使用 `pandas.read_csv()` 讀取上傳檔案
- 會檢查 CSV 是否包含必要欄位：`Pclass`、`Sex`、`Age`、`SibSp`、`Parch`、`Fare`、`Embarked`
- 批次預測會共用單筆預測與訓練時的前處理邏輯，避免欄位不一致
- 前端會以表格顯示每筆資料的預測結果與生還機率
- 可下載一般預測結果 CSV，欄位包含 `row`、`prediction`、`prediction_label`、`survival_probability`
- 若上傳 CSV 含有 `PassengerId`，可額外下載 Kaggle Titanic submission 格式的 `submission.csv`，欄位為 `PassengerId`、`Survived`

### 資料分析視覺化

- 顯示整體生還率
- 顯示缺失值分析表格
- 依性別分析生還率
- 依艙等分析生還率
- 依登船港口分析生還率
- 依頭銜 `Title` 分析生還率
- 依家庭組別 `FamilyGroup` 分析生還率
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
│   ├── predict.html                # 單筆生還預測與 CSV 批次預測頁面
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
| `/ml/predict` | 單筆乘客生還預測與 CSV 批次預測頁面 |
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
| POST | `/api/ml/train` | 訓練指定模型，可接收 `model_type`、模型參數與 `selected_features`，並在 Accuracy 較高時更新正式模型 |
| GET | `/api/ml/status` | 讀取目前正式模型狀態與模型資訊 |
| POST | `/api/ml/predict` | 預測單筆乘客是否生還與生還機率 |
| POST | `/api/ml/predict-csv` | 上傳 CSV 進行批次預測，回傳每筆預測結果、生還機率，並支援前端下載一般結果與 Kaggle submission |

### 資料分析 API

| Method | API | 功能 |
|---|---|---|
| GET | `/api/analysis/summary` | 回傳整體、性別、艙等、登船港口、Title 與 FamilyGroup 的生還率分析 |
| GET | `/api/analysis/missing-values` | 回傳 Titanic 資料表各欄位缺失值數量與比例 |

---

## 特徵工程 Feature Engineering

本專案在原始 Titanic 欄位的基礎上，新增數個可解釋的特徵，讓資料分析與模型訓練更完整。

| 新特徵 | 來源欄位 | 說明 | 是否放入模型 |
|---|---|---|---|
| `Title` | `Name` | 從姓名中萃取頭銜，例如 Mr、Mrs、Miss、Master；少見頭銜合併為 Rare | 是 |
| `FamilyName` | `Name` | 從姓名中取出逗號前的姓氏，可用於家族分析 | 否 |
| `FamilySize` | `SibSp`、`Parch` | 家庭同行人數，計算方式為 `SibSp + Parch + 1` | 是 |
| `FamilyGroup` | `FamilySize` | 將家庭大小分成 Alone、SmallFamily、LargeFamily | 是 |
| `HasCabin` | `Cabin` | Cabin 有值為 1，沒有值為 0，用來表示是否有船艙紀錄 | 是 |

`FamilyName` 目前只作為分析與後續延伸使用，暫時不直接放入模型訓練。原因是姓氏類別數量太多，如果直接 one-hot encoding，會產生大量欄位，對目前專案來說較不容易解釋與維護。

資料前處理與特徵工程目前集中在後端函式中管理，訓練與預測會共用相同的前處理邏輯，避免訓練欄位與預測欄位不一致。

---

## 模型特徵選擇 Feature Selection

`/ml` 模型訓練頁面除了可以選擇模型與調整對應超參數，也支援勾選想要使用的模型特徵。前端會把模型類型以 `model_type` 傳送到後端，並把勾選結果以 `selected_features` 傳送到後端，後端再根據使用者選擇的模型與特徵進行資料前處理與模型訓練。

### 可勾選的特徵

| 類型 | 特徵 |
|---|---|
| 基本特徵 | `Pclass`、`Sex`、`Age`、`SibSp`、`Parch`、`Fare`、`Embarked` |
| 特徵工程 | `Title`、`FamilySize`、`FamilyGroup`、`HasCabin` |

目前不開放直接選擇 `Name`、`Cabin`、`FamilyName`、`Ticket`、`PassengerId` 作為模型特徵。原因是這些欄位不是原始文字類別太多，就是缺失值過多，或只是資料識別欄位，不適合直接進入目前的模型訓練流程。

### 後端安全檢查

後端會檢查前端傳入的 `selected_features`：

| 檢查項目 | 說明 |
|---|---|
| 至少選擇 1 個特徵 | 避免沒有任何欄位可以訓練 |
| 只能選擇允許清單中的特徵 | 避免使用者傳入不應該進模型的欄位 |
| 只對有被選到的類別欄位做 one-hot encoding | 例如只在選到 `Sex`、`Embarked`、`Title`、`FamilyGroup` 時才進行轉換 |
| 記錄本次勾選特徵 | `model_info.json` 會記錄 `selected_features` 與實際訓練欄位 |

### `selected_features` 範例

```json
{
  "selected_features": [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Title",
    "FamilySize",
    "FamilyGroup",
    "HasCabin"
  ]
}
```

訓練結果會同時顯示：

| 顯示項目 | 說明 |
|---|---|
| 本次勾選的原始特徵 | 使用者在 `/ml` 頁面勾選的欄位，例如 `Sex`、`Title` |
| 模型實際使用的特徵欄位 | one-hot encoding 後真正進入模型的欄位，例如 `Sex_female`、`Sex_male`、`Title_Mr` |

---

## 多模型選擇與模型比較

`/ml` 頁面支援使用者選擇不同模型進行訓練。前端會依照模型類型顯示對應的參數欄位，並將 `model_type` 與 `params` 送到後端 `/api/ml/train`。

### 支援的模型

| model_type | 模型名稱 | 說明 |
|---|---|---|
| `logistic_regression` | `LogisticRegression` | 適合作為 baseline model，訓練速度快、結果較容易解釋 |
| `random_forest` | `RandomForestClassifier` | 目前主要模型之一，可調整樹的數量與深度，適合展示非線性分類效果 |
| `gradient_boosting` | `GradientBoostingClassifier` | Boosting 類模型，屬於較進階的樹模型方法，不需要額外安裝 XGBoost |

### 各模型可調參數

| 模型 | 參數 | 說明 |
|---|---|---|
| Logistic Regression | `C` | 正則化強度，數值越小代表限制越強 |
| Logistic Regression | `max_iter` | 最大迭代次數 |
| Logistic Regression | `solver` | 最佳化方法，目前可使用 `liblinear` 或 `lbfgs` |
| Random Forest | `n_estimators` | 樹的數量 |
| Random Forest | `max_depth` | 每棵樹的最大深度 |
| Random Forest | `min_samples_split` | 節點繼續切分所需的最少資料筆數 |
| Random Forest | `min_samples_leaf` | 葉節點最少樣本數 |
| Gradient Boosting | `n_estimators` | 弱學習器數量 |
| Gradient Boosting | `learning_rate` | 學習率，控制每次 boosting 更新幅度 |
| Gradient Boosting | `max_depth` | 每棵弱學習器的最大深度 |
| 共用 | `test_size` | 測試資料比例 |
| 共用 | `random_state` | 隨機種子，方便重現結果 |

### 訓練 API 傳送格式範例

```json
{
  "model_type": "random_forest",
  "params": {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "test_size": 0.2,
    "random_state": 42
  },
  "selected_features": [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Title",
    "FamilySize",
    "FamilyGroup",
    "HasCabin"
  ]
}
```

訓練完成後，系統會在結果表格顯示本次使用的 `model_type`、模型名稱、Accuracy、模型參數與模型是否更新。正式模型資訊也會寫入 `models/model_info.json`，其中包含 `model_type`、`model`、`best_params`、`selected_features`、`features` 與 `preprocessing_info`。


---

## CSV 批次預測與 Kaggle Submission

`/ml/predict` 頁面除了支援單筆輸入，也支援 CSV 批次上傳預測。這個功能適合一次預測多位乘客，也可以用於產生 Kaggle Titanic competition 的 submission 檔案。

### 一般 CSV 批次預測格式

一般批次預測 CSV 至少需要包含以下欄位：

```csv
Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
1,female,30,0,0,100,S
3,male,22,1,0,7.25,S
2,female,45,0,1,26,C
```

上傳後，頁面會顯示每一列的預測結果與生還機率，例如：

| 列數 | 預測結果 | 生還機率 |
|---:|---|---:|
| 1 | 生還 | 63.69% |
| 2 | 未生還 | 7.83% |
| 3 | 未生還 | 37.55% |

### 一般預測結果下載

批次預測完成後，可以下載一般預測結果 CSV。下載檔案包含：

| 欄位 | 說明 |
|---|---|
| `row` | CSV 中的資料列序號 |
| `prediction` | 模型預測結果，0 代表未生還，1 代表生還 |
| `prediction_label` | 中文預測結果，生還 / 未生還 |
| `survival_probability` | 模型預測的生還機率 |

一般結果 CSV 範例：

```csv
row,prediction,prediction_label,survival_probability
1,1,生還,0.6369
2,0,未生還,0.0783
3,0,未生還,0.3755
```

### Kaggle Submission 下載

若要產生 Kaggle Titanic 可上傳的 `submission.csv`，上傳的 CSV 必須包含 `PassengerId`。建議直接使用 Kaggle Titanic competition 提供的 `test.csv` 格式：

```csv
PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
893,3,"Wilkes, Mrs. James",female,47,1,0,363272,7,,S
894,2,"Myles, Mr. Thomas",male,62,0,0,240276,9.6875,,Q
```

Kaggle submission 下載結果只會保留 Kaggle 需要的兩欄：

```csv
PassengerId,Survived
892,0
893,1
894,0
```

注意：如果上傳的 CSV 沒有 `PassengerId`，系統仍然可以進行一般批次預測，但無法產生 Kaggle submission。這是因為 Kaggle 需要用 `PassengerId` 對應每一筆測試資料。

## 機器學習流程

模型訓練流程如下：

```text
讀取 SQLite titanic 資料表
→ 萃取 Title、FamilySize、FamilyGroup、HasCabin 等新特徵
→ 讀取使用者在 `/ml` 頁面勾選的 `selected_features`
→ 選擇特徵欄位與目標欄位
→ 處理缺失值
→ 對 Sex、Embarked、Title、FamilyGroup 做 one-hot encoding
→ 切分訓練集與測試集
→ 根據 `model_type` 建立 Logistic Regression、Random Forest 或 Gradient Boosting
→ 訓練指定模型
→ 計算 Accuracy
→ 與目前正式模型 Accuracy 比較
→ 若新模型較好，更新模型檔案與模型資訊
```

目前使用的目標欄位：

| 欄位 | 說明 |
|---|---|
| `Survived` | 是否生還，0 代表未生還，1 代表生還 |

可在網頁勾選的模型特徵欄位：

| 欄位 | 說明 |
|---|---|
| `Pclass` | 艙等 |
| `Sex` | 性別 |
| `Age` | 年齡 |
| `SibSp` | 船上兄弟姊妹或配偶人數 |
| `Parch` | 船上父母或子女人數 |
| `Fare` | 票價 |
| `Embarked` | 登船港口 |
| `Title` | 從姓名萃取的頭銜 |
| `FamilySize` | 家庭同行人數，`SibSp + Parch + 1` |
| `FamilyGroup` | 家庭大小分組，Alone / SmallFamily / LargeFamily |
| `HasCabin` | 是否有 Cabin 紀錄 |

實際進入模型的欄位會依照本次勾選結果與 one-hot encoding 結果而不同。例如勾選 `Sex` 後，模型實際欄位可能會包含 `Sex_female`、`Sex_male`。

可在網頁調整的模型參數會依模型而不同：

| 模型 | 主要可調參數 |
|---|---|
| Logistic Regression | `C`、`max_iter`、`solver` |
| Random Forest | `n_estimators`、`max_depth`、`min_samples_split`、`min_samples_leaf` |
| Gradient Boosting | `n_estimators`、`learning_rate`、`max_depth` |
| 共用參數 | `test_size`、`random_state` |

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
3. 勾選想要使用的模型特徵。
4. 按下「開始訓練模型」。
5. 查看本次 Accuracy、舊模型 Accuracy、是否取代正式模型，以及本次使用的特徵組合。

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

### 4. CSV 批次預測與下載

同樣開啟預測頁面：

```text
http://127.0.0.1:5000/ml/predict
```

操作流程：

1. 準備 CSV 檔案，至少包含 `Pclass`、`Sex`、`Age`、`SibSp`、`Parch`、`Fare`、`Embarked`。
2. 在 CSV 批次預測區塊選擇檔案並上傳。
3. 頁面顯示每筆資料的預測結果與生還機率。
4. 按下「下載一般預測結果 CSV」可下載完整預測結果。
5. 如果上傳資料含有 `PassengerId`，可以按下「下載 Kaggle Submission CSV」產生 `submission.csv`。

Kaggle submission 檔案格式：

```csv
PassengerId,Survived
892,0
893,1
894,0
```

### 5. 查看資料分析

開啟資料分析頁面：

```text
http://127.0.0.1:5000/analysis
```

可查看整體生還率、缺失值分析，以及依性別、艙等、登船港口、Title、FamilyGroup 分組的生還率表格與圖表。

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
| 訓練模型 | 顯示模型類型、Accuracy、參數、原始勾選特徵、one-hot 後特徵欄位與模型比較結果 |
| 多模型訓練 | 可選擇 Logistic Regression、Random Forest、Gradient Boosting，並觀察不同模型 Accuracy 差異 |
| 依模型顯示參數 | 選擇不同模型時，只顯示該模型對應的參數欄位 |
| 自選特徵訓練 | 可勾選不同特徵組合訓練模型，並觀察 Accuracy 是否改變 |
| 未勾選任何特徵 | 前端或後端應提示至少選擇 1 個模型特徵 |
| 新模型較好 | 更新 `titanic_model.joblib` 與 `model_info.json` |
| 新模型沒有較好 | 保留原本正式模型 |
| 單筆預測 | 顯示是否生還與生還機率 |
| CSV 批次預測 | 上傳 CSV 後，表格顯示每筆預測結果與生還機率 |
| CSV 欄位缺失 | 若缺少 `Pclass`、`Sex`、`Age`、`SibSp`、`Parch`、`Fare`、`Embarked`，應回傳錯誤提示 |
| 下載一般預測結果 CSV | 可下載包含 `row`、`prediction`、`prediction_label`、`survival_probability` 的 CSV |
| 下載 Kaggle Submission CSV | 上傳資料含有 `PassengerId` 時，可下載只包含 `PassengerId`、`Survived` 的 `submission.csv` |
| 無 PassengerId 下載 Kaggle CSV | 應提示無法產生 Kaggle submission，需上傳含 `PassengerId` 的 CSV |
| 開啟 `/analysis` | 顯示缺失值、Title、FamilyGroup 等分析表格與 Chart.js 圖表 |

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
- Titanic 特徵工程：Title、FamilySize、FamilyGroup、HasCabin
- 模型特徵選擇與 `selected_features` 設計
- one-hot encoding 類別欄位轉換
- Logistic Regression、Random Forest、Gradient Boosting 多模型訓練
- `model_type` 多模型選擇 API 設計
- Accuracy 評估與模型版本保護
- joblib 模型儲存與載入
- multipart/form-data CSV 上傳與後端讀取
- 批次預測結果整理與 CSV 下載
- Kaggle Titanic submission 格式輸出
- Chart.js 資料視覺化

