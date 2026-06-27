import os
import sqlite3
import json
import joblib
import pandas as pd
from datetime import datetime

from flask import Flask, jsonify, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ============================================================
# 1. 全域讀取資料庫
# ============================================================

DATABASE = "my_db.db"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.joblib")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# 這裡我們直接在全域讀取資料庫，這樣在每個 route 就可以直接使用 db 來存取資料庫了。
db = sqlite3.connect(DATABASE, check_same_thread=False)

# 讓我們在讀取資料庫時，可以直接用 row["欄位名稱"] 的方式來存取資料，
# 而不是 row[0]、row[1] 這樣的 index。
db.row_factory = sqlite3.Row


# ============================================================
# 2. 小工具：把 SQLite Row 轉成 dict
# load_model_info()
# → 讀取 models/model_info.json
# → 如果檔案不存在，就回傳 None

# save_model_info(info)
# → 把模型資訊寫入 models/model_info.json
# ============================================================

def row_to_dict(row):
    return dict(row)

def load_model_info():
    if not os.path.exists(MODEL_INFO_PATH):
        return None

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_info(info):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


# ============================================================
# 2-1. 小工具：Titanic 資料前處理
# ============================================================

def preprocess_titanic_train_data(df):
    """
    訓練模型用的資料前處理函式。

    目的：
    1. 分離 y = Survived
    2. 處理缺失值
    3. 將文字欄位轉成數字欄位
    4. 回傳 X, y
    """

    # 目標欄位 y：是否生還
    y = df["Survived"]

    # 特徵欄位 X：拿掉 Survived
    X = df.drop(columns=["Survived"])

    # ----------------------------
    # 缺失值處理
    # ----------------------------

    # Age：用中位數補值
    X["Age"] = X["Age"].fillna(X["Age"].median())

    # Fare：用中位數補值
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    # Embarked：用眾數補值
    if X["Embarked"].isna().sum() > 0:
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # ----------------------------
    # 類別欄位轉換
    # ----------------------------

    # Sex、Embarked 是文字，模型不能直接吃，所以要 one-hot encoding
    X = pd.get_dummies(X, columns=["Sex", "Embarked"])

    return X, y


def preprocess_titanic_predict_data(input_df, model_features):
    """
    預測用的資料前處理函式。

    目的：
    1. 處理單筆預測資料
    2. 做 one-hot encoding
    3. 對齊訓練模型時的欄位順序
    """

    # Age：如果使用者沒有輸入，先用 0 補，之後可以再優化成中位數
    input_df["Age"] = input_df["Age"].fillna(0)

    # Fare：如果使用者沒有輸入，先用 0 補
    input_df["Fare"] = input_df["Fare"].fillna(0)

    # Embarked：如果使用者沒有輸入，先用 S 補
    input_df["Embarked"] = input_df["Embarked"].fillna("S")

    # 做 one-hot encoding
    input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"])

    # 對齊模型訓練時的欄位
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    return input_df        


# ============================================================
# 3. 前端頁面 Routes
# ============================================================

# 首頁
@app.route("/")
def index_page():
    return render_template("index.html")

# 新增乘客頁面
@app.route("/passengers/new")
def new_passenger_page():
    return render_template("new.html")

# 編輯乘客頁面
@app.route("/passengers/<int:passenger_id>/edit")
def edit_passenger_page(passenger_id):
    return render_template("edit.html", passenger_id=passenger_id)

# 機器學習頁面
@app.route("/ml")
def machine_learing_titanic():
    return render_template("ml.html")

# 預測頁面
@app.route('/ml/predict')
def ml_predict_page():
    return render_template("predict.html")

# 資料分析視覺化頁面
@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")

# ============================================================
# 4. API：取得全部乘客資料，包含簡單分頁
# GET /api/passengers?page=1&per_page=20
# ============================================================

@app.route("/api/passengers", methods=["GET"])
def get_passengers():
    # 讀取 query string 的 page 和 per_page 參數，並設定預設值
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    # 搜尋姓名
    search = request.args.get("search", "")

    # 計算 SQL 查詢的 offset，用於分頁
    offset = (page - 1) * per_page

    # 根據是否有搜尋關鍵字，執行不同的 SQL 查詢
    if search != "":
        # 有輸入搜尋關鍵字：只查詢姓名符合的資料
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            WHERE Name LIKE ?
            """,
            (f"%{search}%",)
        ).fetchone()

        rows = db.execute(
            """
            SELECT *
            FROM titanic
            WHERE Name LIKE ?
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (f"%{search}%", per_page, offset)
        ).fetchall()

    else:
        # 沒有輸入搜尋關鍵字：查詢全部資料
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            """
        ).fetchone()

        # 根據 page 和 per_page 的值，從資料庫查詢對應的資料列，
        # 並按照 PassengerId 排序。
        rows = db.execute(
            """
            SELECT *
            FROM titanic
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (per_page, offset)
        ).fetchall()

    # 總共有多少筆資料
    total = total_row["total"]

    # 最後回傳 JSON 格式的資料，包含 items（資料列表）、page、per_page 和 total。
    return jsonify({
        "message": "ok",
        "items": [row_to_dict(row) for row in rows],
        "page": page,
        "per_page": per_page,
        "total": total
    }), 200


# ============================================================
# 5. API：取得單一乘客
# GET /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["GET"])
def get_passenger(passenger_id):
    # 根據 passenger_id 查詢資料庫，看看有沒有這個乘客的資料。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    # 如果 row 是 None，代表資料庫裡沒有這個 passenger_id 的資料，我們就回傳 404 Not Found 的錯誤訊息。
    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    # 如果有找到資料，我們就把這筆資料轉成 dict，然後回傳 JSON 格式的資料。
    return jsonify({
        "message": "ok", 
        "item": row_to_dict(row)}
    ), 200


# ============================================================
# 6. API：新增乘客
# POST /api/passengers
# ============================================================

@app.route("/api/passengers", methods=["POST"])
def create_passenger():
    # 從 request 的 JSON body 讀取資料
    data = request.get_json()

    # 執行 SQL INSERT 語句，把新的乘客資料新增到 titanic 資料表中。
    cursor = db.execute(
        """
        INSERT INTO titanic (
            Survived, Pclass, Name, Sex, Age,
            SibSp, Parch, Ticket, Fare, Cabin,
            Embarked
        )
        VALUES (
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?
        )
        """,
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"]
        )
    )

    # 執行 commit()，把剛剛的 INSERT 操作真正寫入資料庫。
    db.commit()

    # cursor.lastrowid 會回傳剛剛 INSERT 的那筆資料的自動增加的 ID，
    # 也就是 PassengerId。
    new_id = cursor.lastrowid

    # 根據 new_id 查詢剛剛新增的那筆資料，這樣我們就可以把完整的資料回傳給前端了。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (new_id,)
    ).fetchone()

    # 最後回傳 JSON 格式的資料，包含 message 和 item（剛剛新增的那筆資料）。
    return jsonify({
        "message": "created",
        "item": row_to_dict(row)
    }), 201


# ============================================================
# 7. API：修改乘客
# PUT /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["PUT"])
def update_passenger(passenger_id):
    # 從 request 的 JSON body 讀取資料
    data = request.get_json()

    # 執行 SQL UPDATE 語句，根據 passenger_id 把對應的資料更新成新的值。
    cursor = db.execute(
        """
        UPDATE titanic
        SET
            Survived = ?,
            Pclass = ?,
            Name = ?,
            Sex = ?,
            Age = ?,
            SibSp = ?,
            Parch = ?,
            Ticket = ?,
            Fare = ?,
            Cabin = ?,
            Embarked = ?
        WHERE PassengerId = ?
        """,
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"],
            passenger_id
        )
    )

    # 執行 commit()，把剛剛的 UPDATE 操作真正寫入資料庫。
    db.commit()

    # 如果沒有更新任何資料，則回傳 404 Not Found 的錯誤訊息。
    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    # 根據 passenger_id 查詢剛剛更新的那筆資料，這樣我們就可以把完整的資料回傳給前端了。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    # 如果 row 是 None，代表資料庫裡沒有這個 passenger_id 的資料，我們就回傳 404 Not Found 的錯誤訊息。
    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    # 最後回傳 JSON 格式的資料，包含 message 和 item（剛剛更新的那筆資料）。
    return jsonify({
        "message": "updated",
        "item": row_to_dict(row)
    }), 200


# ============================================================
# 8. API：刪除乘客
# DELETE /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["DELETE"])
def delete_passenger(passenger_id):
    # 執行 SQL DELETE 語句，根據 passenger_id 把對應的資料從 titanic 資料表中刪除。
    cursor = db.execute(
        "DELETE FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    )

    # 執行 commit()，把剛剛的 DELETE 操作真正寫入資料庫。
    db.commit()

    # 如果沒有刪除任何資料，則回傳 404 Not Found 的錯誤訊息。
    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    # 最後回傳 JSON 格式的資料，包含 message，告訴前端這筆資料已經被刪除了。
    return jsonify({
        "message": "deleted"
    }), 200 # 你也可以設定 204，但不會有 response body，前端無法判斷成功還是失敗

# ============================================================
# 9. API：一鍵訓練 Titanic 生存預測模型
# POST /api/ml/train
# ============================================================

@app.route("/api/ml/train", methods=["POST"])
def train_titanic_model():
    try:
        # 讀取前端從 ml.html 傳來的超參數
        params = request.get_json() or {}

        n_estimators = int(params.get("n_estimators", 100))
        max_depth = int(params.get("max_depth", 5))
        min_samples_split = int(params.get("min_samples_split", 2))
        test_size = float(params.get("test_size", 0.2))
        random_state = int(params.get("random_state", 42))

        # 5-1：後端真的從 SQLite 的 titanic 資料表讀取資料
        df = pd.read_sql_query(
            """
            SELECT 
                Survived,
                Pclass,
                Sex,
                Age,
                SibSp,
                Parch,
                Fare,
                Embarked
            FROM titanic
            """,
            db
        )

        # 檢查資料是否成功讀取
        if df.empty:
            return jsonify({
                "error": "titanic 資料表沒有資料，請先執行 init_db.py"
            }), 400

        # 資料前處理：處理缺失值、類別欄位轉換
        X, y = preprocess_titanic_train_data(df)

        # 切分訓練集與測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # 5-2：訓練一個簡單的隨機森林模型
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # 用測試資料進行預測
        # 用測試資料進行預測
        y_pred = model.predict(X_test)

        # 計算準確率
        accuracy = accuracy_score(y_test, y_pred)

        # 讀取舊的模型資訊，用來比較新舊 accuracy
        old_model_info = load_model_info()

        old_accuracy = None

        if old_model_info is not None:
            old_accuracy = old_model_info.get("accuracy")

        # 判斷本次模型是否比舊模型好
        # 如果還沒有舊模型，就視為本次模型可以成為目前模型
        if old_accuracy is None:
            is_better_model = True
        else:
            is_better_model = accuracy > old_accuracy

        # 產生比較訊息
        if old_accuracy is None:
            compare_message = "目前尚無舊模型，本次模型可作為第一個模型"
        elif is_better_model:
            compare_message = "本次模型準確率較高，可以取代舊模型"
        else:
            compare_message = "本次模型準確率沒有比較高，暫時不取代舊模型"

        # 建立模型資訊
        model_info = {
            "trained": True,
            "message": "模型已訓練完成",
            "model": "RandomForestClassifier",
            "accuracy": round(accuracy, 4),
            "best_params": {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "test_size": test_size,
                "random_state": model.random_state
            },
            "features": list(X.columns),
            "rows": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": MODEL_PATH,
            "old_accuracy": round(old_accuracy, 4) if old_accuracy is not None else None,
            "new_accuracy": round(accuracy, 4),
            "is_better_model": is_better_model,
            "model_updated": True,
            "compare_message": compare_message,
            "update_message": "本次模型已取代目前模型"
        }

        # 只有當本次模型比舊模型好，或目前沒有舊模型時，才更新正式模型檔案
        if is_better_model:
            os.makedirs(MODEL_DIR, exist_ok=True)

            # 覆蓋正式模型
            joblib.dump(model, MODEL_PATH)

            # 覆蓋正式模型資訊
            save_model_info(model_info)

            model_updated = True
            update_message = "本次模型已取代目前模型"
        else:
            model_updated = False
            update_message = "本次模型未取代目前模型，仍保留原本模型"

        # 回傳訓練結果給前端
        return jsonify({
            "message": "training completed",
            "model": model_info["model"],
            "rows": model_info["rows"],
            "features": model_info["features"],
            "train_size": model_info["train_size"],
            "test_size": model_info["test_size"],
            "accuracy": model_info["accuracy"],
            "best_params": model_info["best_params"],
            "trained_at": model_info["trained_at"],
            "model_path": model_info["model_path"],

            # 新舊模型 accuracy 比較結果
            "old_accuracy": round(old_accuracy, 4) if old_accuracy is not None else None,
            "new_accuracy": round(accuracy, 4),
            "is_better_model": is_better_model,
            "compare_message": compare_message,

            # 8-5 新增：是否真的覆蓋正式模型
            "model_updated": model_updated,
            "update_message": update_message
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# 10. 檢查模型檔案與模型資訊
# GET /api/ml/status
# ============================================================  
@app.route('/api/ml/status')
def check_model_status():
    # 1. 檢查模型檔案是否存在
    model_exists = os.path.exists(MODEL_PATH)

    # 2. 讀取模型資訊檔
    model_info = load_model_info()

    # 3. 如果模型檔案或模型資訊檔其中一個不存在，就視為尚未訓練完成
    if not model_exists or model_info is None:
        return jsonify({
            "trained": False,
            "message": "尚未訓練模型",
            "model_exists": model_exists,
            "model_info_exists": model_info is not None
        }), 200

    # 4. 如果都有存在，就回傳 model_info.json 裡面的資訊
    return jsonify({
        "trained": True,
        "message": model_info.get("message", "模型已訓練完成"),
        "model": model_info.get("model"),
        "accuracy": model_info.get("accuracy"),
        "best_params": model_info.get("best_params"),
        "features": model_info.get("features"),
        "rows": model_info.get("rows"),
        "train_size": model_info.get("train_size"),
        "test_size": model_info.get("test_size"),
        "trained_at": model_info.get("trained_at"),
        "model_path": model_info.get("model_path")
    }), 200

# ============================================================
# 11. 預測乘客是否存活
# ============================================================
@app.route("/api/ml/predict", methods=["POST"])
def predict_survival():
    # 1. 先確認模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "尚未訓練模型，請先到機器學習頁面訓練模型"
        }), 400

    # 2. 讀取前端送來的 JSON
    data = request.get_json()

    # 3. 基本檢查：確認必要欄位都有傳進來
    required_fields = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({
                "error": f"缺少欄位：{field}"
            }), 400

    # 4. 載入已訓練好的模型
    model = joblib.load(MODEL_PATH)

       # 5. 整理成 DataFrame
    input_df = pd.DataFrame([{
        "Pclass": int(data["Pclass"]),
        "Sex": data["Sex"],
        "Age": float(data["Age"]),
        "SibSp": int(data["SibSp"]),
        "Parch": int(data["Parch"]),
        "Fare": float(data["Fare"]),
        "Embarked": data["Embarked"]
    }])

    # 6. 預測資料也要做跟訓練時一樣的前處理
    input_df = preprocess_titanic_predict_data(
        input_df,
        model.feature_names_in_
    )

    # 7. 預測結果
    prediction = int(model.predict(input_df)[0])

    # 8. 預測生還機率
    # predict_proba 會回傳 [[未生還機率, 生還機率]]
    probabilities = model.predict_proba(input_df)[0]
    survival_probability = float(probabilities[1])

    # 9. 回傳給前端
    return jsonify({
        "message": "prediction completed",
        "prediction": prediction,
        "prediction_label": "生還" if prediction == 1 else "未生還",
        "survival_probability": survival_probability
    }), 200

# ============================================================
# 12. API：Titanic 資料分析摘要
# GET /api/analysis/summary
# ============================================================

@app.route("/api/analysis/summary", methods=["GET"])
def get_analysis_summary():
    try:
        # 1. 整體生還統計
        overall_row = db.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(Survived) AS survived
            FROM titanic
            """
        ).fetchone()

        total = overall_row["total"]
        survived = overall_row["survived"] or 0
        not_survived = total - survived
        survival_rate = survived / total if total > 0 else 0

        # 2. 依性別計算生還率
        sex_rows = db.execute(
            """
            SELECT
                Sex AS group_name,
                COUNT(*) AS total,
                SUM(Survived) AS survived,
                AVG(Survived) AS survival_rate
            FROM titanic
            GROUP BY Sex
            ORDER BY Sex
            """
        ).fetchall()

        by_sex = [row_to_dict(row) for row in sex_rows]

        # 3. 依艙等計算生還率
        pclass_rows = db.execute(
            """
            SELECT
                Pclass AS group_name,
                COUNT(*) AS total,
                SUM(Survived) AS survived,
                AVG(Survived) AS survival_rate
            FROM titanic
            GROUP BY Pclass
            ORDER BY Pclass
            """
        ).fetchall()

        by_pclass = [row_to_dict(row) for row in pclass_rows]

        # 4. 依登船港口計算生還率
        embarked_rows = db.execute(
            """
            SELECT
                COALESCE(Embarked, 'Unknown') AS group_name,
                COUNT(*) AS total,
                SUM(Survived) AS survived,
                AVG(Survived) AS survival_rate
            FROM titanic
            GROUP BY COALESCE(Embarked, 'Unknown')
            ORDER BY group_name
            """
        ).fetchall()

        by_embarked = [row_to_dict(row) for row in embarked_rows]

        return jsonify({
            "message": "analysis summary loaded",
            "overall": {
                "total": total,
                "survived": survived,
                "not_survived": not_survived,
                "survival_rate": round(survival_rate, 4)
            },
            "by_sex": by_sex,
            "by_pclass": by_pclass,
            "by_embarked": by_embarked
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    
# ============================================================
# 13. API：Titanic 缺失值分析
# GET /api/analysis/missing-values
# ============================================================

@app.route("/api/analysis/missing-values", methods=["GET"])
def get_missing_values():
    try:
        # 1. 先取得 titanic 資料表總筆數
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            """
        ).fetchone()

        total = total_row["total"]

        # 如果資料表沒有資料，直接回傳錯誤
        if total == 0:
            return jsonify({
                "error": "titanic 資料表沒有資料，請先執行 init_db.py"
            }), 400

        # 2. 取得 titanic 資料表所有欄位名稱
        column_rows = db.execute(
            """
            PRAGMA table_info(titanic)
            """
        ).fetchall()

        columns = [row["name"] for row in column_rows]

        # 3. 逐一計算每個欄位的缺失值數量
        items = []

        for column in columns:
            # 這裡同時把 NULL 和空字串 '' 都當成缺失值
            # 因為有些資料可能不是 NULL，而是空白文字
            row = db.execute(
                f"""
                SELECT COUNT(*) AS missing_count
                FROM titanic
                WHERE "{column}" IS NULL
                   OR TRIM(CAST("{column}" AS TEXT)) = ''
                """
            ).fetchone()

            missing_count = row["missing_count"]
            missing_rate = missing_count / total if total > 0 else 0

            items.append({
                "column": column,
                "missing_count": missing_count,
                "missing_rate": round(missing_rate, 4),
                "missing_percent": round(missing_rate * 100, 2)
            })

        # 4. 讓缺失值多的欄位排在前面，比較容易觀察
        items = sorted(
            items,
            key=lambda item: item["missing_count"],
            reverse=True
        )

        return jsonify({
            "message": "missing values loaded",
            "total": total,
            "items": items
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# 14. 啟動 Flask
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )