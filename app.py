import os
import sqlite3
import json
import joblib
import pandas as pd
from datetime import datetime

from flask import Flask, jsonify, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ============================================================
# 1. 專案設定與資料庫連線
# ============================================================

DATABASE = "my_db.db"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.joblib")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# ============================================================
# 2. 模型特徵設定
# ============================================================

ALLOWED_MODEL_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "FamilyGroup",
    "HasCabin"
]

DEFAULT_MODEL_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "FamilyGroup",
    "HasCabin"
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "FamilyGroup"
]

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "HasCabin"
]

# 這裡我們直接在全域讀取資料庫，這樣在每個 route 就可以直接使用 db 來存取資料庫了。
db = sqlite3.connect(DATABASE, check_same_thread=False)

# 讓我們在讀取資料庫時，可以直接用 row["欄位名稱"] 的方式來存取資料，
# 而不是 row[0]、row[1] 這樣的 index。
db.row_factory = sqlite3.Row


# ============================================================
# 3. 共用工具函式
# ============================================================

def row_to_dict(row):
    return dict(row)

def blank_to_none(value):
    """
    將表單中的空字串轉成 None，方便 SQLite 儲存為 NULL。
    """

    if value == "":
        return None

    return value


def normalize_passenger_payload(data):
    """
    將新增 / 編輯乘客時前端送來的字串資料轉成正確型別。

    HTML 表單送出的資料預設都是字串；這裡集中轉型，避免資料庫 CHECK
    約束或後續機器學習流程因型別不一致而出錯。
    """

    required_fields = [
        "Survived", "Pclass", "Name", "Sex", "SibSp", "Parch", "Ticket", "Fare"
    ]

    if data is None:
        raise ValueError("請提供 JSON 格式資料")

    for field in required_fields:
        if field not in data or data[field] == "":
            raise ValueError(f"缺少必要欄位：{field}")

    age_value = blank_to_none(data.get("Age"))
    cabin_value = blank_to_none(data.get("Cabin"))
    embarked_value = blank_to_none(data.get("Embarked"))

    return {
        "Survived": int(data["Survived"]),
        "Pclass": int(data["Pclass"]),
        "Name": data["Name"].strip(),
        "Sex": data["Sex"],
        "Age": float(age_value) if age_value is not None else None,
        "SibSp": int(data["SibSp"]),
        "Parch": int(data["Parch"]),
        "Ticket": data["Ticket"].strip(),
        "Fare": float(data["Fare"]),
        "Cabin": cabin_value,
        "Embarked": embarked_value
    }

def validate_selected_features(selected_features):
    """
    檢查使用者從前端勾選的特徵是否合法。

    規則：
    1. 如果沒有傳 selected_features，就使用 DEFAULT_MODEL_FEATURES
    2. 至少要選 1 個特徵
    3. 只能選 ALLOWED_MODEL_FEATURES 裡面的特徵
    """

    if selected_features is None:
        return DEFAULT_MODEL_FEATURES

    if not isinstance(selected_features, list):
        raise ValueError("selected_features 必須是 list 格式")

    if len(selected_features) == 0:
        raise ValueError("請至少選擇 1 個模型特徵")

    invalid_features = [
        feature for feature in selected_features
        if feature not in ALLOWED_MODEL_FEATURES
    ]

    if len(invalid_features) > 0:
        raise ValueError(f"不允許使用的特徵：{invalid_features}")

    return selected_features

def add_title_feature(df):
    """
    從 Name 欄位萃取 Title 特徵
    例如：
    Braund, Mr. Owen Harris -> Mr
    Cumings, Mrs. John Bradley -> Mrs
    Heikkinen, Miss. Laina -> Miss
    """

    df = df.copy()

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    }

    df["Title"] = df["Title"].replace(title_mapping)

    common_titles = ["Mr", "Mrs", "Miss", "Master"]

    df["Title"] = df["Title"].apply(
        lambda title: title if title in common_titles else "Rare"
    )

    return df

def add_family_name_feature(df):
    """
    從 Name 欄位萃取 FamilyName 特徵
    例如：
    Braund, Mr. Owen Harris -> Braund
    Cumings, Mrs. John Bradley -> Cumings
    """

    df = df.copy()

    df["FamilyName"] = df["Name"].str.split(",").str[0]

    return df

def add_family_size_feature(df):
    """
    新增 FamilySize 特徵
    FamilySize = SibSp + Parch + 1

    SibSp：兄弟姊妹 / 配偶數
    Parch：父母 / 子女數
    +1：乘客本人
    """

    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    return df

def add_family_group_feature(df):
    """
    新增 FamilyGroup 特徵
    根據 FamilySize 分成：
    1       -> Alone
    2 ~ 4   -> SmallFamily
    5 以上  -> LargeFamily
    """

    df = df.copy()

    def classify_family_group(family_size):
        if family_size == 1:
            return "Alone"
        elif family_size <= 4:
            return "SmallFamily"
        else:
            return "LargeFamily"

    df["FamilyGroup"] = df["FamilySize"].apply(classify_family_group)

    return df

def add_has_cabin_feature(df):
    """
    新增 HasCabin 特徵
    Cabin 有值 -> 1
    Cabin 沒有值 -> 0
    """

    df = df.copy()

    df["HasCabin"] = df["Cabin"].notna().astype(int)

    return df

def add_feature_engineering(df):
    """
    統一管理 Titanic 特徵工程
    """

    df = df.copy()

    df = add_title_feature(df)
    df = add_family_name_feature(df)
    df = add_family_size_feature(df)
    df = add_family_group_feature(df)
    df = add_has_cabin_feature(df)

    return df

def load_model_info():
    if not os.path.exists(MODEL_INFO_PATH):
        return None

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_info(info):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

def build_model(model_type, params):
    """
    根據 model_type 建立不同的機器學習模型。

    回傳：
    model：真正要訓練的模型
    model_name：模型名稱，存到 model_info.json
    best_params：本次模型使用的參數
    """

    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))

    if model_type == "random_forest":
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = int(params.get("max_depth", 5))
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

        model_name = "RandomForestClassifier"

        best_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "test_size": test_size,
            "random_state": random_state
        }

    elif model_type == "logistic_regression":
        C = float(params.get("C", 1.0))
        max_iter = int(params.get("max_iter", 1000))
        solver = params.get("solver", "liblinear")

        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state
        )

        model_name = "LogisticRegression"

        best_params = {
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "test_size": test_size,
            "random_state": random_state
        }

    elif model_type == "gradient_boosting":
        n_estimators = int(params.get("n_estimators", 100))
        learning_rate = float(params.get("learning_rate", 0.1))
        max_depth = int(params.get("max_depth", 3))

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

        model_name = "GradientBoostingClassifier"

        best_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": random_state
        }

    else:
        raise ValueError(f"不支援的模型類型：{model_type}")

    return model, model_name, best_params


# ============================================================
# 5. Titanic 資料前處理
# ============================================================

def preprocess_titanic_train_data(df, selected_features=None):
    """
    訓練模型用的資料前處理函式。

    這版支援：
    1. 使用者自選特徵
    2. 自動加入特徵工程
    3. 只對被選到的類別欄位做 one-hot encoding

    回傳：
    X：處理後特徵
    y：目標欄位
    preprocessing_info：記錄訓練時使用的補值策略與欄位資訊
    """

    df = df.copy()

    # ----------------------------
    # 1. 檢查使用者選擇的特徵
    # ----------------------------
    selected_features = validate_selected_features(selected_features)

    # ----------------------------
    # 2. 加入特徵工程
    # 會新增：
    # Title、FamilyName、FamilySize、FamilyGroup、HasCabin
    # ----------------------------
    df = add_feature_engineering(df)

    # ----------------------------
    # 3. 目標欄位 y：是否生還
    # ----------------------------
    y = df["Survived"]

    # ----------------------------
    # 4. 根據使用者選擇的特徵建立 X
    # ----------------------------
    X = df[selected_features].copy()

    # ----------------------------
    # 5. 根據 selected_features 判斷哪些是類別欄位、哪些是數值欄位
    # ----------------------------
    selected_categorical_columns = [
        column for column in CATEGORICAL_FEATURES
        if column in selected_features
    ]

    selected_numeric_columns = [
        column for column in NUMERIC_FEATURES
        if column in selected_features
    ]

    # ----------------------------
    # 6. 記錄訓練時的補值策略
    # 只有被選到的欄位才需要記錄
    # ----------------------------
    preprocessing_info = {
        "selected_features": selected_features,
        "categorical_columns": selected_categorical_columns,
        "numeric_columns": selected_numeric_columns,
        "raw_feature_columns": selected_features
    }

    if "Age" in X.columns:
        age_median = X["Age"].median()
        X["Age"] = X["Age"].fillna(age_median)
        preprocessing_info["age_fill_value"] = float(age_median)

    if "Fare" in X.columns:
        fare_median = X["Fare"].median()
        X["Fare"] = X["Fare"].fillna(fare_median)
        preprocessing_info["fare_fill_value"] = float(fare_median)

    if "Embarked" in X.columns:
        embarked_mode = X["Embarked"].mode()[0]
        X["Embarked"] = X["Embarked"].fillna(embarked_mode)
        preprocessing_info["embarked_fill_value"] = embarked_mode

    if "Title" in X.columns:
        X["Title"] = X["Title"].fillna("Rare")

    if "FamilyGroup" in X.columns:
        X["FamilyGroup"] = X["FamilyGroup"].fillna("Alone")

    # ----------------------------
    # 7. 類別欄位轉換成 one-hot encoding
    # 只轉換這次有被選到的類別欄位
    # ----------------------------
    if len(selected_categorical_columns) > 0:
        X = pd.get_dummies(
            X,
            columns=selected_categorical_columns
        )

    return X, y, preprocessing_info


def preprocess_titanic_predict_data(input_df, model_features, preprocessing_info):
    """
    預測用的資料前處理函式。

    重點：
    1. 預測時使用訓練時記錄下來的補值策略
    2. 預測時也要產生 Title、FamilySize、FamilyGroup、HasCabin
    3. 最後要對齊模型訓練時的欄位
    """

    input_df = input_df.copy()

    # ----------------------------
    # 1. 補上特徵工程需要的欄位
    # ----------------------------
    # 單筆預測表單可能沒有 Name / Cabin 欄位；先給預設值，
    # 讓 add_feature_engineering() 可以正常執行。
    if "Name" not in input_df.columns:
        input_df["Name"] = "Unknown, Mr. Unknown"

    if "Cabin" not in input_df.columns:
        input_df["Cabin"] = None

    # ----------------------------
    # 2. 加入特徵工程
    # ----------------------------
    input_df = add_feature_engineering(input_df)

    # ----------------------------
    # 3. 只保留訓練時使用的原始特徵欄位
    # ----------------------------
    raw_feature_columns = preprocessing_info.get("raw_feature_columns", DEFAULT_MODEL_FEATURES)
    input_df = input_df[raw_feature_columns].copy()

    # ----------------------------
    # 4. 缺失值處理
    # ----------------------------
    # 注意：使用者可以在 /ml 取消勾選 Age / Fare / Embarked，
    # 因此這裡必須先確認欄位存在，不能直接固定補值。
    if "Age" in input_df.columns:
        input_df["Age"] = input_df["Age"].fillna(
            preprocessing_info.get("age_fill_value", input_df["Age"].median())
        )

    if "Fare" in input_df.columns:
        input_df["Fare"] = input_df["Fare"].fillna(
            preprocessing_info.get("fare_fill_value", input_df["Fare"].median())
        )

    if "Embarked" in input_df.columns:
        input_df["Embarked"] = input_df["Embarked"].fillna(
            preprocessing_info.get("embarked_fill_value", "S")
        )

    if "Title" in input_df.columns:
        input_df["Title"] = input_df["Title"].fillna("Rare")

    if "FamilyGroup" in input_df.columns:
        input_df["FamilyGroup"] = input_df["FamilyGroup"].fillna("Alone")

    # ----------------------------
    # 5. 類別欄位轉換成 one-hot encoding
    # ----------------------------
    categorical_columns = [
        column for column in preprocessing_info.get("categorical_columns", CATEGORICAL_FEATURES)
        if column in input_df.columns
    ]

    if len(categorical_columns) > 0:
        input_df = pd.get_dummies(input_df, columns=categorical_columns)

    # ----------------------------
    # 6. 對齊模型訓練時的欄位
    # ----------------------------
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    return input_df


# ============================================================
# 6. 前端頁面 Routes
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
def machine_learning_titanic():
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
# 7-1. API：取得全部乘客資料，包含簡單分頁
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
# 7-2. API：取得單一乘客
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
# 7-3. API：新增乘客
# POST /api/passengers
# ============================================================

@app.route("/api/passengers", methods=["POST"])
def create_passenger():
    # 從 request 的 JSON body 讀取資料，並統一轉成正確型別
    try:
        data = normalize_passenger_payload(request.get_json())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

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
# 7-4. API：修改乘客
# PUT /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["PUT"])
def update_passenger(passenger_id):
    # 從 request 的 JSON body 讀取資料，並統一轉成正確型別
    try:
        data = normalize_passenger_payload(request.get_json())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

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
# 7-5. API：刪除乘客
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
# 8-1. API：一鍵訓練 Titanic 生存預測模型
# POST /api/ml/train
# ============================================================

@app.route("/api/ml/train", methods=["POST"])
def train_titanic_model():
    try:
        # 讀取前端從 ml.html 傳來的訓練設定。
        # 前端目前會同時送：
        # 1. 外層參數，例如 model_type、selected_features
        # 2. params 物件，例如各模型的超參數
        # 這裡合併兩種格式，讓後端更穩定。
        request_data = request.get_json(silent=True) or {}
        nested_params = request_data.get("params", {})

        if not isinstance(nested_params, dict):
            nested_params = {}

        params = {**nested_params, **request_data}
        params.pop("params", None)

        model_type = params.get("model_type", "random_forest")

        allowed_model_types = [
            "random_forest",
            "logistic_regression",
            "gradient_boosting"
        ]

        if model_type not in allowed_model_types:
            return jsonify({
                "error": "不支援的模型類型",
                "model_type": model_type,
                "allowed_model_types": allowed_model_types
            }), 400

        test_size = float(params.get("test_size", 0.2))
        random_state = int(params.get("random_state", 42))

        # 讀取前端勾選的特徵
        selected_features = params.get("selected_features")
        selected_features = validate_selected_features(selected_features)

        # 從 SQLite 的 titanic 資料表讀取訓練資料
        df = pd.read_sql_query(
            """
            SELECT 
                Survived,
                Pclass,
                Name,
                Sex,
                Age,
                SibSp,
                Parch,
                Fare,
                Cabin,
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
        X, y, preprocessing_info = preprocess_titanic_train_data(
            df,
            selected_features=selected_features
        )

        # 切分訓練集與測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # 使用 build_model() 統一建立不同模型
        model, model_name, best_params = build_model(model_type, params)

        model.fit(X_train, y_train)

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
            "model_type": model_type,
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "best_params": best_params,
            "selected_features": selected_features,
            "features": list(X.columns),
            "preprocessing_info": preprocessing_info,
            "rows": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": MODEL_PATH,
            "old_accuracy": round(old_accuracy, 4) if old_accuracy is not None else None,
            "new_accuracy": round(accuracy, 4),
            "is_better_model": is_better_model,
            "compare_message": compare_message
        }

        # 只有當本次模型比舊模型好，或目前沒有舊模型時，才更新正式模型檔案
        if is_better_model:
            os.makedirs(MODEL_DIR, exist_ok=True)

            # 覆蓋正式模型
            joblib.dump(model, MODEL_PATH)

            model_updated = True
            update_message = "本次模型已取代目前模型"

            # 覆蓋正式模型資訊
            model_info["model_updated"] = model_updated
            model_info["update_message"] = update_message
            save_model_info(model_info)
        else:
            model_updated = False
            update_message = "本次模型未取代目前模型，仍保留原本模型"

        # 回傳訓練結果給前端
        return jsonify({
            "message": "training completed",
            "model_type": model_type,
            "model": model_info["model"],
            "rows": model_info["rows"],
            "features": model_info["features"],
            "selected_features": selected_features,
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

            # 是否真的覆蓋正式模型
            "model_updated": model_updated,
            "update_message": update_message
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# 8-2. API：檢查模型檔案與模型資訊
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
    "model_type": model_info.get("model_type"),
    "model": model_info.get("model"),
    "accuracy": model_info.get("accuracy"),
    "best_params": model_info.get("best_params"),
    "selected_features": model_info.get("selected_features"),
    "features": model_info.get("features"),
    "rows": model_info.get("rows"),
    "train_size": model_info.get("train_size"),
    "test_size": model_info.get("test_size"),
    "trained_at": model_info.get("trained_at"),
    "model_path": model_info.get("model_path")
}), 200

# ============================================================
# 8-3. API：單筆預測乘客是否存活
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

    model_info = load_model_info()

    if model_info is None:
        return jsonify({
            "error": "找不到模型資訊，請重新訓練模型"
        }), 400

    preprocessing_info = model_info.get("preprocessing_info")

    if preprocessing_info is None:
        return jsonify({
            "error": "模型資訊缺少 preprocessing_info，請重新訓練模型"
        }), 400

    # 5. 整理成 DataFrame
    input_df = pd.DataFrame([{
        "Pclass": int(data["Pclass"]),
        "Name": data.get("Name", "Unknown, Mr. Unknown"),
        "Sex": data["Sex"],
        "Age": float(data["Age"]),
        "SibSp": int(data["SibSp"]),
        "Parch": int(data["Parch"]),
        "Fare": float(data["Fare"]),
        "Cabin": data.get("Cabin", None),
        "Embarked": data["Embarked"]
    }])

    # 6. 預測資料也要做跟訓練時一樣的前處理
    input_df = preprocess_titanic_predict_data(
        input_df,
        model.feature_names_in_,
        preprocessing_info
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
# 8-4. API：CSV 批次預測乘客是否存活
# POST /api/ml/predict-csv
# ============================================================
@app.route("/api/ml/predict-csv", methods=["POST"])
def predict_survival_csv():
    try:
        # 1. 先確認模型檔案是否存在
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                "error": "尚未訓練模型，請先到機器學習頁面訓練模型"
            }), 400

        # 2. 檢查前端有沒有上傳 file
        if "file" not in request.files:
            return jsonify({
                "error": "沒有收到 CSV 檔案，請確認欄位名稱是 file"
            }), 400

        file = request.files["file"]

        # 3. 檢查檔名是否為空
        if file.filename == "":
            return jsonify({
                "error": "尚未選擇 CSV 檔案"
            }), 400

        # 4. 檢查副檔名
        if not file.filename.lower().endswith(".csv"):
            return jsonify({
                "error": "請上傳 .csv 檔案"
            }), 400

        # 5. 使用 pandas 讀取 CSV
        try:
            input_df = pd.read_csv(file)
        except Exception:
            return jsonify({
                "error": "CSV 讀取失敗，請確認檔案格式是否正確"
            }), 400

        # 6. 檢查 CSV 是否有資料
        if input_df.empty:
            return jsonify({
                "error": "CSV 檔案沒有資料"
            }), 400

        # 7. 檢查必要欄位
        required_fields = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in input_df.columns
        ]

        if len(missing_fields) > 0:
            return jsonify({
                "error": "CSV 缺少必要欄位",
                "missing_fields": missing_fields,
                "required_fields": required_fields
            }), 400

        # 8. 載入模型與模型資訊
        model = joblib.load(MODEL_PATH)

        model_info = load_model_info()
        if model_info is None:
            return jsonify({
                "error": "找不到模型資訊，請重新訓練模型"
            }), 400

        preprocessing_info = model_info.get("preprocessing_info")
        if preprocessing_info is None:
            return jsonify({
                "error": "模型資訊缺少 preprocessing_info，請重新訓練模型"
            }), 400

        # 9. 型別轉換
        # errors="coerce" 代表轉換失敗會變成 NaN
        # 後面的前處理函式會處理 Age / Fare 的 NaN
        numeric_columns = [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare"
        ]

        for column in numeric_columns:
            input_df[column] = pd.to_numeric(input_df[column], errors="coerce")

        # 10. 補上可選欄位
        # 如果 CSV 沒有 Name / Cabin，也可以正常預測
        if "Name" not in input_df.columns:
            input_df["Name"] = "Unknown, Mr. Unknown"

        if "Cabin" not in input_df.columns:
            input_df["Cabin"] = None

        # 11. 使用共用前處理函式
        processed_df = preprocess_titanic_predict_data(
            input_df,
            model.feature_names_in_,
            preprocessing_info
        )

        # 12. 批次預測
        predictions = model.predict(processed_df)
        probabilities = model.predict_proba(processed_df)

        # 13. 整理回傳結果
        items = []

        has_passenger_id = "PassengerId" in input_df.columns

        for index, prediction in enumerate(predictions):
            prediction_int = int(prediction)
            survival_probability = float(probabilities[index][1])

            item = {
                "row": index + 1,
                "prediction": prediction_int,
                "prediction_label": "生還" if prediction_int == 1 else "未生還",
                "survival_probability": survival_probability
            }

            # 如果 CSV 有 PassengerId，就一起回傳
            # Kaggle submission 需要 PassengerId
            if has_passenger_id:
                item["PassengerId"] = int(input_df.iloc[index]["PassengerId"])

            items.append(item)

        return jsonify({
            "message": "batch prediction completed",
            "total": len(items),
            "has_passenger_id": has_passenger_id,
            "items": items
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# 9-1. API：Titanic 資料分析摘要
# GET /api/analysis/summary
# ============================================================

@app.route("/api/analysis/summary", methods=["GET"])
def get_analysis_summary():
    try:
        # 1. 從 SQLite 讀取分析需要的欄位
        # 注意：這裡要讀 Name、SibSp、Parch、Cabin
        # 因為 Title / FamilyGroup / HasCabin 會用到這些欄位
        df = pd.read_sql_query(
            """
            SELECT
                Survived,
                Pclass,
                Name,
                Sex,
                Age,
                SibSp,
                Parch,
                Fare,
                Cabin,
                Embarked
            FROM titanic
            """,
            db
        )

        # 2. 如果資料表沒有資料，回傳錯誤
        if df.empty:
            return jsonify({
                "error": "titanic 資料表沒有資料，請先執行 init_db.py"
            }), 400

        # 3. 套用我們前面建立好的特徵工程
        # 這裡會新增：
        # Title、FamilyName、FamilySize、FamilyGroup、HasCabin
        df = add_feature_engineering(df)

        # 4. 整體生還統計
        total = len(df)
        survived = int(df["Survived"].sum())
        not_survived = total - survived
        survival_rate = survived / total if total > 0 else 0

        # 5. 依性別計算生還率
        by_sex = (
            df.groupby("Sex")
            .agg(
                total=("Survived", "count"),
                survived=("Survived", "sum"),
                survival_rate=("Survived", "mean")
            )
            .reset_index()
            .rename(columns={"Sex": "group_name"})
        )

        by_sex["survival_rate"] = by_sex["survival_rate"].round(4)

        # 6. 依艙等計算生還率
        by_pclass = (
            df.groupby("Pclass")
            .agg(
                total=("Survived", "count"),
                survived=("Survived", "sum"),
                survival_rate=("Survived", "mean")
            )
            .reset_index()
            .rename(columns={"Pclass": "group_name"})
        )

        by_pclass["survival_rate"] = by_pclass["survival_rate"].round(4)

        # 7. 依登船港口計算生還率
        df["Embarked"] = df["Embarked"].fillna("Unknown")

        by_embarked = (
            df.groupby("Embarked")
            .agg(
                total=("Survived", "count"),
                survived=("Survived", "sum"),
                survival_rate=("Survived", "mean")
            )
            .reset_index()
            .rename(columns={"Embarked": "group_name"})
        )

        by_embarked["survival_rate"] = by_embarked["survival_rate"].round(4)

        # 8. 依 Title 計算生還率
        by_title = (
            df.groupby("Title")
            .agg(
                total=("Survived", "count"),
                survived=("Survived", "sum"),
                survival_rate=("Survived", "mean")
            )
            .reset_index()
            .rename(columns={"Title": "group_name"})
        )

        by_title["survival_rate"] = by_title["survival_rate"].round(4)

        # 9. 依 FamilyGroup 計算生還率
        by_family_group = (
            df.groupby("FamilyGroup")
            .agg(
                total=("Survived", "count"),
                survived=("Survived", "sum"),
                survival_rate=("Survived", "mean")
            )
            .reset_index()
            .rename(columns={"FamilyGroup": "group_name"})
        )

        by_family_group["survival_rate"] = by_family_group["survival_rate"].round(4)

        # 10. 回傳 JSON 給前端
        return jsonify({
            "message": "analysis summary loaded",
            "overall": {
                "total": total,
                "survived": survived,
                "not_survived": not_survived,
                "survival_rate": round(survival_rate, 4)
            },
            "by_sex": by_sex.to_dict(orient="records"),
            "by_pclass": by_pclass.to_dict(orient="records"),
            "by_embarked": by_embarked.to_dict(orient="records"),

            # Title / FamilyGroup 分析
            "by_title": by_title.to_dict(orient="records"),
            "by_family_group": by_family_group.to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    
# ============================================================
# 9-2. API：Titanic 缺失值分析
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
# 10. 啟動 Flask
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )