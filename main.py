import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 請在這裡貼上你的 Gemini API Key
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", "AQ.Ab8RN6JkwCl3P_uZ6nCr_JaJ0G0zkhHCmvzS8H1owIS9GyoaOg"
)

# 初始化 Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# 🤖 載入機器學習模型 (請確保 suzuki_model.pkl 在同一個資料夾)
try:
    ml_model = joblib.load("suzuki_model.pkl")
    print("✅ ML 模型成功載入！")
except Exception as e:
    ml_model = None
    print(
        f"⚠️ ML 模型載入失敗：{e}（請確認 suzuki_model.pkl 是否存放在同一資料夾）"
    )


# --- 1. 原本的 Gemini AI 顧問部分 ---
class ReactionData(BaseModel):
    tpe_br_mg: float = 0
    b_acid_mg: float = 0
    pd_mg: float = 0
    ligand_mg: float = 0
    user_question: str = ""


@app.post("/api/ai-consult")
async def ai_consult(data: ReactionData):
    prompt = f"""
你是一位精通 Suzuki 偶聯反應、TPE 材料合成與 NMR 定量分析的專業化學 AI 顧問。
使用者正在進行 TPE-Br 與 4-羥基苯硼酸 (4-HBPA) 的 Suzuki 偶聯反應。

當前實驗投料數據：
- TPE-Br: {data.tpe_br_mg} mg
- 4-羥基苯硼酸: {data.b_acid_mg} mg
- 鈀催化劑 Pd(OAc)2: {data.pd_mg} mg
- SPhos 配體: {data.ligand_mg} mg

使用者提出的問題/討論：
『{data.user_question}』

請根據上述投料當量、化學機制與 NMR 相關知識，給出專業、簡潔且具體的分析與建議。
【重要格式要求】：
1. 請用繁體中文回答。
2. 化學式請直接寫成純文字格式（例如直接寫 K2CO3、Pd(OAc)2、TPE-4OH），**千萬不要使用 LaTeX 語法（例如不要用 $...$ 或 \\text{{...}}）**，以利網頁直接閱讀。
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        ai_reply = "🤖 Gemini 專家分析：\n" + response.text
    except Exception as e:
        ai_reply = f"🤖 Gemini API 連線失敗：\n請檢查 main.py 中的 API Key 是否正確設定。\n錯誤訊息：{str(e)}"

    return {"status": "success", "reply": ai_reply}


# --- 2. 新增的 ML 機器學習預測 API 部分 ---
class MLPredictRequest(BaseModel):
    tpe_br_mg: float
    b_acid_mg: float
    pd_mg: float
    ligand_mg: float


@app.post("/predict")
def predict_yield(data: MLPredictRequest):
    if ml_model is None:
        return {"status": "error", "message": "ML 模型尚未載入！"}

    # 將前端送來的 4 個投料數據整理成模型要求的格式
    # 附註：請確保特徵順序與你在 Colab 訓練時放入的 X 順序完全一致
    input_features = np.array(
        [[data.tpe_br_mg, data.b_acid_mg, data.pd_mg, data.ligand_mg]]
    )

    # 呼叫模型進行預測
    prediction = ml_model.predict(input_features)[0]

    return {"status": "success", "predicted_yield": round(float(prediction), 2)}