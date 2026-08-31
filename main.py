import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

# 1. 允許 cross-origin 存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 載入 ML 模型 (.pkl)
MODEL_PATH = "suzuki_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ suzuki_model.pkl 成功載入！")
    except Exception as e:
        print(f"⚠️ 載入 suzuki_model.pkl 失敗: {e}")

# 3. 初始化 Gemini API Client
# 請確保 Render 後端的 Environment Variables 有設定 GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# 4. 首頁路由：直接回傳 index.html 網頁畫面
@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "error", "message": "index.html 不存在於根目錄"}


# 5. ML 產率預測接口
class PredictRequest(BaseModel):
    tpe_br: float
    b_acid: float

@app.post("/predict")
def predict_yield(data: PredictRequest):
    if model is None:
        # 若模型載入失敗則回傳預設參考值
        return {"predicted_yield": 85.0, "status": "fallback"}
    
    try:
        # 建立特徵 DataFrame (欄位名稱請依據訓練時的名稱調整)
        input_data = pd.DataFrame([{
            "tpe_br": data.tpe_br,
            "b_acid": data.b_acid
        }])
        prediction = model.predict(input_data)[0]
        return {"predicted_yield": float(prediction), "status": "success"}
    except Exception as e:
        print(f"預測出錯: {e}")
        return {"predicted_yield": 85.0, "status": "error", "detail": str(e)}


# 6. AI 智慧診斷諮詢接口
class ConsultRequest(BaseModel):
    prompt: str
    tpe_br: float
    b_acid: float

@app.post("/ai-consult")
def ai_consult(data: ConsultRequest):
    if not gemini_client:
        raise HTTPException(
            status_code=500, 
            detail="後端未設定 GEMINI_API_KEY 環境變數"
        )
    
    system_instruction = (
        "你是一位 Suzuki 偶聯反應與 2D MIM 拓樸聚合專家。"
        "請針對使用者的提問以及目前的投料參數（TPE-Br 核心、4-羥基苯硼酸）給出專業、簡明且精準的實驗建議。"
    )
    
    user_context = (
        f"【目前實驗投料參數】\n"
        f"- TPE-Br: {data.tpe_br} mg\n"
        f"- 4-羥基苯硼酸: {data.b_acid} mg\n\n"
        f"【使用者提問】\n{data.prompt}"
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_context,
            config={"system_instruction": system_instruction}
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API 呼叫失敗: {str(e)}")
