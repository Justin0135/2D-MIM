import os
import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# 1. 開放 CORS 跨域存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 全域模型變數 (延遲載入)
model = None

def get_model():
    global model
    if model is None:
        MODEL_PATH = "suzuki_model.pkl"
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                print("✅ suzuki_model.pkl 成功載入！")
            except Exception as e:
                print(f"⚠️ 載入 suzuki_model.pkl 失敗: {e}")
    return model


# 3. 首頁路由：回傳 index.html 畫面
@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "Backend API is running!"}


# 4. ML 產率預測接口
class PredictRequest(BaseModel):
    tpe_br: float
    b_acid: float

@app.post("/predict")
def predict_yield(data: PredictRequest):
    mdl = get_model()
    if mdl is None:
        return {"predicted_yield": 85.0, "status": "fallback"}
    
    try:
        input_data = pd.DataFrame([{
            "tpe_br": data.tpe_br,
            "b_acid": data.b_acid
        }])
        prediction = mdl.predict(input_data)[0]
        return {"predicted_yield": float(prediction), "status": "success"}
    except Exception as e:
        print(f"預測出錯: {e}")
        return {"predicted_yield": 85.0, "status": "error", "detail": str(e)}


# 5. AI 智慧診斷諮詢接口 (多模型自動備援修復版)
class ConsultRequest(BaseModel):
    prompt: str
    tpe_br: float
    b_acid: float

@app.post("/ai-consult")
async def ai_consult(data: ConsultRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="後端未偵測到 GEMINI_API_KEY 環境變數，請至 Render Environment 設定。"
        )
    
    clean_api_key = api_key.strip().strip('"').strip("'")
    
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

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_api_key
    }

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "parts": [{"text": user_context}]
            }
        ]
    }

    # 備援模型優先順序表
    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3.6-flash"
    ]

    last_error_msg = ""

    async with httpx.AsyncClient(timeout=30.0) as client:
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                response = await client.post(url, json=payload, headers=headers)
                
                # 請求成功，提取回應內容回傳
                if response.status_code == 200:
                    res_data = response.json()
                    reply_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"reply": reply_text, "model_used": model_name}
                
                # 若回應非 200 (包含 503 忙碌, 429 限流, 404 等)，紀錄錯誤並自動進行下一個模型重試
                last_error_msg = f"({response.status_code}): {response.text}"
                print(f"⚠️ 模型 {model_name} 傳回 status {response.status_code}，自動切換至下一個備用模型...")
                
            except Exception as e:
                last_error_msg = str(e)
                print(f"⚠️ 呼叫模型 {model_name} 時發生網路/連線例外: {e}")

    # 若列表內所有模型皆呼叫失敗，才丟出異常
    raise HTTPException(
        status_code=503, 
        detail=f"Google AI 服務全數忙碌中或回應異常。最後錯誤細節: {last_error_msg}"
    )
