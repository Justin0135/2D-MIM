import os
import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel  # <--- 就是漏了這行！

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


# 5. AI 智慧診斷諮詢接口 (使用 x-goog-api-key 驗證，解決 401 錯誤)
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

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"

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

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Gemini API 傳回錯誤 ({response.status_code}): {response.text}"
                )
            
            res_data = response.json()
            reply_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return {"reply": reply_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"呼叫 API 過程發生例外: {str(e)}")
