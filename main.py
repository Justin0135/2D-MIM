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


# 5. AI 智慧診斷諮詢接口 (系統指令優化 + 多模型自動備援)
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
    
    # 🎯 優化後的 System Instruction：嚴格切離 Suzuki 與 MIM 話題
    system_instruction = (
        "你是一位有機合成與 Suzuki-Miyaura 偶聯反應專家。\n"
        "【回答原則】\n"
        "1. 請針對使用者的提問以及當前投料參數（如 TPE-Br 核心、4-羥基苯硼酸、催化劑、鹼、溶劑等）給出專業、簡明且精準的實驗建議與化學機制分析。\n"
        "2. 【重要約束】當使用者討論 Suzuki 反應的參數、鹼量、催化劑或產率失敗原因時，請專注在 Suzuki 反應本身的有機合成機制（如氧化加成、轉金屬、還原消除、鹼作用機制等）。\n"
        "3. 【嚴禁事項】除非使用者在問題中主動提及「MIM」、「拓樸」、「2D MIM」或「2D 聚合物」，否則回答中【絕對不要】提到任何 2D MIM、拓樸聚合或分子印跡等相關詞彙與概念。"
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

    # 最新備援模型優先順序表（已移除過時的 2.5，加入 3.6）
    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]

    last_error_msg = ""

    async with httpx.AsyncClient(timeout=30.0) as client:
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    res_data = response.json()
                    reply_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"reply": reply_text, "model_used": model_name}
                
                last_error_msg = f"({response.status_code}): {response.text}"
                print(f"⚠️ 模型 {model_name} 傳回 status {response.status_code}，自動切換至下一個備用模型...")
                
            except Exception as e:
                last_error_msg = str(e)
                print(f"⚠️ 呼叫模型 {model_name} 時發生網路/連線例外: {e}")

    raise HTTPException(
        status_code=503, 
        detail=f"Google AI 服務全數忙碌中或回應異常。最後錯誤細節: {last_error_msg}"
    )
