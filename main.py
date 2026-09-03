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


# 5. AI 智慧診斷諮詢接口 (嵌入文獻基準數據 + 多模型自動備援)
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
    
    # 🎯 融入文獻基準數據的 System Instruction
    system_instruction = (
        "你是一位有機合成與 Suzuki-Miyaura 偶聯反應專家。\n\n"
        "【文獻標準實驗基準 (Standard Protocol)】\n"
        "反應容器：20 mL Scintillation vial\n"
        "1. TPE-Br (1,1,2,2-tetrakis-(4-bromophenyl)ethylene): 960 mg (1.48 mmol, 1.00 equiv)\n"
        "2. 4-羥基苯硼酸 (4-hydroxyboronic acid): 1226 mg / 1.226 g (8.88 mmol, 6.00 equiv)\n"
        "3. 碳酸鉀 (Potassium carbonate, K2CO3): 870 mg (6.30 mmol, 4.25 equiv)\n"
        "4. TBAB (Tetrabutylammonium bromide): 239 mg (0.741 mmol, 0.5 equiv)\n"
        "5. 催化劑 Pd(OAc)2: 3.3 mg (0.015 mmol, 1.00 mol%)\n"
        "6. 配體 SPhos: 9.1 mg (0.022 mmol, 1.50 mol%)\n"
        "7. 內標物 1,3,5-trimethoxybenzene: 82.3 mg (0.49 mmol, 0.33 equiv)\n\n"
        "【回答原則】\n"
        "1. 當使用者提問或變動投料量（如 TPE-Br、4-羥基苯硼酸、K2CO3 等）時，請務必先將使用者的數據與上述【文獻標準實驗基準】進行莫耳比（equiv）與比例的對照分析，評估過量或不足對產率及副反應的影響。\n"
        "2. 專注於 Suzuki 偶聯反應本身的化學機制（氧化加成、轉金屬、還原消除、鹼作用機制、相轉移催化等）。\n"
        "3. 【嚴禁事項】除非使用者問題中主動提及「MIM」、「拓樸」、「2D MIM」或「2D 聚合物」，否則【絕對不要】提到任何 MIM、拓樸聚合或分子印跡等概念。"
    )
    
    user_context = (
        f"【目前使用者設定投料】\n"
        f"- TPE-Br 核心: {data.tpe_br} mg\n"
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
