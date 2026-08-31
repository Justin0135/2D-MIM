# 5. AI 智慧診斷諮詢接口 (支援新版 AQ. 金鑰與 x-goog-api-key Header)
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
    
    # 自動清除 Key 前後可能誤複製到的空白與雙單引號
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

    # 1. 網址不要掛 ?key= (避免 AQ. 金鑰被 Google API Gateway 誤判為 OAuth Token)
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    # 2. 強制使用 Google 規範的 x-goog-api-key Header
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
