import time
import requests
import re
import os
from supabase import create_client, Client

# 環境變數載入
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def fetch_suzuki_literature(keyword="Suzuki-Miyaura coupling catalyst ligand", max_results=5):
    """從 Crossref API 抓取 Suzuki 偶聯反應文獻"""
    print(f"🔍 正在爬取關鍵字：[{keyword}] 的最新學術文獻...")
    url = f"https://api.crossref.org/works?query={keyword}&rows={max_results}&sort=published&order=desc"
    headers = {
        "User-Agent": "SuzukiAICrawler/1.0 (mailto:your_email@example.com)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"❌ API 請求失敗，HTTP 狀態碼: {response.status_code}")
            return []
        
        data = response.json()
        items = data.get("message", {}).get("items", [])
        
        crawled_data = []
        for item in items:
            title = item.get("title", ["無標題"])[0]
            doi = item.get("DOI", "")
            publisher = item.get("publisher", "Unknown")
            published_year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            abstract = item.get("abstract", "無摘要")
            
            # 清除 JATS/XML 標籤
            clean_abstract = re.sub(r'<[^>]+>', '', abstract) if abstract else "無摘要"
            
            # 擷取催化劑與鹼關鍵字
            catalysts_found = extract_keywords(clean_abstract, ["Pd(OAc)2", "Pd(PPh3)4", "Pd2(dba)3", "SPhos", "XPhos", "RuPhos", "dppf"])
            bases_found = extract_keywords(clean_abstract, ["K2CO3", "Na2CO3", "Cs2CO3", "K3PO4", "NEt3", "tBuOK"])

            crawled_data.append({
                "title": title,
                "doi": doi,
                "publisher": publisher,
                "year": published_year,
                "abstract": clean_abstract,
                "extracted_catalysts": ", ".join(catalysts_found),
                "extracted_bases": ", ".join(bases_found),
                "source_url": f"https://doi.org/{doi}" if doi else ""
            })
            
        print(f"✅ 成功擷取 {len(crawled_data)} 筆文獻！")
        return crawled_data

    except Exception as e:
        print(f"❌ 爬蟲執行過程發生異常: {e}")
        return []

def extract_keywords(text, keyword_list):
    """提取摘要中包含的化學試劑關鍵字"""
    found = []
    for kw in keyword_list:
        if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE):
            found.append(kw)
    return found

def sync_to_supabase(articles):
    """將爬取到的文獻資料寫入 Supabase knowledge_base"""
    if not articles:
        print("⚠️ 無可同步的數據")
        return

    clean_url = SUPABASE_URL.strip().strip('"').strip("'") if SUPABASE_URL else ""
    clean_key = SUPABASE_KEY.strip().strip('"').strip("'") if SUPABASE_KEY else ""

    if not clean_url or not clean_key:
        print("⚠️ 未設定 Supabase URL/Key，跳過資料庫同步")
        return

    print("🚀 正在同步數據至 Supabase...")
    supabase: Client = create_client(clean_url, clean_key)
    
    for article in articles:
        db_payload = {
            "title": article["title"],
            "doi": article["doi"],
            "abstract": article["abstract"],
            "catalyst_info": article["extracted_catalysts"],
            "base_info": article["extracted_bases"],
            "source_url": article["source_url"],
            "category": "Suzuki-Miyaura"
        }
        
        try:
            supabase.table("knowledge_base").upsert(db_payload, on_conflict="doi").execute()
            print(f" └─ 已存入 DB: {article['title'][:40]}...")
        except Exception as e:
            print(f" └─ 寫入失敗 ({article['doi']}): {e}")
