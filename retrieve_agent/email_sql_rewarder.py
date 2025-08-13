
# email_sql_rewarder.py
# -*- coding: utf-8 -*-
'''
Package the SQL + vector-query scoring flow into a class, with the LLM (gpt4o) created *inside*.

Quick start (functional API):
    from email_sql_rewarder import score_email_sql

    # Option 1: rely on environment variables for Azure/OpenAI auth (see notes below)
    res = score_email_sql(
        rel_requirement="(在账户 wanderbirds@email.com 下)",
        text_requirements="""```json
[
  {"正文": "包含鸟类摄影项目的详细信息"},
  {"附件列表": "含有北美鸟类的图片"},
  {"主题": "与鸟类摄影相关的项目"}
]
```""",

        candidate={
            "sql": "SELECT email.message_id FROM email WHERE email.account_email='wanderbirds@email.com' AND email.message_id IN <vector_0>",
            "vector_query_list": [{"email_content": "与鸟类摄影相关的项目，正文包含细节并有北美鸟类图片"}]
        }
    )
    print(res["rank_tag"], res["final"])

Quick start (class API):
    from email_sql_rewarder import EmailSQLRewarder
    rewarder = EmailSQLRewarder(temperature=0.2, top_p=0.2)  # config via env by default
    res = rewarder.compute_reward([ "(在账户 ...)" ], text_requirements, candidate)

Auth notes (resolution order inside the class):
  1) Azure AD (DefaultAzureCredential) → Azure OpenAI (deployment_name).
  2) Azure with API key: AZURE_OPENAI_API_KEY.
  3) OpenAI with API key: OPENAI_API_KEY (model fallback: gpt-4o).
You may also pass azure_endpoint/deployment_name/api_version via ctor; otherwise env is used:
  - AZURE_OPENAI_ENDPOINT / LLM_AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_API_VERSION / LLM_AZURE_OPENAI_API_VERSION
  - AZURE_OPENAI_DEPLOYMENT / LLM_AZURE_OPENAI_DEPLOYMENT
'''
import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional, Union

# Optional deps
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers.json import SimpleJsonOutputParser
except Exception:
    ChatPromptTemplate = None
    SimpleJsonOutputParser = None


# ---------- 工具：解析 inputs ----------
def parse_text_requirements(text_requirements: str) -> List[Dict[str, str]]:
    """text_requirements 是带 ```json ... ``` 的字符串，这里提取为 list[ {字段中文名: 内容} ]"""
    m = re.search(r"```json\s*(.*?)\s*```", text_requirements, flags=re.S)
    payload = m.group(1) if m else text_requirements
    return json.loads(payload)


CN2FIELD = {
    "正文": "email_content",
    "主题": "subject",
    "附件列表": "attachment_list",
    "抄送": "cc_list",
    "收件人": "recipient_list",
    "发件人": "from_list",
    "标签": "folder_labels",
}

ALLOWED_IDS = {"message_id", "folder_id", "attachment_id", "person_id"}
ALLOWED_VECTOR_FIELDS = {
    "attachment_list","email_content","subject",
    "cc_list","to_list","from_list","recipient_list","folder_labels",
    # person 表：
    "contact_name","relationship","person_id"
}


class EmailSQLRewarder:
    """End-to-end wrapper that computes a reward for SQL + vector_query_list candidates.
    
    LLM client (gpt4o) is constructed *inside* this class. Resolution order:
      1) Azure AD (DefaultAzureCredential) → Azure OpenAI (deployment_name).
      2) Azure with API key (AZURE_OPENAI_API_KEY).
      3) OpenAI with API key (OPENAI_API_KEY), model fallback 'gpt-4o'.
    """

    SYSTEM_PROMPT = "你是严格的检索策略评审员，请根据给定的规则为候选 SQL+向量参数打分。只返回 JSON。"

    JUDGE_TMPL = """
# 关系需求（逐条满足）
{rel_requirements}

# 文本需求（字段→内容，允许同义改写但不可改变语义）
{text_requirements}

# 候选输出（SQL + vector_query_list）
{candidate_json}

评分维度（0~1，小数可用）：
- coverage_rel：是否覆盖关系需求（账户、日期），且无错误归属；
- coverage_text：是否覆盖文本属性（主题/正文/附件），允许同义但不可遗漏；
- field_mapping：语义→字段是否正确；
- keyword_correctness：关键词不臆造；
- time_format：日期筛选更推荐使用相对时间（如 date('now', ...)）或 LIKE/区间，而非死等号；
- deduplication：查询不会因笛卡尔积导致相同 ID 多次返回（如使用 EXISTS / DISTINCT / GROUP BY 规避）；
- overall：总体评分。

严格返回：
{{
  "coverage_rel": 0.0,
  "coverage_text": 0.0,
  "field_mapping": 0.0,
  "keyword_correctness": 0.0,
  "overall": 0.0,
  "notes": "一句话简述主要扣分点"
}}
"""

    def __init__(self, 
                 temperature: float = 0.2,
                 top_p: float = 0.2,
                 azure_endpoint: Optional[str] = None,
                 deployment_name: Optional[str] = None,
                 api_version: Optional[str] = None,
                 model_fallback: str = "gpt-4o",
                 llm_enabled: bool = True):
        # Runtime config (env as default)
        self.temperature = temperature
        self.top_p = top_p
        self.azure_endpoint = azure_endpoint or os.getenv("LLM_AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv("LLM_AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt4o"
        self.api_version = api_version or os.getenv("LLM_AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"
        self.model_fallback = model_fallback
        self.llm_enabled = llm_enabled

    # ---------- 内部：创建 LLM ----------
    def _make_llm(self):
        """Create a LangChain chat model client internally.
        Prefers Azure OpenAI with AAD → Azure key → OpenAI key.
        """
        # Try Azure AD first
        try:
            from langchain_openai import AzureChatOpenAI
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            return AzureChatOpenAI(
                openai_api_version="2024-08-01-preview",
                azure_endpoint="https://loox-eastus.openai.azure.com/",
                deployment_name="gpt4o",
                azure_ad_token_provider=token_provider,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        except Exception as e:
            raise RuntimeError("Failed to initialize any LLM backend: " + str(e))

    # ---------- 规则打分 ----------
    @staticmethod
    def only_allowed_ids(sql: str) -> float:
        m = re.search(r"select\s+(distinct\s+)?(.*?)\s+from\s", sql, flags=re.I|re.S)
        if not m: 
            return 0.0
        cols = re.split(r"\s*,\s*", re.sub(r"\s+", " ", m.group(2).strip()))
        ok = True
        for c in cols:
            alias_m = re.search(r"\bas\s+([a-zA-Z_][\w]*)\b", c, flags=re.I)
            colname = alias_m.group(1) if alias_m else c.strip()
            colname = colname.split(".")[-1]
            if colname.lower() not in ALLOWED_IDS:
                ok = False
                break
        return 1.0 if ok else 0.0

    @staticmethod
    def single_table(sql: str) -> float:
        email = re.search(r"\bfrom\s+email\b", sql, flags=re.I)
        person = re.search(r"\bfrom\s+person\b", sql, flags=re.I)
        if email and person:
            return 0.0
        if not email and not person:
            return 0.0
        if re.search(r"\bfrom\s+email\s*,\s*(?!json_each)", sql, flags=re.I):
            return 0.0
        if re.search(r"\bfrom\s+person\s*,\s*", sql, flags=re.I):
            return 0.0
        return 1.0

    @staticmethod
    def vector_usage_and_order(sql: str, vector_query_list: List[Dict[str,str]]) -> Tuple[float, float]:

        vecs = re.findall(r"<vector_(\d+)>", sql)
        if not vecs:
            return 0.0, 0.0
        vec_index_seq = list(map(int, vecs))
        order_ok = vec_index_seq == list(range(len(vec_index_seq)))

        fields_in_sql = []
        sql_norm = re.sub(r"\s+", " ", sql.lower())
        for i in vec_index_seq:
            pat_msg = rf"message_id\s+in\s+<vector_{i}>"
            pat_json = rf"json_extract\([^)]*,'\$\.(?:id)'\)\s+in\s+<vector_{i}>"
            if re.search(pat_msg, sql_norm):
                fields_in_sql.append("message_id")
            elif re.search(pat_json, sql_norm):
                fields_in_sql.append("json_id")
            else:
                fields_in_sql.append("unknown")

        keys = []
        for item in vector_query_list or []:
            if isinstance(item, dict) and item:
                for k in item.keys():
                    keys.append(k); break

        order_score = 1.0 if order_ok and len(keys) == len(vec_index_seq) else 0.5 if order_ok else 0.0
        used_ok = all(k in ALLOWED_VECTOR_FIELDS for k in keys) and len(keys) == len(vec_index_seq)
        use_score = 1.0 if used_ok else 0.0
        return use_score, order_score

    @staticmethod
    def attachment_clause_ok(sql: str, text_requirements_list: List[Dict[str,str]]) -> float:
        need_attach = any("附件" in k or "附件" in "".join(v for v in d.values()) 
                          for d in text_requirements_list for k in d.keys())
        if not need_attach:
            return 1.0
        has_exists = re.search(r"exists\s*\(\s*select\s+1\s+from\s+json_each\s*\(\s*email\.attachment_list", sql, flags=re.I|re.S)
        has_json = re.search(r"json_each\s*\(\s*email\.attachment_list", sql, flags=re.I)
        has_extract = re.search(r"json_extract\s*\([^)]*'\$\.id'\s*\)\s+in\s+<vector_\d+>", sql, flags=re.I)
        return 1.0 if (has_exists or (has_json and has_extract)) else 0.0

    @staticmethod
    def account_ok(sql: str, rel_requirements: List[str]) -> float:
        m = re.search(r"account\s+([\w\.-]+@[\w\.-]+)", " ".join(rel_requirements), flags=re.I)
        if not m:
            return 1.0
        email = m.group(1)
        has = re.search(rf"email\.account_email\s*=\s*'?{re.escape(email)}'?", sql, flags=re.I)
        return 1.0 if has else 0.0

    @staticmethod
    def date_ok(sql: str, rel_requirements: List[str]) -> float:
        iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})", " ".join(rel_requirements))
        if iso:
            day = iso.group(1)
            ok = (
                re.search(r"received_date\s*=\s*'[^']*"+day, sql, flags=re.I) or
                re.search(r"received_date\s+like\s+'"+day+r"[%_]'", sql, flags=re.I) or
                re.search(r"between\s*'"+day+r"[^']*'\s*and\s*'"+day, sql, flags=re.I)
            )
            return 1.0 if ok else 0.0
        fuzzy = re.search(r"最近|过去|近\s*\d+\s*(天|周|月)|within|last\s+\d+\s*(day|week|month)", " ".join(rel_requirements), flags=re.I)
        if fuzzy:
            return 1.0 if re.search(r"date\(\s*'now'", sql, flags=re.I) or re.search(r"\bCURRENT_DATE\b", sql, flags=re.I) else 0.0
        return 1.0

    @staticmethod
    def vector_keys_match_requirements(vector_query_list: List[Dict[str,str]], text_requirements_list: List[Dict[str,str]]) -> float:
        expect = set()
        for d in text_requirements_list:
            for k in d.keys():
                eng = CN2FIELD.get(k)
                if eng:
                    expect.add(eng)
        got = {next(iter(x.keys())) for x in vector_query_list or [] if isinstance(x, dict) and x}
        if not expect:
            return 1.0
        if expect.issubset(got):
            return 1.0
        if expect & got:
            return 0.5
        return 0.0

    def auto_score(self, rel_requirements: List[str], text_requirements: str, candidate: Dict[str, Any]) -> Tuple[float, Dict[str,float]]:
        try:
            tlist = parse_text_requirements(text_requirements)
        except Exception:
            tlist = []
        sql = str(candidate.get("sql",""))
        vlist = candidate.get("vector_query_list", [])

        parts = {
            "only_allowed_ids": self.only_allowed_ids(sql),
            "single_table": self.single_table(sql),
            "vector_use": 0.0,
            "vector_order": 0.0,
            "attachment_clause": self.attachment_clause_ok(sql, tlist),
            "account_ok": self.account_ok(sql, rel_requirements),
            "date_ok": self.date_ok(sql, rel_requirements),
            "keys_cover": self.vector_keys_match_requirements(vlist, tlist),
            "format_ok": 1.0 if isinstance(candidate.get("vector_query_list", None), list) and "sql" in candidate else 0.0,
        }
        vu, vo = self.vector_usage_and_order(sql, vlist)
        parts["vector_use"], parts["vector_order"] = vu, vo

        W = {
            "only_allowed_ids": 0.20,
            "single_table":     0.15,
            "vector_use":       0.15,
            "vector_order":     0.05,
            "attachment_clause":0.15,
            "account_ok":       0.10,
            "date_ok":          0.10,
            "keys_cover":       0.08,
            "format_ok":        0.02,
        }
        score = sum(W[k]*parts[k] for k in W)
        return score, parts

    # ---------- LLM 评分 ----------
    def build_llm_chain(self):
        if ChatPromptTemplate is None or SimpleJsonOutputParser is None:
            raise RuntimeError("LangChain not installed. Please install langchain packages to use LLM scoring.")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.JUDGE_TMPL),
        ])
        return prompt | self._make_llm() | SimpleJsonOutputParser()

    def llm_score(self, rel_requirements: List[str], text_requirements: str, candidate: Dict[str, Any]) -> Tuple[float, Dict[str,float]]:
        chain = self.build_llm_chain()
        cj = json.dumps(candidate, ensure_ascii=False)
        rr = "\n".join(f"- {x}" for x in rel_requirements)
        out = chain.invoke({
            "rel_requirements": rr,
            "text_requirements": text_requirements,
            "candidate_json": cj
        })
        detail = {
            "coverage_rel": float(out.get("coverage_rel", 0.0)),
            "coverage_text": float(out.get("coverage_text", 0.0)),
            "field_mapping": float(out.get("field_mapping", 0.0)),
            "keyword_correctness": float(out.get("keyword_correctness", 0.0)),
        }
        overall = float(out.get("overall", 0.0))
        detail["notes"] = str(out.get("notes", "")).strip()
        return overall, detail

    # ---------- 融合为 reward，并输出 <rank> ----------
    def compute_reward(self, rel_requirements: List[str], text_requirements: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        auto_overall, auto_detail = self.auto_score(rel_requirements, text_requirements, candidate)
        if self.llm_enabled:
            llm_overall, llm_detail  = self.llm_score(rel_requirements, text_requirements, candidate)
        else:
            llm_overall, llm_detail = 0.0, {"coverage_rel": 0.0, "coverage_text": 0.0, "field_mapping": 0.0, "keyword_correctness": 0.0, "notes": "LLM disabled"}
        final = 0.45 * auto_overall + 0.55 * llm_overall
        final = max(0.0, min(1.0, final))
        return {
            "final": final,
            "rank_tag": f"<rank>{final:.4f}</rank>",
            "auto": auto_detail,
            "llm": llm_detail
        }

    # ---------- 批量接口（可选） ----------
    def rank_candidates(self, rel_requirements: List[str], text_requirements: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = [self.compute_reward(rel_requirements, text_requirements, c) for c in candidates]
        return sorted(results, key=lambda x: x["final"], reverse=True)


# --------- Simple functional API ---------
def score_email_sql(rel_requirement: Union[str, List[str]], text_requirements: str, candidate: Dict[str, Any], **rewarder_kwargs) -> Dict[str, Any]:
    """Convenience function: external callers can pass a single requirement string or a list."""
    rel_reqs = [rel_requirement] if isinstance(rel_requirement, str) else list(rel_requirement or [])
    rewarder = EmailSQLRewarder(**rewarder_kwargs)
    return rewarder.compute_reward(rel_reqs, text_requirements, candidate)

def rank_email_sql(rel_requirement: Union[str, List[str]], text_requirements: str, candidates: List[Dict[str, Any]], **rewarder_kwargs) -> List[Dict[str, Any]]:
    """Rank multiple candidates with a single call."""
    rel_reqs = [rel_requirement] if isinstance(rel_requirement, str) else list(rel_requirement or [])
    rewarder = EmailSQLRewarder(**rewarder_kwargs)
    return rewarder.rank_candidates(rel_reqs, text_requirements, candidates)


if __name__ == "__main__":
    # Minimal demo (requires proper Azure/OpenAI env vars set)
    rewarder = EmailSQLRewarder()
    rel_requirements = [
        "(收件于 2023-10-15T10:00:00Z，今天日期：2025-08-07)",
        "(在账户 wanderbirds@email.com 下)"
    ]
    text_requirements = '''```json
[
  {"正文": "包含鸟类摄影项目的详细信息"},
  {"附件列表": "含有北美鸟类的图片"},
  {"主题": "与鸟类摄影相关的项目"}
]
```'''
    candidate = {
        "sql": "SELECT email.message_id FROM email, json_each(email.attachment_list) WHERE email.account_email = 'wanderbirds@email.com' AND received_date = '2023-10-15' AND email.message_id IN <vector_0>",
        "vector_query_list": [
            {"email_content": "鸟类摄影项目，正文包含关于鸟类摄影项目的详细信息，并附有北美鸟类的图片"}
        ]
    }
    try:
        res = rewarder.compute_reward(rel_requirements, text_requirements, candidate)
        print(res["rank_tag"])
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Demo failed:", e)
