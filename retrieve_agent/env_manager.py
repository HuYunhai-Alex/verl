import random
import re
from typing import List, Tuple, Dict, Union, Any, Optional
from verl import DataProto
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_sql_vector_block(log_text: str) -> str:
    """
    在 TaskRunner 日志中查找 <answer>...</answer> 标签包围的 JSON，
    并返回形如 {"sql": ..., "vector_query_list": ...} 的字符串块。
    若找不到则返回空字符串 ""。
    """
    # ① 先定位 <answer>...</answer> 块（非贪婪）
    m = re.search(r"<answer>(.*?)</answer>", log_text, flags=re.S)
    if not m:
        print("No <answer>...</answer> block found.")
        return ""

    answer_block = m.group(1)  # 只取标签内部文本

    # ② 在 answer_block 内找到第一个 “{” 并配对 “}”
    start = answer_block.find("{")
    if start == -1:
        print("No JSON object found inside <answer> block.")
        return ""

    brace_depth = 0
    end = None
    for idx, ch in enumerate(answer_block[start:], start=start):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end = idx + 1
                break

    if end is None:
        print("Unmatched braces in JSON block.")
        return ""

    json_block = answer_block[start:end]
    return json_block


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile


def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class RetrieveEnvironmentManager:
    def __init__(self, config, tokenizer):
        self.config = config
        self.url = config.get('url', 'http://127.0.0.1:8023/api/v1/search/execute_sql')
        self.headers = config.get('headers', {
            "accept": "application/json",
            "Content-Type": "application/json",
        })
        self.tokenizer = tokenizer

        # 线程池配置（可通过 config 覆盖）
        self._max_workers = int(config.get('max_workers', 32))
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # 请求超时
        self._timeout = config.get('timeout', (3, 20))  # (connect, read)

    def _do_request(self, query: str, top_k: int, filters: dict) -> Optional[dict]:
        payload = {
            "search_query": {
                "query": query,
                "top_k": top_k,
                "filters": filters or {}
            }
        }
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"请求失败: {e}")
            print(f"请求内容: {query}")
            return None

    def search_mail(self, query: str, top_k: int = 5, filters: dict = None) -> Optional[dict]:
        return self._do_request(query=query, top_k=top_k, filters=filters or {})

    def step(self, text_actions: Union[str, List[str]]):
        """
        - 兼容旧用法：传入 str -> 同步查询并返回 dict（或 None）
        - 新增并发用法：传入 List[str] -> 多线程并发查询，返回 List[dict|None]，顺序与输入一致
        """
        top_k = self.config.get('top_k', 5)
        filters = self.config.get('filters', {})

        # 单条：保持行为不变（同步）
        if isinstance(text_actions, str):
            return self.search_mail(query=text_actions, top_k=top_k, filters=filters)

        # 批量：并发执行并按输入顺序返回
        if isinstance(text_actions, list):
            futures = []
            for idx, q in enumerate(text_actions):
                futures.append((idx, self._executor.submit(self.search_mail, q, top_k, filters)))

            results: List[Optional[dict]] = [None] * len(text_actions)
            for idx, fut in futures:
                try:
                    results[idx] = fut.get() if hasattr(fut, "get") else fut.result()
                except Exception as e:
                    print(f"并发请求失败(idx={idx}): {e}")
                    results[idx] = None
            return results

        # 其他类型直接返回 None（防御）
        print(f"Unsupported type for step: {type(text_actions)}")
        return None

    def process_batch(self, batch):
        monc_sqls = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
        retrieve_answer = np.empty(len(monc_sqls), dtype=object)
        for i, monc_sql in enumerate(monc_sqls):
            monc_sql = extract_sql_vector_block(monc_sql)
            result = self.step(monc_sql)  # 不变：逐条调用
            retrieve_answer[i] = result
        batch.non_tensor_batch["retrieve_answer"] = retrieve_answer
        return batch


if __name__ == "__main__":
    query = """<think>
        好的，我现在需要处理用户的查询：“查找名字里有'张'这个字的联系人”。首先，我要确定用户的需求是什么。用户想要找到联系人姓名中包含“张”的人，这应该是在person表中进行搜索，因为联系人信息存储在person表里，而邮件信息在email表里。根据角色描述，我只能单独查询email表或person表，这里显然应该查询person表。

        接下来，我需要检查person表的结构。根据数据表结构说明，person表中有person_id、contact_name、email_count和relationship字段。其中，contact_name是联系人姓名，是非结构化的，需要使用向量检索工具。因此，用户查询的“名字里有张”应该对应到contact_name字段，使用向量匹配。

        根据工具使用规范，非结构化字段需要使用向量工具，所以这里应该用person_id IN <vector_0>的形式。因为用户需要返回联系人ID，所以SQL语句应该选择person_id，并且使用向量查询来匹配contact_name中的“张”。

        然后，我需要确保生成的SQL语句正确。因为person表中的contact_name需要向量匹配，所以SQL应该是SELECT DISTINCT person_id FROM person WHERE person_id IN <vector_0>。对应的vector_query_list应该包含{"contact_name": "张"}。

        还要注意，用户没有提到其他条件，比如关系或邮件数量，所以不需要额外的过滤条件。同时，确保没有使用任何结构化字段的直接匹配，只使用向量查询。最后，检查是否符合输出格式，只返回指定的字段和JSON结构。
        </think>

        <answer>
        {
        "sql": "SELECT DISTINCT person_id FROM person WHERE person_id IN <vector_0>",
        "vector_query_list": [
            {"contact_name": "张"}
        ]
        }
        </answer>"""
    sql = extract_sql_vector_block(query)
    print(sql)
