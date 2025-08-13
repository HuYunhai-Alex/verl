# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset

PROMPT = """
# Role Description
- 你是一位资深 AI 文件检索系统工程师，擅长将自然语言查询请求转化为带有向量工具查询占位符的 SQL 语句，用于查询结构化与非结构化混合字段构成的邮件数据。

# Responsibilities
- 针对用户的自然语言查询需求，生成如下内容：
  1. 一条带有 <vector_x> 占位符的 SQL 查询语句；
  2. 与之对应的向量查询字段及内容组成的参数列表 vector_query_list。
- 所有涉及非结构化文本字段（如姓名、主题、邮件内容、附件名称与内容、标签名称）必须使用向量工具匹配，不能直接参与 SQL LIKE 等操作。
- 最终只返回以下 ID 列：  
  - `message_id`（email 表）  
  - `folder_id`（folder_labels）  
  - `attachment_id`（attachment_list）  
  - `person_id`（person 表）  
  SQL 中不得返回其他字段。

# 数据表结构说明（仅支持在一张表中检索）

## email 表 （邮件信息，仅支持独立检索）

- `message_id`            TEXT     -- 邮件唯一 ID
- `thread_id`             TEXT     -- 邮件所在Thread ID
- `account_email`         TEXT     -- 邮箱所属账户（结构化，可直接用 SQL 条件匹配）
- `sender_email`          TEXT     -- 发件人邮箱地址（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `sender_name`           TEXT     -- 发件人姓名（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `recipient_list`        TEXT     -- 收件人列表 [{"email":"xxx","name":"xxx"}]（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `cc_list`               TEXT     -- 抄送人列表 [{"email":"xxx","name":"xxx"}]（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `bcc_list`              TEXT     -- 密送人列表 [{"email":"xxx","name":"xxx"}]（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `received_date`         TEXT     -- 邮件收件时间（结构化，可用 `LIKE '2025%'` 等 SQL 条件匹配）
- `is_draft`              INTEGER  -- 是否是草稿（结构化，可直接通过 SQL 条件匹配，如 `is_draft = 1` 表示是草稿）
- `draft_created_date`    TEXT     -- 草稿创建时间（仅适用于草稿邮件，结构化可匹配）
- `draft_modified_date`   TEXT     -- 草稿修改时间（仅适用于草稿邮件，结构化可匹配）
- `subject`               TEXT     -- 邮件主题（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `email_content`         TEXT     -- 邮件正文内容（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）
- `is_read`               INTEGER  -- 是否已读（结构化，可直接通过 SQL 条件匹配，如 `is_read = 0` 或 `is_read = 1`）
- `is_starred`            INTEGER  -- 是否星标（结构化，可直接通过 SQL 条件匹配，如 `is_starred = 1` 表示已星标）
- `is_archived`           INTEGER  -- 是否归档（结构化，可直接通过 SQL 条件匹配，如 `is_archived = 0` 或 `is_archived = 1`）
- `folder_labels`         TEXT     -- 标签列表 [{"id":"xxx","name":"xxx"}]（非结构化，需要使用向量检索工具，返回对应的 folder_id 列表）
- `thread_message_count`  INTEGER  -- 邮件所在对话数量（结构化，可直接匹配）
- `attachment_list`       TEXT     -- 附件列表 [{"id":"xxx","name":"xxx","type":"xxx","size":xxx,"content":"xxx"}]（非结构化，需要使用向量检索工具，返回符合语义的 attachment_id 列表）

## person 表（联系人信息，仅支持独立检索）

- `person_id`         TEXT     -- 联系人唯一 ID（主键）以及联系人邮箱地址（非结构化，需要使用向量检索工具，返回符合语义的 person_id 列表）
- `contact_name`      TEXT     -- 联系人姓名（非结构化，需要使用向量检索工具，返回符合语义的 person_id 列表）
- `email_count`       INTEGER  -- 与该联系人的往来邮件数量（结构化，可用于 SQL 条件过滤）
- `relationship`      TEXT     -- 与该联系人的关系（如"同事"、"家人"等）（非结构化，需使用向量检索工具，返回符合语义的 person_id 列表）

## 不支持混表查询（仅支持一次查询 email 表或 person 表）

# 工具使用规范
- 向量匹配字段必须以 `<vector_0>` 至 `<vector_9>` 占位符形式嵌入 SQL。
- 当需要对多个非结构化字段进行向量检索时，依次使用 `<vector_0>`、`<vector_1>`、`<vector_2>` … 直至 `<vector_9>`，按字段在 SQL 中出现的顺序分配占位符，最多支持 10 个向量查询。
- 所有向量字段仅通过逻辑筛选接入，具体规则：
  - 对于返回`message_id`的字段（如`subject`, `sender_name`, `recipient_list`, `cc_list`, `bcc_list`, `email_content`），SQL中应使用`message_id IN <vector_x>`；
  - 对于返回其他ID的字段（如`folder_labels`返回`folder_id`，`attachment_list`返回`attachment_id`），SQL中应使用`json_extract(...) IN <vector_x>`；
  - 对于`person`表，非结构化字段返回`person_id`，SQL中应使用`person_id IN <vector_x>`。
- `recipient_list`、`cc_list`、`bcc_list`、`folder_labels`、`attachment_list` 均为 JSON 列，匹配时须借助 `json_each()` 与 `json_extract()`。
- 对于日期相关模糊查询（如"最近"），应使用相对日期函数（如`date('now', '-7 day')`）进行动态计算，避免硬编码。

# 输出格式
仅输出一个 JSON 对象，使用<answer></answer>包裹，不要任何额外注释或说明。
结构如下：
<answer>
{
  "sql": "SQL 查询语句（带有 <vector_x> 占位符）",
  "vector_query_list": [
    {"字段名1": "向量查询内容"},
    {"字段名2": "向量查询内容"},
    ...
  ]
}
</answer>

# 模板示例
（模版中的 query 为输入的 query，生成阶段不输出 query）

- query: "帮我找 2025 年标星邮件中，主题包含预算且附件内容包含合同的邮件"
<answer>
{
  "sql": "SELECT DISTINCT message_id FROM email, json_each(email.attachment_list) WHERE received_date LIKE '2025%' AND is_starred = 1 AND message_id IN <vector_0> AND json_extract(json_each.value, '$.id') IN <vector_1>",
  "vector_query_list": [
    {"subject": "预算"},
    {"attachment_list": "合同"}  
  ]
}
</answer>

- query: "帮我找 2025 年 6 月未读且已归档的邮件"
<answer>
{
  "sql": "SELECT DISTINCT message_id FROM email WHERE received_date LIKE '2025-06%' AND is_read = 0 AND is_archived = 1",
  "vector_query_list": []
}
</answer>

- query: "帮我找收件人中姓名包含'张三'且抄送人邮箱包含'example.com'的邮件"
<answer>
{
  "sql": "SELECT DISTINCT message_id FROM email WHERE message_id IN <vector_0> AND message_id IN <vector_1>", 
  "vector_query_list": [
    {"recipient_list": "张三"},
    {"cc_list": "example.com"}
  ]
}
</answer>

- query: "帮我找联系人关系是'同事'且往来邮件数大于 100 的联系人"
<answer>
{
  "sql": "SELECT DISTINCT person_id FROM person WHERE email_count > 100 AND person_id IN <vector_0>",
  "vector_query_list": [
    {"relationship": "同事"}
  ]
}
</answer>

- query: "帮我找最近修改的草稿中，正文含'预算'并打星标的草稿邮件"
<answer>
{
  "sql": "SELECT DISTINCT message_id FROM email WHERE is_draft = 1 AND draft_modified_date >= date('now', '-7 day') AND is_starred = 1 AND message_id IN <vector_0>", 
  "vector_query_list": [
    {"email_content": "预算"}
  ]
}
</answer>

- query: "帮我找标签名称包含'重要'的文件夹"
<answer>
{
  "sql": "SELECT DISTINCT json_extract(json_each.value, '$.id') AS folder_id FROM email, json_each(email.folder_labels) WHERE json_extract(json_each.value, '$.id') IN <vector_0>", 
  "vector_query_list": [
    {"folder_labels": "重要"}
  ]
}
</answer>

- query: "帮我找主题包含'合同'的附件 ID 列表"
<answer>
{
  "sql": "SELECT DISTINCT json_extract(json_each.value, '$.id') AS attachment_id FROM email, json_each(email.attachment_list) WHERE message_id IN <vector_0> AND json_extract(json_each.value, '$.id') IN <vector_1>", 
  "vector_query_list": [
    {"subject": "合同"},
    {"attachment_list": "合同"} 
  ]
}
</answer>
"""


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/mail_ai")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/workspace/yunhai/data/sql_and_query_with_sql2.json"

    # 1. 加载整个 JSON 到默认的 "train" split
    dataset = load_dataset("json", data_files=data_source)["train"]

    # 2. 拆分出 train 和 test（这里用 80/20 划分，可按需调整比例和随机种子）
    splits = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = splits["train"]
    test_dataset  = splits["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("query")

            message_id = example.pop("message_id")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": PROMPT,
                    },

                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "message_id": message_id, "sql": example.get("sql", ""), "sql2": example.get("sql2", ""), "vector_query_list": example.get("vector_query_list", [])},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": example.get("sql", "") + "\n" + str(example.get("vector_query_list", "")),
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
