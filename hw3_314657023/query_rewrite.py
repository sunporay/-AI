import json
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 定義 Parser 預期的 JSON 結構，強制 3B 模型輸出符合格式的結果
class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 1 to 3 targeted search queries.")

class QueryRewriter:
    def __init__(self, llm):
        """
        專為高密度學術文本與簡短技術宣告設計的 Query Rewriter。
        它的功能是將複雜的摘要「拆解」成多個獨立的搜尋查詢，並保留複合名詞與技術用語。
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=SearchQueries)

        # PROMPT 保持英文，這對 3B 模型的 Instruction Following 能力非常有幫助
        self.prompt = ChatPromptTemplate.from_template(
            """You are an academic search expert.

            Convert the text into 1-3 short search queries.

            Rules:
            - Keep technical terms unchanged
            - Split different concepts
            - Use concise phrases (not full sentences)

            Return JSON:
            {{"queries": ["query1", "query2"]}}

            Text: {claim}
            """
            )

    def generate_queries(self, claim: str) -> List[str]:
        """
        回傳原始句子 + 生成的子查詢陣列。
        """
        chain = self.prompt | self.llm | self.parser

        try:
            # 解析 JSON 輸出以提取查詢列表
            result = chain.invoke({
                "claim": claim,
                "format_instructions": self.parser.get_format_instructions()
            })
            generated_queries = result.get("queries", [])
        except Exception as e:
            # 容錯機制：萬一 3B 模型偶爾失常沒有輸出合法 JSON，至少退回原始查詢
            print(f"JSON parsing failed: {e}")
            generated_queries = [claim]

        # 回傳格式：[原始 Claim, 子查詢 1, 子查詢 2, ...]
        return [claim] + generated_queries