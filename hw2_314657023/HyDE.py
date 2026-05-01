from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class HyDE:
    def __init__(self, llm, num_hypothetical: int = 1):
        """
        num_hypothetical: 生成幾份假設文件（多份可增加召回率）
        """
        self.llm = llm
        self.num_hypothetical = num_hypothetical
        self.prompt = ChatPromptTemplate.from_template(
            """Write a concise academic-style passage that answers the question using precise technical terminology. Do not mention that it is hypothetical.

        Question: {question}

        Passage:"""
        )

    def generate_queries(self, query: str) -> List[str]:
        """
        回傳原始問題 + 生成的假設文件，格式與原本相容，
        讓 pipeline.py 裡的 fusion_retrieval 可以直接替換使用。
        """
        chain = self.prompt | self.llm | StrOutputParser()

        hypothetical_docs = []
        for _ in range(self.num_hypothetical):
            result = chain.invoke({"question": query})
            hypothetical_docs.append(result.strip())

        # 原始問題放第一個，假設文件接在後面
        return [query] + hypothetical_docs

    #Please write a scientific paper passage to answer the question.