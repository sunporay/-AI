import os
import re
import pickle
from embeddings import split_oversized_chunks
from filelock import FileLock
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_experimental.text_splitter import SemanticChunker

from retriever import create_bm25_index
from query_rewrite import QueryRewriter


QWEN_CHATML_PATTERN = re.compile(
    r"<\|im_start\|>(?:user|assistant|system)?\n?|<\|im_end\|>\n?|<\|endoftext\|>"
)

class PaperRAGPipeline:
    def __init__(
        self,
        embedding_model,
        llm=None,
        alpha: float = 0.5,
        fusion_k: int = 10,
        rerank_top_k: int = 3,
        num_queries: int = 3,
        persist_dir: str = "./faiss",
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.mq_expander = QueryRewriter(llm=llm, num_hypothetical=1) if llm else None
        self.splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.8,
        )
        self.alpha = alpha
        self.fusion_k = fusion_k
        self.rerank_top_k = rerank_top_k
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""

                You are a hallucination detection assistant.

                Your task:
                Determine whether the claim is supported by the context.

                Definition:
                A hallucination is any part of the answer that is not directly supported by the provided context.

                This includes:
                - Attribution Failure: A claim lacks proper attribution, either crediting the wrong source or presenting information as fact without citation.
                - Entity: A claim includes swapped, incorrectly specified, or inserted noun phrases (e.g. one named entity used in a context where another word is expected).
                - Number: A claim has a different number than the original context (e.g. 20% vs. 0.7%). Any number, including year, dimensions, ages, etc.
                - Overgeneralization: A claim is based on accurate contextual information but is too broad or too general to be supported by the context.
                - Temporal: A claim does not accurately incorporate tense, modality (e.g. might vs. will), or time reference in relation to the context.

                Instruction:
                    Based on the context and evidence provided, determine which type of hallucination (if any) is present.
            """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Original Claim:
                {question}

                Context (retrieved via expanded queries):
                {context}
                """
            ),
        ])

    def _get_index_path(self, rel_path: str) -> str:
        # rel_path 例如 train/paper_4 → faiss/train/paper_4
        return os.path.join(self.persist_dir, rel_path)

    def _build_index(self, full_text: str, title: str, rel_path: str = None):
        # rel_path 沒傳就退回用 title 當路徑（相容舊呼叫）
        index_path  = self._get_index_path(rel_path if rel_path else title)
        bm25_path   = os.path.join(index_path, "bm25.pkl")
        splits_path = os.path.join(index_path, "splits.pkl")
        lock_path   = index_path + ".lock"

        with FileLock(lock_path):
            if (
                os.path.exists(index_path)
                and os.path.exists(bm25_path)
                and os.path.exists(splits_path)
            ):
                try:
                    vectorstore = FAISS.load_local(
                        index_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                    with open(bm25_path, "rb") as f:
                        bm25 = pickle.load(f)
                    with open(splits_path, "rb") as f:
                        splits = pickle.load(f)
                    return splits, vectorstore, bm25
                except Exception as e:
                    print(f"[WARN] Failed to load index for '{title}': {e}. Rebuilding...")

            os.makedirs(index_path, exist_ok=True)
            splits = self.splitter.create_documents(
                [full_text], metadatas=[{"title": title}]
            )
            splits = split_oversized_chunks(splits)
            vectorstore = FAISS.from_documents(splits, self.embedding_model)
            vectorstore.save_local(index_path)
            bm25 = create_bm25_index(splits)

            with open(bm25_path, "wb") as f:
                pickle.dump(bm25, f)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)

        return splits, vectorstore, bm25

    def _generate_answer(self, question: str, docs) -> str:
        context = "\n\n".join(d.page_content for d in docs)
        chain = self.rag_prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"context": context, "question": question})
        return QWEN_CHATML_PATTERN.sub("", raw).strip()

