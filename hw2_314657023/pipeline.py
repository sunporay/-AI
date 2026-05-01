import os
import re
import pickle
import hashlib
from embeddings import split_oversized_chunks
#from embeddings import pre_chunk_text
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
from HyDE import HyDE

LLAMA_SPECIAL_TOKENS = re.compile(
    r"<\|(?:begin_of_text|end_of_text|start_header_id|end_header_id|eot_id)\|>"
    r"|<<SYS>>|<</SYS>>|\[INST\]|\[/INST\]"
)

def clean_text (text: str) -> str:
    import re, unicodedata

    # normalize
    text = unicodedata.normalize("NFKC", text)

    # remove urls & emails
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # remove citations
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\([A-Za-z]+ et al\., \d{4}\)", "", text)

    # remove references section
    text = re.split(r"References", text, flags=re.IGNORECASE)[0]

    # remove figure/table captions
    text = re.sub(r"Figure \d+.*", "", text)
    text = re.sub(r"Table \d+.*", "", text)

    # clean spacing
    text = text.replace("\t", " ")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()

class PaperRAGPipeline:
    def __init__(
        self,
        embedding_model,
        llm=None,
        alpha: float = 0.5,
        fusion_k: int = 10,
        rerank_top_k: int = 3,
        num_queries: int = 3,
        persist_dir: str = "./faiss_db",
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.mq_expander = HyDE(llm=llm, num_hypothetical=1) if llm else None
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
            SystemMessagePromptTemplate.from_template(
                "Answer the question based only on the provided context. "
                "Be concise and factual."
            ),
            HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n\nQuestion:\n{question}"
            ),
        ])
    def _get_index_path(self, title: str) -> str:
        folder_name = "paper_" + hashlib.md5(title.encode()).hexdigest()[:12]
        return os.path.join(self.persist_dir, folder_name)

    def _build_index(self, full_text: str, title: str):
        index_path  = self._get_index_path(title)
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
            #full_text = clean_text(full_text)
            #texts = pre_chunk_text(full_text)
            splits = self.splitter.create_documents(
                [full_text], metadatas=[{"title": title}]
            )
            splits = split_oversized_chunks(splits)  # ← 新增這行
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
        return LLAMA_SPECIAL_TOKENS.sub("", raw).strip()