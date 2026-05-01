# reranker.py
import uuid
import torch
import numpy as np
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class OfficialQwenReranker:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B", gpu_id=1, max_length=2048, batch_size=1):
        print(f"Loading Reranker Model: {model_name} on cuda:{gpu_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=f"cuda:{gpu_id}"
        ).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = max_length
        self.batch_size = batch_size

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def format_instruction(self, query: str, doc: str) -> str:
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=self.instruction, query=query, doc=doc
        )

    def process_inputs(self, pairs_text: List[str]):
        inputs = self.tokenizer(
            pairs_text, padding=False, truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs) -> List[float]:
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        if isinstance(scores, float):
            return [scores]
        return scores

    def predict(self, pairs: List[List[str]]) -> List[float]:
        formatted_texts = [self.format_instruction(query, doc) for query, doc in pairs]

        all_scores = []
        for i in tqdm(
                    range(0, len(formatted_texts), self.batch_size),
                    desc="Reranking",
                    leave=False
        ):
            batch_texts = formatted_texts[i : i + self.batch_size]
            inputs = self.process_inputs(batch_texts)
            scores = self.compute_logits(inputs)
            all_scores.extend(scores)
        torch.cuda.empty_cache()
        return all_scores


class MultiGPUReranker:
    """
    多張 GPU 平行 rerank：每張 GPU 各載一個 OfficialQwenReranker，
    predict 時把 pairs 切成 n 份後用 ThreadPoolExecutor 平行推論。
    """
    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B",
                 gpu_ids: List[int] = [1, 2, 3],
                 max_length: int = 2048,
                 batch_size: int = 8):
        self.n = len(gpu_ids)
        print(f"[MultiGPUReranker] Loading {self.n} rerankers on GPUs {gpu_ids} ...")
        self.rerankers = [
            OfficialQwenReranker(
                model_name=model_name,
                gpu_id=g,
                max_length=max_length,
                batch_size=batch_size,
            )
            for g in gpu_ids
        ]
        print(f"[MultiGPUReranker] Ready.")

    def predict(self, pairs: List[List[str]]) -> List[float]:
        if not pairs:
            return []

        # 平均切成 n 份（最後一份可能稍少）
        chunks = np.array_split(pairs, self.n)

        def _run(args):
            reranker, chunk = args
            chunk = list(chunk)
            if not chunk:
                return []
            return reranker.predict(chunk)

        with ThreadPoolExecutor(max_workers=self.n) as exe:
            results = list(exe.map(_run, zip(self.rerankers, chunks)))

        # 依序合併，保持原本 pairs 的順序
        return [score for chunk_scores in results for score in chunk_scores]


# ──────────────────────────────────────────────────────────────────────────────
# Service helpers（供 pipeline 跨 process 呼叫）
# ──────────────────────────────────────────────────────────────────────────────

def reranker_service_fn(request_queue, response_queues,
                        gpu_ids: List[int] = [1, 2, 3],
                        batch_size: int = 8):
    print("[RerankerService] Loading MultiGPUReranker...", flush=True)
    reranker = MultiGPUReranker(gpu_ids=gpu_ids, batch_size=batch_size)
    print("[RerankerService] Ready.", flush=True)

    while True:
        item = request_queue.get()
        if item is None:
            print("[RerankerService] Shutting down.", flush=True)
            break
        worker_id, request_id, pairs = item
        try:
            scores = reranker.predict(pairs)
        except Exception as e:
            print(f"[RerankerService] predict error: {e}", flush=True)
            scores = [0.0] * len(pairs)
        response_queues[worker_id].put((request_id, scores))


def rerank_via_service(worker_id, request_queue, response_queue, pairs):
    req_id = str(uuid.uuid4())
    request_queue.put((worker_id, req_id, pairs))
    while True:
        resp_id, scores = response_queue.get()
        if resp_id == req_id:
            return scores
        response_queue.put((resp_id, scores))