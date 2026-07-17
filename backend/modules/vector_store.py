"""
图文双向量存储模块

核心设计：
  - 文本向量索引：BGE-M3 编码 -> Chroma Collection (dense + sparse)
  - 图像向量索引：CLIP 编码 -> Chroma Collection (dense)
  - 混合检索：双路独立检索 + RRF 联合排序
  - 降级方案：BM25 关键词检索（向量检索无结果时自动启用）
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# 必须在 import huggingface 相关库之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from PIL import Image

from backend.config import (
    CHROMA_DIR, VECTOR_TOP_K,
    TEXT_EMBEDDING_MODEL, IMAGE_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """单条检索结果"""
    chunk_id: str
    text: str
    score: float                    # 归一化后的检索分数 [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""                 # "text" | "image" | "bm25"


# ============================================================
# 嵌入模型封装
# ============================================================

class TextEmbedder:
    """BGE-M3 文本向量编码器（延迟加载）"""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载文本嵌入模型: {TEXT_EMBEDDING_MODEL}")
            for attempt in range(3):
                try:
                    self._model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"模型加载失败 (尝试 {attempt+1}/3): {e}")
                    import time; time.sleep(2)
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        model = self._load()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class ImageEmbedder:
    """CLIP 图像向量编码器（延迟加载）"""

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPProcessor, CLIPModel
            logger.info(f"加载图像嵌入模型: {IMAGE_EMBEDDING_MODEL}")
            for attempt in range(3):
                try:
                    self._model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
                    self._processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"模型加载失败 (尝试 {attempt+1}/3): {e}")
                    import time; time.sleep(2)
        return self._model, self._processor

    def encode(self, image_paths: List[str]) -> List[List[float]]:
        import torch
        model, processor = self._load()

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.numpy().tolist()


# ============================================================
# BM25 关键词检索（降级方案）
# ============================================================

class BM25Fallback:
    """当向量检索无结果时，回退到 BM25 关键词匹配"""

    def __init__(self):
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._metadatas: List[Dict] = []
        self._bm25 = None

    def index(self, texts: List[str], metadatas: List[Dict]):
        import jieba
        self._corpus = texts
        self._metadatas = metadatas
        self._tokenized_corpus = [list(jieba.cut(t)) for t in texts]

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        import jieba
        if self._bm25 is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 归一化分数
        max_score = np.max(scores) if len(scores) > 0 else 1.0
        if max_score == 0:
            return []

        # Top-K 索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk_id=f"bm25_{idx}",
                    text=self._corpus[idx],
                    score=float(scores[idx] / max_score),
                    metadata=self._metadatas[idx],
                    source="bm25"
                ))
        return results


# ============================================================
# Chroma 双向量存储
# ============================================================

class DualVectorStore:
    """图文双向量存储：文本索引 + 图像索引"""

    def __init__(
        self,
        text_collection_name: str = "paper_texts",
        image_collection_name: str = "paper_images"
    ):
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name

        # 嵌入器（延迟加载）
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()

        # BM25 降级
        self._bm25 = BM25Fallback()

        # Chroma 客户端（延迟连接）
        self._client = None
        self._text_collection = None
        self._image_collection = None

    def _connect(self):
        """连接到 Chroma"""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            chroma_path = str(CHROMA_DIR)
            logger.info(f"连接 Chroma: {chroma_path}")

            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取或创建文本集合
            self._text_collection = self._client.get_or_create_collection(
                name=self.text_collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # 获取或创建图像集合
            self._image_collection = self._client.get_or_create_collection(
                name=self.image_collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(
                f"Chroma 就绪: 文本集合({self._text_collection.count()}条), "
                f"图像集合({self._image_collection.count()}条)"
            )

        return self._client, self._text_collection, self._image_collection

    # ---------- 索引 ----------

    def index_document(
        self,
        document_id: str,
        text_chunks: List[str],
        chunk_metadatas: List[Dict],
        image_path: Optional[str] = None
    ):
        """
        索引一个文档：文本存入文本集合，图像存入图像集合

        Args:
            document_id: 文档唯一ID（如 page_id）
            text_chunks: OCR 文本块列表
            chunk_metadatas: 每个文本块的元数据
            image_path: 可选的页面图像路径
        """
        _, text_col, image_col = self._connect()

        # 1. 文本向量 -> 文本集合
        if text_chunks:
            text_embeddings = self.text_embedder.encode(text_chunks)
            text_ids = [f"{document_id}_chunk_{i}" for i in range(len(text_chunks))]

            for meta in chunk_metadatas:
                meta["document_id"] = document_id

            text_col.add(
                ids=text_ids,
                embeddings=text_embeddings,
                documents=text_chunks,
                metadatas=chunk_metadatas
            )
            logger.info(f"文本索引: {len(text_chunks)} 个块 -> {document_id}")

            # 同步更新 BM25 索引
            all_texts = text_col.get()["documents"] or []
            all_metas = text_col.get()["metadatas"] or []
            self._bm25.index(all_texts, all_metas) if all_texts else None

        # 2. 图像向量 -> 图像集合
        if image_path and Path(image_path).exists():
            image_embedding = self.image_embedder.encode([image_path])
            image_col.add(
                ids=[f"{document_id}_image"],
                embeddings=image_embedding,
                documents=[f"Image of {document_id}"],
                metadatas=[{"document_id": document_id, "image_path": image_path}]
            )
            logger.info(f"图像索引: {image_path} -> {document_id}")

    # ---------- 检索 ----------

    def search_text(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K
    ) -> List[SearchResult]:
        """纯文本语义检索"""
        _, text_col, _ = self._connect()

        query_embedding = self.text_embedder.encode([query])

        raw_results = text_col.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        if raw_results["ids"] and raw_results["ids"][0]:
            for i, chunk_id in enumerate(raw_results["ids"][0]):
                # Chroma cosine distance -> similarity
                distance = raw_results["distances"][0][i]
                similarity = 1.0 - (distance / 2.0)  # 归一化到 [0, 1]

                results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=raw_results["documents"][0][i],
                    score=similarity,
                    metadata=raw_results["metadatas"][0][i],
                    source="text"
                ))

        return results

    def search_image(
        self,
        image_path: str,
        top_k: int = VECTOR_TOP_K
    ) -> List[SearchResult]:
        """以图搜图：用图像向量检索相似图像"""
        _, _, image_col = self._connect()

        query_embedding = self.image_embedder.encode([image_path])

        raw_results = image_col.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        if raw_results["ids"] and raw_results["ids"][0]:
            for i, chunk_id in enumerate(raw_results["ids"][0]):
                distance = raw_results["distances"][0][i]
                similarity = 1.0 - (distance / 2.0)

                results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=raw_results["documents"][0][i],
                    score=similarity,
                    metadata=raw_results["metadatas"][0][i],
                    source="image"
                ))

        return results

    def search_hybrid(
        self,
        query: str,
        image_path: Optional[str] = None,
        top_k: int = VECTOR_TOP_K
    ) -> List[SearchResult]:
        """
        混合检索：文本语义 + 图像特征 + BM25 兜底

        检索策略：
        1. 文本向量检索（主通道）
        2. 如果有图像，图像向量检索（辅助通道）
        3. RRF（Reciprocal Rank Fusion）联合排序
        4. 如果向量检索无结果，降级到 BM25 关键词检索
        """
        all_results = []

        # 文本检索
        text_results = self.search_text(query, top_k=top_k * 2)
        all_results.extend(text_results)

        # 图像检索（如有）
        if image_path and Path(image_path).exists():
            image_results = self.search_image(image_path, top_k=top_k * 2)
            all_results.extend(image_results)

        # RRF 融合排序
        fused = self._rrf_fusion([text_results, all_results[len(text_results):]])

        # 如果向量检索无结果，降级到 BM25
        if not fused:
            logger.info("向量检索无结果，降级到 BM25")
            fused = self._bm25.search(query, top_k=top_k)

        return fused[:top_k]

    def _rrf_fusion(
        self,
        result_groups: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion: 多路检索结果联合排序

        对每个结果，按其在各自列表中的排名计算 RRF 分数：
        RRF_score = sum(1 / (k + rank_i))  for each group

        然后合并去重，按 RRF 分数降序排列
        """
        # chunk_id -> [累计RRF分数, 最佳结果对象]
        fusion_map: Dict[str, List] = {}

        for group in result_groups:
            for rank, result in enumerate(group, start=1):
                rrf_score = 1.0 / (k + rank)

                if result.chunk_id in fusion_map:
                    fusion_map[result.chunk_id][0] += rrf_score
                    # 保留分数更高的结果对象
                    if result.score > fusion_map[result.chunk_id][1].score:
                        fusion_map[result.chunk_id][1] = result
                else:
                    fusion_map[result.chunk_id] = [rrf_score, result]

        # 按 RRF 分数降序排列
        sorted_items = sorted(
            fusion_map.values(),
            key=lambda x: x[0],
            reverse=True
        )

        return [item[1] for item in sorted_items]

    # ---------- 管理 ----------

    def get_document_ids(self) -> List[str]:
        """获取所有已索引的文档ID"""
        _, text_col, _ = self._connect()
        if text_col.count() == 0:
            return []
        metadatas = text_col.get()["metadatas"]
        ids = set()
        for m in metadatas:
            if m and "document_id" in m:
                ids.add(m["document_id"])
        return list(ids)

    def delete_document(self, document_id: str):
        """删除一个文档的所有索引"""
        _, text_col, image_col = self._connect()

        # 删除文本
        text_results = text_col.get(
            where={"document_id": document_id}
        )
        if text_results["ids"]:
            text_col.delete(ids=text_results["ids"])
            logger.info(f"删除文本索引: {document_id} ({len(text_results['ids'])}条)")

        # 删除图像
        image_results = image_col.get(
            where={"document_id": document_id}
        )
        if image_results["ids"]:
            image_col.delete(ids=image_results["ids"])
            logger.info(f"删除图像索引: {document_id}")


# 全局单例
vector_store = DualVectorStore()
