"""BGE-M3 dense + sparse Qdrant adapter.

Imports are lazy so API startup and metadata review still work when the embedding
model is not installed on a developer machine.
"""
from __future__ import annotations
from functools import lru_cache
from uuid import UUID
from qdrant_client import QdrantClient, models
from ..config import get_settings

COLLECTION = "legal_provisions_v1"


class BgeM3Embedder:
    def __init__(self) -> None:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())

    def encode(self, texts: list[str]):
        result = self.model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return result["dense_vecs"], result["lexical_weights"]


@lru_cache(maxsize=1)
def get_embedder() -> BgeM3Embedder:
    """Load BGE-M3 once per API process instead of once per user request."""
    return BgeM3Embedder()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


class HybridVectorStore:
    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self._embedder: BgeM3Embedder | None = None

    @property
    def embedder(self) -> BgeM3Embedder:
        # Status-only operations such as withdrawing a faulty version must not
        # load the 500MB+ embedding model.
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    @staticmethod
    def _sparse_vector(weights: dict[int, float]) -> models.SparseVector:
        return models.SparseVector(indices=[int(key) for key in weights.keys()], values=list(weights.values()))

    def upsert(self, provision_id: UUID, text: str, payload: dict) -> None:
        self.upsert_many([(provision_id, text, payload)])

    def upsert_many(self, records: list[tuple[UUID, str, dict]]) -> None:
        """Embed and write a batch so publishing does not run one inference per clause."""
        if not records:
            return
        self.ensure_collection()
        dense, sparse = self.embedder.encode([text for _, text, _ in records])
        self.client.upsert(
            COLLECTION,
            [
                models.PointStruct(
                    id=str(provision_id),
                    vector={"dense": dense[index].tolist(), "sparse": self._sparse_vector(sparse[index])},
                    payload=payload,
                )
                for index, (provision_id, _, payload) in enumerate(records)
            ],
        )

    def search(self, question: str, as_of_iso: str, limit: int = 16, expansion_query: str | None = None):
        self.ensure_collection()
        queries = [question]
        if expansion_query and expansion_query != question:
            queries.append(expansion_query)
        dense, sparse = self.embedder.encode(queries)
        # Effective-date validation is repeated from PostgreSQL after recall. That
        # is authoritative and handles open-ended versions without Qdrant date
        # serialization assumptions.
        effective_filter = models.Filter(must=[models.FieldCondition(key="status", match=models.MatchValue(value="PUBLISHED"))])
        # Qdrant server-side RRF combines independent dense and sparse rankings.
        prefetches = []
        for index in range(len(queries)):
            # Preserve the original question as an independent ranking. The
            # optional expanded query only contributes extra recall; it never
            # replaces the user's wording.
            prefetches.extend([
                models.Prefetch(query=dense[index].tolist(), using="dense", filter=effective_filter, limit=limit),
                models.Prefetch(query=self._sparse_vector(sparse[index]), using="sparse", filter=effective_filter, limit=limit),
            ])
        result = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return result.points

    def set_version_status(self, version_id: UUID, status: str) -> None:
        self.client.set_payload(COLLECTION, {"status": status}, models.Filter(
            must=[models.FieldCondition(key="version_id", match=models.MatchValue(value=str(version_id)))]
        ))
