"""Opt-in real Milvus count semantics after deleting memories."""

import os
import uuid

import pytest

from memos.configs.vec_db import MilvusVecDBConfig
from memos.vec_dbs.item import MilvusVecDBItem
from memos.vec_dbs.milvus import MilvusVecDB


@pytest.mark.skipif(
    not os.getenv("MEMOS_AUDIT_MILVUS_URI"), reason="requires isolated Milvus server"
)
@pytest.mark.parametrize("with_filter", [False, True])
def test_count_excludes_deleted_memories(with_filter):
    collection = "audit_logical_count_" + uuid.uuid4().hex[:12]
    db = MilvusVecDB(
        MilvusVecDBConfig(
            uri=os.environ["MEMOS_AUDIT_MILVUS_URI"],
            collection_name=[collection],
            vector_dimension=2,
            distance_metric="cosine",
        )
    )
    try:
        items = [
            MilvusVecDBItem(
                id=str(uuid.uuid4()),
                memory=f"count audit {i}",
                original_text="synthetic",
                vector=[1.0, 0.0],
                payload={"user_id": "count-test"},
            )
            for i in range(3)
        ]
        db.add(collection, items)
        db.client.flush(collection_name=collection)
        db.delete(collection, [items[0].id])
        db.client.flush(collection_name=collection)
        assert (
            db.client.query(
                collection_name=collection,
                filter="",
                output_fields=["count(*)"],
                consistency_level="Strong",
            )[0]["count(*)"]
            == 2
        )
        assert db.count(collection, {"user_id": "count-test"} if with_filter else None) == 2
    finally:
        db.delete_collection(collection)
