"""Opt-in real-server regression for supported Milvus filter string values."""

import os
import uuid

import pytest

from memos.configs.vec_db import MilvusVecDBConfig
from memos.vec_dbs.item import MilvusVecDBItem
from memos.vec_dbs.milvus import MilvusVecDB


@pytest.mark.skipif(
    not os.getenv("MEMOS_AUDIT_MILVUS_URI"), reason="requires isolated Milvus server"
)
@pytest.mark.parametrize("operator", ["eq", "in", "like"])
def test_milvus_filters_preserve_apostrophes(operator):
    collection = "audit_filter_literals_" + uuid.uuid4().hex[:12]
    db = MilvusVecDB(
        MilvusVecDBConfig(
            uri=os.environ["MEMOS_AUDIT_MILVUS_URI"],
            collection_name=[collection],
            vector_dimension=2,
            distance_metric="cosine",
        )
    )
    try:
        item = MilvusVecDBItem(
            id=str(uuid.uuid4()),
            memory="Concise answers",
            original_text="Keep it short",
            vector=[1.0, 0.0],
            payload={"user_id": "O'Brien"},
        )
        db.add(collection, [item])
        db.client.flush(collection_name=collection)
        filters = {"eq": "O'Brien", "in": {"in": ["O'Brien"]}, "like": {"like": "O'B%"}}
        results = db.get_by_filter(collection, {"user_id": filters[operator]})
        assert [hit.id for hit in results] == [item.id]
    finally:
        db.delete_collection(collection)
