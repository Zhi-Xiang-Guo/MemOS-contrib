"""Empty membership is false, including inside logical filter composition."""

import os
import uuid

import pytest

from memos.configs.vec_db import MilvusVecDBConfig
from memos.vec_dbs.item import MilvusVecDBItem
from memos.vec_dbs.milvus import MilvusVecDB


@pytest.mark.skipif(
    not os.getenv("MEMOS_AUDIT_MILVUS_URI"), reason="requires isolated Milvus server"
)
@pytest.mark.parametrize("composition", ["plain", "and", "or"])
def test_empty_membership_keeps_its_boolean_meaning(composition):
    collection = "audit_empty_membership_" + uuid.uuid4().hex[:12]
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
            memory="Synthetic preference",
            original_text="Synthetic input",
            vector=[1.0, 0.0],
            payload={"user_id": "audit-user"},
        )
        db.add(collection, [item])
        db.client.flush(collection_name=collection)
        empty = {"user_id": {"in": []}}
        filters = {
            "plain": empty,
            "and": {"and": [empty, {"user_id": "audit-user"}]},
            "or": {"or": [empty, {"user_id": "audit-user"}]},
        }
        results = db.get_by_filter(collection, filters[composition])
        assert [x.id for x in results] == ([item.id] if composition == "or" else [])
    finally:
        db.delete_collection(collection)
