"""Multi-memory cube export must not overwrite one component with another."""

import json

from types import SimpleNamespace

import pytest

from memos.configs.memory import NaiveTextMemoryConfig, PreferenceTextMemoryConfig
from memos.exceptions import MemCubeError
from memos.mem_cube.general import GeneralMemCube


def cube_with_file_memories():
    cube = object.__new__(GeneralMemCube)
    text_config = NaiveTextMemoryConfig.model_construct(extractor_llm=None)
    pref_config = PreferenceTextMemoryConfig.model_construct(
        extractor_llm=None,
        vector_db=None,
        embedder=None,
        extractor=None,
        adder=None,
        retriever=None,
    )

    def write_config(path):
        from pathlib import Path

        Path(path).write_text("{}")

    cube.config = SimpleNamespace(config_filename="config.json", to_json_file=write_config)

    def component(config, data):
        def dump(folder):
            from pathlib import Path

            (Path(folder) / config.memory_filename).write_text(json.dumps(data))

        return SimpleNamespace(config=config, dump=dump)

    cube._text_mem = component(text_config, [{"memory": "text must survive"}])
    cube._pref_mem = component(
        pref_config, {"explicit_preference": [{"memory": "preference must survive"}]}
    )
    cube._act_mem = None
    cube._para_mem = None
    return cube


def test_default_text_and_preference_exports_keep_both_payloads(tmp_path):
    cube = cube_with_file_memories()
    cube.dump(str(tmp_path), memory_types=["text_mem", "pref_mem"])
    assert json.loads((tmp_path / cube.text_mem.config.memory_filename).read_text()) == [
        {"memory": "text must survive"}
    ]
    assert json.loads((tmp_path / cube.pref_mem.config.memory_filename).read_text()) == {
        "explicit_preference": [{"memory": "preference must survive"}]
    }


def test_explicit_collision_is_rejected_before_any_file_is_written(tmp_path):
    cube = cube_with_file_memories()
    cube.pref_mem.config.memory_filename = cube.text_mem.config.memory_filename
    with pytest.raises(MemCubeError, match="collid|same|duplicate"):
        cube.dump(str(tmp_path), memory_types=["text_mem", "pref_mem"])
    assert list(tmp_path.iterdir()) == []
