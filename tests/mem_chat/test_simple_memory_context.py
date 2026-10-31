"""Normal stored memories need not contain the legacy memory_time extra field."""

from memos.mem_chat.simple import SimpleMemChat
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata


def test_chat_builds_context_for_default_memory_metadata():
    chat = object.__new__(SimpleMemChat)
    item = TextualMemoryItem(memory="Prefer short hotel answers", metadata=TextualMemoryMetadata())
    prompt = chat._build_system_prompt([item])
    assert "1. Prefer short hotel answers" in prompt
    assert "None" not in prompt


def test_chat_preserves_explicit_legacy_memory_time():
    chat = object.__new__(SimpleMemChat)
    item = TextualMemoryItem(
        memory="Booked a hotel", metadata=TextualMemoryMetadata(memory_time="2026-10-31")
    )
    assert "1. (2026-10-31) Booked a hotel" in chat._build_system_prompt([item])
