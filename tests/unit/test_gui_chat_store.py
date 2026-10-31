"""Unit tests for forge_gui.api.chat_store (persistence, ratings, SFT export)."""
import json

import pytest

from forge_gui.api.chat_store import ChatStore


@pytest.fixture()
def store(tmp_path):
    return ChatStore(root=tmp_path)


def test_create_and_persist(store, tmp_path):
    conv = store.create(title="Test chat", model="forgelm_v2")
    assert conv["id"] in [c["id"] for c in store.conversations]
    assert (tmp_path / "chats" / "conversations.json").is_file()

    # reload from disk
    store2 = ChatStore(root=tmp_path)
    assert len(store2.conversations) == 1
    assert store2.conversations[0]["title"] == "Test chat"


def test_append_and_rate_toggle(store):
    conv = store.create()
    i_user = store.append_message(conv["id"], "user", "hello")
    i_asst = store.append_message(conv["id"], "assistant", "hi there")

    assert i_user == 0 and i_asst == 1

    # rate good → persisted
    new = store.rate_message(conv["id"], i_asst, "good")
    assert new == "good"
    assert store.get(conv["id"])["messages"][1]["rating"] == "good"

    # same rating again → toggle off
    new = store.rate_message(conv["id"], i_asst, "good")
    assert new is None

    # bad rating
    new = store.rate_message(conv["id"], i_asst, "bad")
    assert new == "bad"


def test_export_only_good_turns(store, tmp_path):
    conv = store.create(title="rated")
    store.append_message(conv["id"], "user", "write fib")
    i1 = store.append_message(conv["id"], "assistant", "def fib(n): ...")
    store.append_message(conv["id"], "user", "now memoize it")
    i2 = store.append_message(conv["id"], "assistant", "use functools.lru_cache")

    store.rate_message(conv["id"], i1, "good")
    store.rate_message(conv["id"], i2, "bad")

    out, n = store.export_training_data()
    assert out.parent.name == "sft"
    assert out.suffix == ".jsonl"
    assert n == 1
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    assert len(lines) == 1
    msgs = lines[0]["messages"]
    # prefix up to the good turn: user + assistant (no system configured)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "def fib(n): ..."


def test_export_format_matches_sft_train(store):
    """Exported lines must be loadable by sft_train.load_examples."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from forge.training.runners.sft_train import load_examples

    conv = store.create()
    store.append_message(conv["id"], "user", "q1")
    i = store.append_message(conv["id"], "assistant", "a1")
    store.rate_message(conv["id"], i, "good")
    out, n = store.export_training_data()
    assert n == 1
    examples = load_examples([str(out)])
    assert len(examples) == 1
    assert examples[0]["type"] == "multi_turn"
    assert examples[0]["messages"][1]["content"] == "a1"


def test_export_with_system_prompt(store):
    conv = store.create()
    store.append_message(conv["id"], "system", "You are Forge.")
    store.append_message(conv["id"], "user", "q")
    i = store.append_message(conv["id"], "assistant", "a")
    store.rate_message(conv["id"], i, "good")
    out, _n = store.export_training_data()
    msgs = next(json.loads(l)["messages"] for l in
            out.read_text(encoding="utf-8").splitlines() if l.strip())
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are Forge."


def test_export_selected_convs_only(store):
    c1 = store.create()
    c2 = store.create()
    for c in (c1, c2):
        store.append_message(c["id"], "user", "q")
        i = store.append_message(c["id"], "assistant", "a")
        store.rate_message(c["id"], i, "good")
    _out, n = store.export_training_data(conv_ids=[c1["id"]])
    assert n == 1


def test_auto_title_and_rename(store):
    conv = store.create()
    store.append_message(conv["id"], "user", "fix the fibonacci bug\nmore detail")
    store.touch(conv["id"])
    assert store.get(conv["id"])["title"].startswith("fix the fibonacci bug")
    store.rename(conv["id"], "custom name")
    assert store.get(conv["id"])["title"] == "custom name"


def test_delete(store):
    conv = store.create()
    assert store.delete(conv["id"]) is True
    assert store.get(conv["id"]) is None
    assert store.delete(conv["id"]) is False
