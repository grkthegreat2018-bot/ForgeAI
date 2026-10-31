"""Tests for train_flux.py corpus tooling — topic filter, dict flatten,
jsonl field extraction."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train_flux import (
    _auto_qa,
    _flatten_dict_entry,
    _hf_row_text,
    _iter_texts,
    _render_chat,
    _resolve_topics,
    _topic_scorer,
)


class TestTopicScorer:
    def test_title_match_accepts(self):
        sc = _topic_scorer(_resolve_topics("ai"))
        assert sc("Transformer (deep learning)", "anything")

    def test_multiword_title_keywords(self):
        sc = _topic_scorer(_resolve_topics("ai"))
        assert sc("Large language model", "x")
        assert sc("History of machine learning", "x")

    def test_off_topic_rejected(self):
        sc = _topic_scorer(_resolve_topics("ai,code"))
        assert not sc("Battle of Hastings",
                      "fought in 1066 between the Norman-French army "
                      "and an English army under Harold Godwinson")
        assert not sc("Manchester United",
                      "professional football club based in Old Trafford")

    def test_text_needs_distinct_hits(self):
        sc = _topic_scorer(_resolve_topics("science"))
        # one keyword repeated does not reach min_hits=2
        assert not sc("Untitled", "quantum quantum quantum quantum")
        assert sc("Untitled", "the quantum experiment confirmed "
                              "the hypothesis")

    def test_custom_keywords(self):
        sc = _topic_scorer(_resolve_topics("flux capacitor"))
        assert sc("Flux capacitor", "plain text")
        # single distinct custom keyword in text < min_hits
        assert not sc("Untitled", "flux capacitor flux capacitor")

    def test_preset_merge(self):
        spec = _resolve_topics("ai,code")
        assert "transformer" in spec["title"]
        assert "compiler" in spec["title"]


class TestDictFlatten:
    def test_glosses_render(self):
        row = {"word": "backpropagation", "pos": "noun",
               "senses": [{"glosses": ["the algorithm for training "
                                       "neural networks",
                                       "second gloss"]}]}
        out = _flatten_dict_entry(row)
        assert out.startswith("backpropagation (noun):")
        assert "algorithm" in out

    def test_missing_word_or_gloss(self):
        assert _flatten_dict_entry({"pos": "noun"}) is None
        assert _flatten_dict_entry({"word": "x"}) is None


class TestIterTextsFilter:
    def _write_jsonl(self, tmp_path, rows):
        p = tmp_path / "corpus.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows),
                     encoding="utf-8")
        return p

    def test_jsonl_title_scored(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        rows = [
            {"title": "Neural network",
             "text": "A neural network is a machine learning model."},
            {"title": "Cricket (sport)",
             "text": "Cricket is a bat-and-ball game played between "
                     "two teams."},
        ]
        p = self._write_jsonl(tmp_path, rows)
        got = list(_iter_texts(p, None, False, sc))
        assert len(got) == 1 and "neural network" in got[0]

    def test_jsonl_no_topic_yields_all(self, tmp_path):
        rows = [{"title": "Cricket", "text": "bat and ball game"},
                {"title": "Neural network", "text": "ml model"}]
        p = self._write_jsonl(tmp_path, rows)
        assert len(list(_iter_texts(p, None, False, None))) == 2

    def test_dict_flatten_filtered(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        rows = [
            {"word": "backpropagation", "pos": "noun",
             "senses": [{"glosses": ["algorithm for training "
                                     "neural networks"]}]},
            {"word": "aardvark", "pos": "noun",
             "senses": [{"glosses": ["a nocturnal mammal"]}]},
        ]
        p = self._write_jsonl(tmp_path, rows)
        got = list(_iter_texts(p, None, True, sc))
        assert len(got) == 1 and got[0].startswith("backpropagation")

    def test_txt_unfiltered(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        p = tmp_path / "notes.txt"
        p.write_text("a recipe for chocolate cake", encoding="utf-8")
        assert len(list(_iter_texts(p, None, False, sc))) == 1


class TestHfTemplate:
    def test_qa_template(self):
        row = {"question": "What is entropy?",
               "answer": "a measure of disorder"}
        out = _hf_row_text(row, "Q: {question} A: {answer}")
        assert out == "Q: What is entropy? A: a measure of disorder"

    def test_squad_style_answers_dict(self):
        # squad rows: answers = {"text": [...], "answer_start": [...]}
        row = {"question": "Capital of France?",
               "answers": {"text": ["Paris"], "answer_start": [0]}}
        out = _hf_row_text(row, "Q: {question} A: {answers}")
        assert out == "Q: Capital of France? A: Paris"

    def test_triviaqa_style_value_list(self):
        row = {"question": "H2O is?",
               "answer": {"value": "water", "aliases": ["w"]}}
        out = _hf_row_text(row, "Q: {question} A: {answer}")
        assert "water" in out

    def test_missing_key_blank(self):
        out = _hf_row_text({"a": "1"}, "{a}|{missing}")
        assert out == "1|"


class TestChatRender:
    TURNS = [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "What is 2+2?"},
             {"role": "assistant", "content": "4."}]

    def test_natural_default(self):
        out = _render_chat(self.TURNS)
        assert out == ("Context: You are helpful.\n"
                       "Question: What is 2+2?\nAnswer: 4.\n")
        assert "im_start" not in out and "im_end" not in out

    def test_chatml_only_when_asked(self):
        out = _render_chat(self.TURNS, "chatml")
        assert "im_start" in out and "im_end" in out
        assert "system" in out

    def test_plain_content_only(self):
        out = _render_chat(self.TURNS, "plain")
        assert out == "You are helpful.\nWhat is 2+2?\n4.\n"

    def test_role_aliases(self):
        out = _render_chat([{"role": "human", "value": "hi"},
                            {"role": "gpt", "value": "hello"}])
        assert "Question: hi" in out and "Answer: hello" in out

    def test_sharegpt_from_key(self):
        # OpenHermes / ShareGPT schema: 'from'/'value', not role/content
        out = _render_chat([{"from": "system", "value": "ctx"},
                            {"from": "human", "value": "q"},
                            {"from": "gpt", "value": "a"}])
        assert out == "Context: ctx\nQuestion: q\nAnswer: a\n"

    def test_malformed_turns_skipped(self):
        out = _render_chat(["junk", {"role": "user", "content": "  "},
                            {"role": "user", "content": "ok"}])
        assert out == "Question: ok\n"
        assert _render_chat([]) == ""

    def test_jsonl_messages_natural(self, tmp_path):
        p = tmp_path / "chat.jsonl"
        p.write_text(json.dumps({"messages": self.TURNS}),
                     encoding="utf-8")
        got = list(_iter_texts(p, None, False, None, "natural"))
        assert len(got) == 1
        assert "Question: What is 2+2?" in got[0]
        assert "im_start" not in got[0]

    def test_jsonl_conversations_respected(self, tmp_path):
        p = tmp_path / "chat.jsonl"
        row = {"conversations": [{"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"}]}
        p.write_text(json.dumps(row), encoding="utf-8")
        got = list(_iter_texts(p, None, False, None, "natural"))
        assert "Question: q1" in got[0] and "Answer: a1" in got[0]

    def test_json_messages_rendered(self, tmp_path):
        p = tmp_path / "chat.json"
        p.write_text(json.dumps([{"messages": self.TURNS}]),
                     encoding="utf-8")
        got = list(_iter_texts(p, None, False, None, "natural"))
        assert len(got) == 1 and "Answer: 4." in got[0]

    def test_jsonl_chat_topic_filter(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        p = tmp_path / "chat.jsonl"
        rows = [{"messages": [{"role": "user", "content":
                               "explain neural networks"},
                             {"role": "assistant", "content":
                               "a neural network is a machine "
                               "learning model"}]},
                {"messages": [{"role": "user", "content":
                               "best cake recipe"}]}]
        p.write_text("\n".join(json.dumps(r) for r in rows),
                     encoding="utf-8")
        got = list(_iter_texts(p, None, False, sc, "natural"))
        assert len(got) == 1 and "neural network" in got[0]


class TestAutoQa:
    def test_instruction_response_context(self):
        out = _auto_qa({"instruction": "Name a color",
                        "context": "Pick one", "response": "Blue"})
        assert out == ("Context: Pick one\n"
                       "Question: Name a color\nAnswer: Blue")

    def test_alpaca_schema(self):
        out = _auto_qa({"instruction": "Do x", "input": "with y",
                        "output": "done"})
        assert "Context: with y" in out and "Answer: done" in out

    def test_question_answer(self):
        assert _auto_qa({"question": "Q", "answer": "A"}) == \
            "Question: Q\nAnswer: A"

    def test_prompt_response(self):
        assert "Answer: R" in _auto_qa({"prompt": "P", "response": "R"})

    def test_missing_pair_returns_none(self):
        assert _auto_qa({"text": "x"}) is None
        assert _auto_qa({"question": "Q"}) is None
