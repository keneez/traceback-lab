import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from traceback_lab.calibration import fit_calibrator, Calibrator
from traceback_lab.ml.encoders import BiEncoder
from traceback_lab.ml.reranker import CrossEncoder
from traceback_lab.ml.training import (TrainConfig, multi_positive_loss, train_bi, train_cross,
                                       labeled_examples, make_batch)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
    directory = tmp_path / "tiny"
    directory.mkdir()
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["def request(connection): return reset_pool(connection)", "timeout connection reset pool",
         "def format_date(value): return str(value)"],
        vocab_size=300, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<encoder-only>"])
    tokenizer.save_model(str(directory))
    fast = RobertaTokenizerFast(vocab_file=str(directory / "vocab.json"),
                                merges_file=str(directory / "merges.txt"),
                                bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                pad_token="<pad>", mask_token="<mask>")
    fast.save_pretrained(directory)
    config = RobertaConfig(vocab_size=len(fast), hidden_size=24, num_hidden_layers=1,
                           num_attention_heads=4, intermediate_size=32,
                           max_position_embeddings=514, pad_token_id=fast.pad_token_id,
                           bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
                           hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    RobertaModel(config).save_pretrained(directory)
    return directory


def test_invalid_negative_has_zero_influence_and_multiple_positives_are_supported():
    q = torch.tensor([[1., 0.]], requires_grad=True)
    c = torch.tensor([[1., 0.], [100., 0.], [0., 1.]], requires_grad=True)
    loss = multi_positive_loss(q, c, torch.tensor([[True, False, True]]),
                              torch.tensor([[True, False, True]]), temperature=1)
    loss.backward()
    assert torch.all(c.grad[1] == 0)
    assert torch.isfinite(loss)


def test_same_fix_group_cannot_create_in_batch_false_negatives(tiny_checkpoint, corpus):
    import random
    _, store, cases, _ = corpus
    first = labeled_examples(cases, store, "train")[0]
    alternate = next(s["id"] for s in first["symbols"] if s["id"] not in first["positives"])
    second = {**first, "case": {**first["case"], "id": "related-case"},
              "positives": {alternate}}
    _, _, positive, valid = make_batch(
        [first, second], BiEncoder(tiny_checkpoint), random.Random(31), TrainConfig(negatives=0))
    assert int(positive.sum()) == 2
    assert int(valid.sum()) == 2


def test_token_budgets_and_vectors_are_finite(tiny_checkpoint, corpus):
    _, store, _, snapshot = corpus
    model = BiEncoder(tiny_checkpoint)
    symbol = store.symbols(snapshot["id"])[0]
    symbol["search_code"] *= 500
    assert len(model.symbol_ids(symbol)) <= 512
    assert len(model.query_ids("timeout " * 1000)) <= 256
    vectors = model.encode_queries(["timeout", "request"])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)


def test_both_training_entrypoints_save_reloadable_checkpoints(tiny_checkpoint, corpus, tmp_path):
    _, store, cases, _ = corpus
    torch.set_num_threads(1)
    config = TrainConfig(epochs=1, micro_batch=1, accumulation=1, negatives=1,
                         max_steps=1, gradient_checkpointing=False)
    output = tmp_path / "trained-bi"
    report = train_bi(tiny_checkpoint, output, cases, store, config)
    assert report["steps"] == 1
    assert BiEncoder(output).identity != BiEncoder(tiny_checkpoint).identity
    cross = tmp_path / "trained-cross"
    report = train_cross(tiny_checkpoint, cross, cases, store, config)
    assert report["steps"] == 1
    reranker = CrossEncoder(cross)
    predictions = reranker.score("request timeout", store.symbols(cases.list()[0]["snapshot_id"]))
    assert all(np.isfinite(p["rerank_score"]) for p in predictions)


def test_untrained_cross_encoder_is_rejected(tiny_checkpoint):
    with pytest.raises(ValueError, match="trained"):
        CrossEncoder(tiny_checkpoint)


def test_calibrator_rejects_overlap_between_independent_parts(tmp_path):
    records = [{"case_id": "case", "group_id": "group", "correct": True, "result": {}}]
    with pytest.raises(ValueError, match="overlap"):
        fit_calibrator(records, records, tmp_path / "calibration.json")


def test_calibrator_fits_and_rejects_changed_pipeline(tmp_path):
    pytest.importorskip("sklearn")
    from dataclasses import asdict
    from traceback_lab.config import SearchConfig
    def record(index, prefix):
        correct = index % 2 == 0
        result = {"query": "connection timeout", "models": {"encoder": "e", "reranker": "r"},
                  "config": asdict(SearchConfig()),
                  "candidates": [
                      {"rrf_score": .03, "rerank_score": 2.0 if correct else -2.0,
                       "ranks": {"bm25": 1}, "trace": [], "start_line": 1, "end_line": 10},
                      {"rrf_score": .02, "rerank_score": -3.0, "ranks": {}, "trace": [],
                       "start_line": 1, "end_line": 10}]}
        return {"case_id": f"{prefix}-{index}", "group_id": f"{prefix}-group-{index}",
                "correct": correct, "result": result}
    fit = [record(i, "fit") for i in range(40)]
    threshold = [record(i, "threshold") for i in range(20)]
    output = tmp_path / "calibrator.json"
    fit_calibrator(fit, threshold, output)
    calibrator = Calibrator(output)
    prediction = calibrator.predict(record(0, "test")["result"])
    assert prediction["status"] == "calibrated"
    assert prediction["probability"] > .5
    assert prediction["accepted"]
    changed = record(0, "other")["result"]
    changed["config"]["budget"] = 20
    assert calibrator.predict(changed)["probability"] is None



def test_shared_vector_index_works_with_unixcoder(tiny_checkpoint, corpus, monkeypatch, tmp_path):
    from traceback_lab import vectors
    from traceback_lab.config import SearchConfig
    from traceback_lab.retrieval import SearchEngine
    monkeypatch.setattr(vectors, "STATE_ROOT", tmp_path)
    _, store, _, snapshot = corpus
    model = BiEncoder(tiny_checkpoint)
    output = SearchEngine(store, model).search(snapshot["id"], "request", config=SearchConfig(mode="dense"))
    meta, matrix = store.vectors(snapshot["id"], model.identity)
    assert output["methods"] == ["dense"] and output["candidates"]
    assert meta["serialization"] == model.serialization
    assert matrix.shape == (snapshot["symbol_count"], model.dimensions)


def test_numpy_compact_inference_matches_trained_torch_forward():
    from traceback_lab.ml.compact import CompactEncoder, tokens
    from traceback_lab.ml.train_compact import DualEncoder
    model, mirror = CompactEncoder(), DualEncoder()
    with torch.no_grad():
        mirror.embedding.weight.copy_(torch.from_numpy(model.embedding))
        mirror.query.copy_(torch.from_numpy(model.query_head))
        mirror.code.copy_(torch.from_numpy(model.code_head))
        texts = ["request timeout", "stream_response decode Unicode"]
        expected = mirror.encode([tokens(t, 80) for t in texts], mirror.query).numpy()
        np.testing.assert_allclose(model.encode_queries(texts), expected, rtol=1e-5, atol=1e-6)
