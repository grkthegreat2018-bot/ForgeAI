"""Unit tests for the Vast.ai cloud connector (research/cloud/vast_connector.py).

All tests run on CPU with NO network access. The ``vastai.VastAI`` SDK is
mocked via patching, and paramiko is mocked with lightweight fakes so the
full ensure→sync→train→download→stop lifecycle can be exercised offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research.cloud.vast_connector import (
    REMOTE_CKPT_DIR,
    REMOTE_DATA_DIR,
    REMOTE_OUT_DIR,
    FileEntry,
    RemoteTrainingSpec,
    VastConnector,
    _compute_file_hash,
    _compute_local_manifest,
    build_spec_from_args,
)


# ─── Fake VastAI SDK ───────────────────────────────────────────────────────
class FakeVastAI:
    """Minimal fake of vastai.VastAI for testing.

    Tracks all calls and returns configurable responses.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, tuple, dict]] = []
        self._offers: list[dict] = []
        self._instances: list[dict] = []
        self._ssh_url = "ssh://root@1.2.3.4:12345"
        self._logs = "step 0 loss 5.0\nstep 1 loss 4.5\n"

    def set_offers(self, offers: list[dict]):
        self._offers = offers

    def set_instances(self, instances: list[dict]):
        self._instances = instances

    def search_offers(
        self, query=None, type="on-demand", order="score-", limit=None, storage=5.0, no_default=False, **kwargs
    ):
        self.calls.append(("search_offers", (), {"query": query, "order": order}))
        return list(self._offers)

    def create_instance(self, id, image=None, disk=10, **kwargs):
        self.calls.append(("create_instance", (id,), {"image": image, "disk": disk, **kwargs}))
        return {"success": True, "new_contract": 12345}

    def show_instances(self):
        self.calls.append(("show_instances", (), {}))
        return list(self._instances)

    def show_instance(self, id):
        self.calls.append(("show_instance", (id,), {}))
        for inst in self._instances:
            if inst.get("id") == id:
                return inst
        return None

    def destroy_instance(self, id):
        self.calls.append(("destroy_instance", (id,), {}))
        self._instances = [i for i in self._instances if i.get("id") != id]

    def stop_instance(self, id):
        self.calls.append(("stop_instance", (id,), {}))
        for inst in self._instances:
            if inst.get("id") == id:
                inst["actual_status"] = "stopped"

    def start_instance(self, id):
        self.calls.append(("start_instance", (id,), {}))
        for inst in self._instances:
            if inst.get("id") == id:
                inst["actual_status"] = "running"

    def ssh_url(self, id):
        self.calls.append(("ssh_url", (id,), {}))
        return self._ssh_url

    def label_instance(self, id, label):
        self.calls.append(("label_instance", (id,), {"label": label}))

    def logs(self, instance_id, tail=None, filter=None, daemon_logs=False):
        self.calls.append(("logs", (instance_id,), {"tail": tail, "filter": filter}))
        return self._logs

    def create_volume(self, id, size=15, name=None):
        self.calls.append(("create_volume", (id,), {"size": size, "name": name}))
        return {"id": 999}

    def show_volumes(self, type="all"):
        self.calls.append(("show_volumes", (), {"type": type}))
        return []


# ─── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def ssh_key(tmp_path):
    key = tmp_path / "id_ed25519"
    key.write_text("FAKE PRIVATE KEY")
    return str(key)


@pytest.fixture
def fake_vast():
    return FakeVastAI()


@pytest.fixture
def connector(ssh_key, fake_vast):
    conn = VastConnector(api_key="test-key", ssh_key=ssh_key)
    conn._vast = fake_vast  # inject fake SDK, skip lazy creation
    return conn


# ─── search_offers ─────────────────────────────────────────────────────────
def test_search_offers_filters_and_sorts(connector, fake_vast):
    # Uses real Vast.ai API values: 4090 reports gpu_total_ram=24564 MB.
    # With /1000 conversion: 24564/1000 = 24.564 GB (decimal, matches Vast.ai).
    fake_vast.set_offers(
        [
            {
                "id": 1,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24564,
                "dph_total": 0.30,
                "reliability": 0.98,
                "dlperf": 50,
            },
            {
                "id": 2,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24564,
                "dph_total": 0.60,
                "reliability": 0.99,
                "dlperf": 50,
            },
            {
                "id": 3,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 8192,
                "dph_total": 0.10,
                "reliability": 0.95,
                "dlperf": 20,
            },
            {
                "id": 4,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24564,
                "dph_total": 0.25,
                "reliability": 0.50,
                "dlperf": 50,
            },
        ]
    )
    offers = connector.search_offers(max_price=0.5, min_vram_gb=20, min_reliability=0.9)
    assert len(offers) == 1
    assert offers[0].id == 1
    assert offers[0].vram_gb == pytest.approx(24.564)
    assert offers[0].dph_total == pytest.approx(0.30)


def test_search_offers_sorted_by_perf_per_dollar(connector, fake_vast):
    """When sort_by_perf=True, best perf/$ comes first."""
    fake_vast.set_offers(
        [
            {
                "id": 10,
                "gpu_name": "H100",
                "num_gpus": 1,
                "gpu_total_ram": 81920,
                "dph_total": 0.45,
                "reliability": 0.99,
                "dlperf": 100,
            },
            {
                "id": 11,
                "gpu_name": "A100",
                "num_gpus": 1,
                "gpu_total_ram": 81920,
                "dph_total": 0.40,
                "reliability": 0.99,
                "dlperf": 60,
            },
        ]
    )
    offers = connector.search_offers(max_price=0.5, min_vram_gb=24, min_reliability=0.9, sort_by_perf=True)
    # H100 has better perf/$ (100/0.45=222 vs 60/0.40=150)
    assert [o.id for o in offers] == [10, 11]


def test_search_offers_sorted_cheapest(connector, fake_vast):
    """When sort_by_perf=False, cheapest $/hr comes first."""
    fake_vast.set_offers(
        [
            {
                "id": 10,
                "gpu_name": "H100",
                "num_gpus": 1,
                "gpu_total_ram": 81920,
                "dph_total": 0.45,
                "reliability": 0.99,
                "dlperf": 100,
            },
            {
                "id": 11,
                "gpu_name": "H100",
                "num_gpus": 1,
                "gpu_total_ram": 81920,
                "dph_total": 0.40,
                "reliability": 0.99,
                "dlperf": 100,
            },
        ]
    )
    offers = connector.search_offers(max_price=0.5, min_vram_gb=24, min_reliability=0.9, sort_by_perf=False)
    assert [o.id for o in offers] == [11, 10]


# ─── create_instance ───────────────────────────────────────────────────────
def test_create_instance_parses_new_contract(connector, fake_vast):
    inst_id = connector.create_instance(99, disk_gb=50, label="forgeai-test")
    assert inst_id == 12345
    # Verify SDK was called with correct params
    create_calls = [c for c in fake_vast.calls if c[0] == "create_instance"]
    assert len(create_calls) == 1
    _, args, kwargs = create_calls[0]
    assert args[0] == 99
    assert kwargs["disk"] == 50
    assert kwargs["label"] == "forgeai-test"
    assert kwargs["runtype"] == "ssh_direct"


def test_create_instance_with_volume(connector, fake_vast):
    inst_id = connector.create_instance(99, volume_id=42)
    assert inst_id == 12345
    create_calls = [c for c in fake_vast.calls if c[0] == "create_instance"]
    _, _, kwargs = create_calls[0]
    assert kwargs["volume_info"]["volume_id"] == 42
    assert kwargs["volume_info"]["mount_path"] == "/workspace"


# ─── get_ssh_info ──────────────────────────────────────────────────────────
def test_get_ssh_info_prefers_direct_port(connector, fake_vast):
    """Direct port mapping (from ports field) is preferred over proxy."""
    fake_vast._ssh_url = "ssh://root@proxy.vast.ai:30700"
    fake_vast.set_instances(
        [
            {
                "id": 5,
                "public_ipaddr": "41.138.66.130",
                "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "17573"}]},
            },
        ]
    )
    host, port = connector.get_ssh_info(5)
    assert host == "41.138.66.130"
    assert port == 17573


def test_get_ssh_info_fallback_to_proxy(connector, fake_vast):
    """No direct port mapping → fall back to ssh_url proxy."""
    fake_vast._ssh_url = "ssh://root@5.6.7.8:9999"
    fake_vast.set_instances(
        [
            {"id": 5, "public_ipaddr": "1.2.3.4", "ports": {}},
        ]
    )
    host, port = connector.get_ssh_info(5)
    assert host == "5.6.7.8"
    assert port == 9999


def test_get_ssh_info_fallback_to_instance_fields(connector, fake_vast):
    """No direct port, no ssh_url → use ssh_host + ssh_port from instance."""
    fake_vast._ssh_url = ""
    fake_vast.set_instances(
        [
            {"id": 5, "ssh_host": "9.8.7.6", "ssh_port": 8888, "ports": {}, "actual_status": "running"},
        ]
    )
    host, port = connector.get_ssh_info(5)
    assert host == "9.8.7.6"
    assert port == 8888


# ─── wait_for_running ──────────────────────────────────────────────────────
def test_wait_for_running_poll_trap(connector, fake_vast):
    """Poll-trap states (exited/unknown/offline) should raise immediately."""
    fake_vast.set_instances([{"id": 5, "actual_status": "exited"}])
    with pytest.raises(RuntimeError, match="poll-trap"):
        connector.wait_for_running(5, timeout=100, poll=1)


def test_wait_for_running_times_out(connector, fake_vast):
    fake_vast.set_instances([{"id": 5, "actual_status": "loading"}])
    with (
        patch("research.cloud.vast_connector.time.sleep"),
        patch("research.cloud.vast_connector.time.time", side_effect=[0, 1, 2, 9999]),
        pytest.raises(TimeoutError),
    ):
        connector.wait_for_running(5, timeout=100, poll=1)


def test_wait_for_running_succeeds(connector, fake_vast):
    states = [
        {"id": 5, "actual_status": "loading"},
        {
            "id": 5,
            "actual_status": "running",
            "public_ipaddr": "1.2.3.4",
            "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "99"}]},
        },
    ]
    state_idx = [0]

    def fake_show_instance(id):
        idx = min(state_idx[0], len(states) - 1)
        state_idx[0] += 1
        return states[idx]

    fake_vast.show_instance = fake_show_instance
    fake_vast._ssh_url = "ssh://root@1.2.3.4:99"
    with (
        patch("research.cloud.vast_connector.time.sleep"),
        patch("research.cloud.vast_connector.socket.socket") as mock_sock,
    ):
        mock_sock.return_value.connect.return_value = None
        inst = connector.wait_for_running(5, timeout=100, poll=1)
    assert inst["actual_status"] == "running"


# ─── stop/start/destroy ────────────────────────────────────────────────────
def test_stop_instance(connector, fake_vast):
    fake_vast.set_instances([{"id": 777, "actual_status": "running"}])
    connector.stop_instance(777)
    stop_calls = [c for c in fake_vast.calls if c[0] == "stop_instance"]
    assert len(stop_calls) == 1
    assert stop_calls[0][1][0] == 777


def test_start_instance(connector, fake_vast):
    connector.start_instance(777)
    start_calls = [c for c in fake_vast.calls if c[0] == "start_instance"]
    assert len(start_calls) == 1


def test_destroy_instance(connector, fake_vast):
    fake_vast.set_instances([{"id": 777, "actual_status": "running"}])
    connector.destroy_instance(777)
    destroy_calls = [c for c in fake_vast.calls if c[0] == "destroy_instance"]
    assert len(destroy_calls) == 1


# ─── find_instance_by_label ────────────────────────────────────────────────
def test_find_instance_by_label(connector, fake_vast):
    fake_vast.set_instances(
        [
            {"id": 1, "label": "other", "actual_status": "running"},
            {"id": 2, "label": "forgeai-forgelm_v2-auto", "actual_status": "stopped"},
        ]
    )
    inst = connector.find_instance_by_label("forgeai-forgelm_v2-auto")
    assert inst is not None
    assert inst["id"] == 2


def test_find_instance_by_label_not_found(connector, fake_vast):
    fake_vast.set_instances([{"id": 1, "label": "other"}])
    inst = connector.find_instance_by_label("forgeai-forgelm_v2-auto")
    assert inst is None


# ─── ensure_instance (reuse lifecycle) ─────────────────────────────────────
def test_ensure_instance_reuses_running(connector, fake_vast):
    """If a running instance with our label exists, reuse it."""
    fake_vast.set_instances(
        [
            {"id": 42, "label": "forgeai-forgelm_v2-auto", "actual_status": "running"},
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=True,
    )
    inst_id, is_new = connector.ensure_instance(spec)
    assert inst_id == 42
    assert is_new is False


def test_ensure_instance_starts_stopped(connector, fake_vast):
    """If a stopped instance with our label exists, start it."""
    fake_vast.set_instances(
        [
            {"id": 42, "label": "forgeai-forgelm_v2-auto", "actual_status": "stopped"},
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=True,
    )
    inst_id, is_new = connector.ensure_instance(spec)
    assert inst_id == 42
    assert is_new is False
    start_calls = [c for c in fake_vast.calls if c[0] == "start_instance"]
    assert len(start_calls) == 1


def test_ensure_instance_destroys_poll_trap(connector, fake_vast):
    """If existing instance is in poll-trap state, destroy + create new."""
    fake_vast.set_instances(
        [
            {"id": 42, "label": "forgeai-forgelm_v2-auto", "actual_status": "exited"},
        ]
    )
    fake_vast.set_offers(
        [
            {
                "id": 100,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24576,
                "dph_total": 0.30,
                "reliability": 0.98,
                "dlperf": 50,
            },
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=True,
        budget=100.0,
        est_sec_per_step=1.0,
        min_vram_gb=20,
        min_reliability=0.9,
    )
    _inst_id, is_new = connector.ensure_instance(spec)
    assert is_new is True
    destroy_calls = [c for c in fake_vast.calls if c[0] == "destroy_instance"]
    assert len(destroy_calls) == 1


def test_ensure_instance_creates_new_when_not_found(connector, fake_vast):
    """No existing instance → create new."""
    fake_vast.set_instances([])
    fake_vast.set_offers(
        [
            {
                "id": 100,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24576,
                "dph_total": 0.30,
                "reliability": 0.98,
                "dlperf": 50,
            },
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=True,
        budget=100.0,
        est_sec_per_step=1.0,
        min_vram_gb=20,
        min_reliability=0.9,
    )
    inst_id, is_new = connector.ensure_instance(spec)
    assert inst_id == 12345  # from FakeVastAI.create_instance
    assert is_new is True


def test_ensure_instance_destroys_extras_single_cap(connector, fake_vast):
    """Multiple instances → destroy extras, keep first viable (single-instance cap)."""
    fake_vast.set_instances(
        [
            {"id": 10, "label": "forgeai-forgelm_v2-auto", "actual_status": "running"},
            {"id": 20, "label": "other", "actual_status": "running"},
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=True,
    )
    inst_id, is_new = connector.ensure_instance(spec)
    assert inst_id == 10  # kept the first viable
    assert is_new is False
    destroy_calls = [c for c in fake_vast.calls if c[0] == "destroy_instance"]
    assert len(destroy_calls) == 1
    assert destroy_calls[0][1][0] == 20


def test_ensure_instance_reuse_false_destroys_existing(connector, fake_vast):
    """reuse_instance=False → destroy any existing instance before creating new."""
    fake_vast.set_instances(
        [
            {"id": 10, "label": "forgeai-forgelm_v2-auto", "actual_status": "running"},
        ]
    )
    fake_vast.set_offers(
        [
            {
                "id": 100,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24576,
                "dph_total": 0.30,
                "reliability": 0.98,
                "dlperf": 50,
            },
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=False,
        budget=100.0,
        est_sec_per_step=1.0,
        min_vram_gb=20,
        min_reliability=0.9,
    )
    _inst_id, is_new = connector.ensure_instance(spec)
    assert is_new is True
    destroy_calls = [c for c in fake_vast.calls if c[0] == "destroy_instance"]
    assert len(destroy_calls) == 1
    assert destroy_calls[0][1][0] == 10


def test_ensure_instance_reuses_any_live_no_label_match(connector, fake_vast):
    """If no label match but a live instance exists, reuse it (single-instance cap)."""
    fake_vast.set_instances(
        [
            {"id": 10, "label": "some-other-label", "actual_status": "running"},
        ]
    )
    spec = RemoteTrainingSpec(
        train_args={"--config": "forgelm_v2"},
        gpu_filter="",
        reuse_instance=True,
    )
    inst_id, is_new = connector.ensure_instance(spec)
    assert inst_id == 10
    assert is_new is False


# ─── select_best_offer (budget + perf/$) ───────────────────────────────────
def test_select_best_offer_within_budget(connector, fake_vast):
    fake_vast.set_offers(
        [
            {
                "id": 1,
                "gpu_name": "A100",
                "num_gpus": 1,
                "gpu_total_ram": 81920,
                "dph_total": 1.20,
                "reliability": 0.99,
                "dlperf": 80,
            },
            {
                "id": 2,
                "gpu_name": "H100",
                "num_gpus": 1,
                "gpu_total_ram": 81920,
                "dph_total": 2.50,
                "reliability": 0.99,
                "dlperf": 100,
            },
        ]
    )
    spec = RemoteTrainingSpec(
        budget=10.0,
        est_sec_per_step=5.0,
        train_args={"--max-steps": 500},
        min_vram_gb=24,
        min_reliability=0.9,
    )
    offer = connector.select_best_offer(spec)
    assert offer is not None
    # A100 has better perf/$ (80/1.2=66 vs 100/2.5=40)
    assert offer.id == 1


def test_select_best_offer_no_offers(connector, fake_vast):
    fake_vast.set_offers([])
    spec = RemoteTrainingSpec(
        budget=10.0,
        est_sec_per_step=5.0,
        train_args={"--max-steps": 500},
        min_vram_gb=24,
        min_reliability=0.9,
    )
    offer = connector.select_best_offer(spec)
    assert offer is None


def test_estimate_training_hours(connector):
    spec = RemoteTrainingSpec(
        est_sec_per_step=5.0,
        train_args={"--max-steps": 500},
    )
    hours = connector.estimate_training_hours(spec)
    assert hours == pytest.approx(500 * 5.0 / 3600.0)


# ─── path remapping ────────────────────────────────────────────────────────
def test_remap_paths(connector):
    spec = RemoteTrainingSpec(
        checkpoints=["/local/ForgeLM_V2_LFM25-1.2B.safetensors"],
        data_files=["/local/tool_use_fc_70.jsonl"],
        train_args={"--save": "ForgeLM_V2_LFM25-1.2B.sft.safetensors"},
    )
    assert (
        connector._remap_path("/local/ForgeLM_V2_LFM25-1.2B.safetensors", spec)
        == f"{REMOTE_CKPT_DIR}/ForgeLM_V2_LFM25-1.2B.safetensors"
    )
    assert connector._remap_path("/local/tool_use_fc_70.jsonl", spec) == f"{REMOTE_DATA_DIR}/tool_use_fc_70.jsonl"
    assert (
        connector._remap_path("ForgeLM_V2_LFM25-1.2B.sft.safetensors", spec)
        == f"{REMOTE_OUT_DIR}/ForgeLM_V2_LFM25-1.2B.sft.safetensors"
    )


# ─── remote train command construction ─────────────────────────────────────
def test_build_remote_train_cmd_forwards_args(connector):
    spec = RemoteTrainingSpec(
        checkpoints=["/local/base.safetensors"],
        data_files=["/local/data.jsonl"],
        train_args={
            "--data": ["/local/data.jsonl"],
            "--config": "forgelm_v2",
            "--checkpoint": "/local/base.safetensors",
            "--save": "out.safetensors",
            "--max-steps": 500,
            "--lora": False,
            "--no-bitnet-everywhere": True,
        },
    )
    cmd = connector._build_remote_train_cmd(spec)
    assert "forge.training.runners.sft_train" in cmd
    assert "--config forgelm_v2" in cmd
    assert "--max-steps 500" in cmd
    assert "--lora" not in cmd
    assert "--no-bitnet-everywhere" in cmd
    assert f"{REMOTE_CKPT_DIR}/base.safetensors" in cmd
    assert f"{REMOTE_DATA_DIR}/data.jsonl" in cmd
    assert f"{REMOTE_OUT_DIR}/out.safetensors" in cmd
    assert "__FORGE_EXIT__" in cmd


# ─── file manifest ─────────────────────────────────────────────────────────
def test_compute_file_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h = _compute_file_hash(f)
    assert len(h) == 16
    assert h == hashlib_sha256_16("hello world")


def test_compute_local_manifest(tmp_path):
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("print('b')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.pyc").write_text("cache")
    manifest = _compute_local_manifest(str(tmp_path), exclude=["__pycache__"])
    assert "a.py" in manifest
    assert "sub/b.py" in manifest
    assert "__pycache__" not in str(manifest)
    assert "c.pyc" not in str(manifest)


def test_file_entry_roundtrip():
    entry = FileEntry(size=100, mtime=1234567890.0, sha256="abcdef0123456789")
    d = entry.to_dict()
    assert d["size"] == 100
    assert d["mtime"] == 1234567890.0
    assert d["sha256"] == "abcdef0123456789"
    entry2 = FileEntry.from_dict(d)
    assert entry2.size == 100
    assert entry2.mtime == 1234567890.0
    assert entry2.sha256 == "abcdef0123456789"


# ─── log parsing ───────────────────────────────────────────────────────────
def test_parse_log_line_metric():
    parsed = VastConnector._parse_log_line("step 42 loss 3.14 lr 1e-4")
    assert parsed["type"] == "metric"
    assert parsed["step"] == 42
    assert parsed["loss"] == pytest.approx(3.14)
    assert parsed["lr"] == pytest.approx(1e-4)


def test_parse_log_line_exit():
    parsed = VastConnector._parse_log_line("__FORGE_EXIT__:0")
    assert parsed["type"] == "exit"
    assert parsed["exit_code"] == 0


def test_parse_log_line_error():
    parsed = VastConnector._parse_log_line("ERROR: something went wrong")
    assert parsed["level"] == "ERROR"


def test_parse_log_line_warning():
    parsed = VastConnector._parse_log_line("WARNING: deprecated function")
    assert parsed["level"] == "WARNING"


def test_should_show_line_with_filter():
    parsed = {"raw": "step 42 loss 3.14"}
    assert VastConnector._should_show_line(parsed, "loss") is True
    assert VastConnector._should_show_line(parsed, "error") is False
    assert VastConnector._should_show_line(parsed, None) is True


# ─── get_logs (SDK) ────────────────────────────────────────────────────────
def test_get_logs_via_sdk(connector, fake_vast):
    logs = connector.get_logs(123, tail=100, filter_str="error")
    assert "step 0 loss 5.0" in logs
    logs_calls = [c for c in fake_vast.calls if c[0] == "logs"]
    assert len(logs_calls) == 1
    _, args, kwargs = logs_calls[0]
    assert args[0] == 123
    assert kwargs["tail"] == "100"
    assert kwargs["filter"] == "error"


# ─── build_spec_from_args ──────────────────────────────────────────────────
def test_build_spec_from_args_resolves_uploads(tmp_path):
    from argparse import Namespace

    ckpt = tmp_path / "base.safetensors"
    ckpt.write_bytes(b"x")
    data = tmp_path / "data.jsonl"
    data.write_text("{}")
    args = Namespace(
        data=[str(data)],
        config="forgelm_v2",
        checkpoint=str(ckpt),
        save="out.safetensors",
        max_steps=100,
        lr=5e-5,
        batch_size=2,
        seq_len=1024,
        lora=True,
        bitnet_everywhere=True,
        gpu_filter="gpu_name=H100",
        max_price=1.0,
        min_vram_gb=40,
        min_reliability=0.95,
        vast_disk_gb=200,
        vast_image=None,
        vast_on_demand=True,
        vast_ssh_key="",
        vast_auto_destroy=False,
        vast_reuse_instance=True,
        vast_use_volume=False,
        vast_volume_size_gb=200,
        vast_stream_logs=True,
        vast_download_output=True,
        vast_poll_interval=5.0,
        vast_startup_timeout=300.0,
        vast_maximize_throughput=True,
        vast_budget=10.0,
        vast_est_sec_per_step=1.5,
        vast_log_filter=None,
        from_scratch=False,
        remote_vast=True,
    )
    train_dests = {
        "data",
        "config",
        "checkpoint",
        "save",
        "max_steps",
        "lr",
        "batch_size",
        "seq_len",
        "lora",
        "bitnet_everywhere",
    }
    spec = build_spec_from_args(args, train_dests)
    assert spec.gpu_filter == "gpu_name=H100"
    assert spec.max_price == 1.0
    assert spec.min_vram_gb == 40
    assert spec.disk_gb == 200
    assert spec.auto_destroy is False  # v2 default
    assert spec.reuse_instance is True
    assert str(ckpt) in spec.checkpoints
    assert str(data) in spec.data_files
    assert spec.train_args["--config"] == "forgelm_v2"
    assert spec.train_args["--max-steps"] == 100
    assert spec.train_args["--lora"] is True


# ─── Full lifecycle (orchestration only — I/O methods mocked) ──────────────
def test_run_remote_training_lifecycle_stop(connector, fake_vast, tmp_path):
    """Verify the ensure→sync→train→download→stop orchestration.

    The heavy I/O methods are mocked so the test does NOT walk the real
    repo or open real SSH connections. Verifies that the instance is
    STOPPED (not destroyed) after training when auto_destroy=False.
    """
    ckpt = tmp_path / "base.safetensors"
    ckpt.write_bytes(b"BASECKPT")
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt":"hi","response":"yo"}')

    fake_vast.set_offers(
        [
            {
                "id": 42,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24576,
                "dph_total": 0.30,
                "reliability": 0.98,
                "dlperf": 50,
            },
        ]
    )
    fake_vast.set_instances([])  # no existing → create new

    spec = RemoteTrainingSpec(
        checkpoints=[str(ckpt)],
        data_files=[str(data)],
        train_args={
            "--data": [str(data)],
            "--config": "forgelm_tiny",
            "--checkpoint": str(ckpt),
            "--save": "out.safetensors",
            "--max-steps": 2,
        },
        auto_destroy=False,  # v2: stop, not destroy
        reuse_instance=False,  # skip label lookup for this test
        stream_logs=True,
        download_output=True,
        startup_timeout=60.0,
        poll_interval=0.01,
        budget=100.0,
        est_sec_per_step=1.0,
        min_vram_gb=20,
        min_reliability=0.9,
    )

    def fake_wait(self, inst_id, **k):
        return {"id": inst_id, "actual_status": "running"}

    def fake_ssh_info(self, inst_id):
        return "1.2.3.4", 99

    def fake_sync_dir(self, host, port, local, remote, exclude=None):
        return 0, 10  # 0 uploaded, 10 skipped

    def fake_upload_file(self, host, port, local, remote, show_progress=False):
        pass

    def fake_exec(self, host, port, cmd, timeout=None):
        return 0, "ok", ""

    def fake_stream(self, host, port, cmd, reconnect=True):
        yield "step 0 loss 5.0"
        yield "step 1 loss 4.5"
        yield "__FORGE_EXIT__:0"

    def fake_download(self, host, port, remote, local, show_progress=True):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(b"TRAINEDCKPT")

    def fake_remote_exists(self, host, port, remote_path, expected_size):
        return False  # force upload

    # Fake SSH client for _ssh_connect (used by sftp mkdir block in _drive_instance)
    fake_sftp = MagicMock()
    fake_ssh_client = MagicMock()
    fake_ssh_client.open_sftp.return_value = fake_sftp

    with (
        patch.object(VastConnector, "wait_for_running", autospec=True, side_effect=fake_wait),
        patch.object(VastConnector, "get_ssh_info", autospec=True, side_effect=fake_ssh_info),
        patch.object(VastConnector, "sync_dir", autospec=True, side_effect=fake_sync_dir),
        patch.object(VastConnector, "upload_file", autospec=True, side_effect=fake_upload_file),
        patch.object(VastConnector, "exec_remote", autospec=True, side_effect=fake_exec),
        patch.object(VastConnector, "stream_remote", autospec=True, side_effect=fake_stream),
        patch.object(VastConnector, "download_file", autospec=True, side_effect=fake_download),
        patch.object(VastConnector, "_remote_file_exists", autospec=True, side_effect=fake_remote_exists),
        patch.object(VastConnector, "_check_provision_hash", autospec=True, return_value=False),
        patch.object(VastConnector, "_write_provision_hash", autospec=True),
        patch.object(VastConnector, "_ssh_connect", autospec=True, return_value=fake_ssh_client),
    ):
        rc = connector.run_remote_training(spec)

    assert rc == 0
    # Instance was STOPPED (not destroyed) since auto_destroy=False
    stop_calls = [c for c in fake_vast.calls if c[0] == "stop_instance"]
    destroy_calls = [c for c in fake_vast.calls if c[0] == "destroy_instance"]
    assert len(stop_calls) == 1
    assert len(destroy_calls) == 0


def test_run_remote_training_lifecycle_destroy(connector, fake_vast, tmp_path):
    """When auto_destroy=True, instance is destroyed after training."""
    ckpt = tmp_path / "base.safetensors"
    ckpt.write_bytes(b"BASECKPT")
    data = tmp_path / "data.jsonl"
    data.write_text("{}")

    fake_vast.set_offers(
        [
            {
                "id": 42,
                "gpu_name": "RTX_4090",
                "num_gpus": 1,
                "gpu_total_ram": 24576,
                "dph_total": 0.30,
                "reliability": 0.98,
                "dlperf": 50,
            },
        ]
    )
    fake_vast.set_instances([])

    spec = RemoteTrainingSpec(
        checkpoints=[str(ckpt)],
        data_files=[str(data)],
        train_args={
            "--data": [str(data)],
            "--config": "forgelm_tiny",
            "--checkpoint": str(ckpt),
            "--save": "out.safetensors",
            "--max-steps": 2,
        },
        auto_destroy=True,
        reuse_instance=False,
        stream_logs=True,
        download_output=True,
        startup_timeout=60.0,
        poll_interval=0.01,
        budget=100.0,
        est_sec_per_step=1.0,
        min_vram_gb=20,
        min_reliability=0.9,
    )

    fake_sftp = MagicMock()
    fake_ssh_client = MagicMock()
    fake_ssh_client.open_sftp.return_value = fake_sftp

    with (
        patch.object(
            VastConnector, "wait_for_running", autospec=True, return_value={"id": 12345, "actual_status": "running"}
        ),
        patch.object(VastConnector, "get_ssh_info", autospec=True, return_value=("1.2.3.4", 99)),
        patch.object(VastConnector, "sync_dir", autospec=True, return_value=(0, 10)),
        patch.object(VastConnector, "upload_file", autospec=True),
        patch.object(VastConnector, "exec_remote", autospec=True, return_value=(0, "ok", "")),
        patch.object(
            VastConnector, "stream_remote", autospec=True, side_effect=lambda *a, **k: iter(["__FORGE_EXIT__:0"])
        ),
        patch.object(VastConnector, "download_file", autospec=True),
        patch.object(VastConnector, "_remote_file_exists", autospec=True, return_value=False),
        patch.object(VastConnector, "_check_provision_hash", autospec=True, return_value=False),
        patch.object(VastConnector, "_write_provision_hash", autospec=True),
        patch.object(VastConnector, "_ssh_connect", autospec=True, return_value=fake_ssh_client),
    ):
        rc = connector.run_remote_training(spec)

    assert rc == 0
    destroy_calls = [c for c in fake_vast.calls if c[0] == "destroy_instance"]
    stop_calls = [c for c in fake_vast.calls if c[0] == "stop_instance"]
    assert len(destroy_calls) == 1
    assert len(stop_calls) == 0


# ─── Helper ────────────────────────────────────────────────────────────────
def hashlib_sha256_16(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]
