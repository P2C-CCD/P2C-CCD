from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import p2cccd.bench as bench  # noqa: E402


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    method_name: str
    module_name: str
    public_symbols: tuple[str, ...]
    doc_name: str
    test_name: str


ARTIFACTS = (
    BaselineArtifact(
        method_name="PureExactCPU",
        module_name="pure_exact_cpu",
        public_symbols=("PureExactCPUConfig", "run_pure_exact_cpu_on_generated_dataset"),
        doc_name="pure_exact_cpu_baseline.md",
        test_name="test_pure_exact_cpu_baseline.py",
    ),
    BaselineArtifact(
        method_name="BVHExact",
        module_name="bvh_exact",
        public_symbols=("BVHExactConfig", "run_bvh_exact_on_generated_dataset"),
        doc_name="bvh_exact_baseline.md",
        test_name="test_bvh_exact_baseline.py",
    ),
    BaselineArtifact(
        method_name="RTExact",
        module_name="rt_exact",
        public_symbols=("RTExactConfig", "run_rt_exact_on_generated_dataset"),
        doc_name="rt_exact_baseline.md",
        test_name="test_rt_exact_baseline.py",
    ),
    BaselineArtifact(
        method_name="RTSTPFExact",
        module_name="rt_stpf_exact",
        public_symbols=("RTSTPFExactConfig", "run_rt_stpf_exact_on_generated_dataset"),
        doc_name="rt_stpf_exact_baseline.md",
        test_name="test_rt_stpf_exact_baseline.py",
    ),
    BaselineArtifact(
        method_name="NoProposal",
        module_name="no_proposal",
        public_symbols=("NoProposalConfig", "run_no_proposal_on_generated_dataset"),
        doc_name="no_proposal_ablation.md",
        test_name="test_no_proposal_ablation.py",
    ),
    BaselineArtifact(
        method_name="SortBroadPhaseExact",
        module_name="sort_broad_phase_exact",
        public_symbols=("SortBroadPhaseConfig", "run_sort_broad_phase_exact_on_generated_dataset"),
        doc_name="sort_broad_phase_exact_baseline.md",
        test_name="test_sort_broad_phase_exact.py",
    ),
    BaselineArtifact(
        method_name="IntervalOnly",
        module_name="stpf_head_ablations",
        public_symbols=("IntervalOnlyConfig", "run_interval_only_on_generated_dataset"),
        doc_name="stpf_head_ablations.md",
        test_name="test_stpf_head_ablations.py",
    ),
    BaselineArtifact(
        method_name="RankingOnly",
        module_name="stpf_head_ablations",
        public_symbols=("RankingOnlyConfig", "run_ranking_only_on_generated_dataset"),
        doc_name="stpf_head_ablations.md",
        test_name="test_stpf_head_ablations.py",
    ),
    BaselineArtifact(
        method_name="NoQueueDecouple",
        module_name="no_queue_decouple",
        public_symbols=("NoQueueDecoupleConfig", "run_no_queue_decouple_microbenchmark"),
        doc_name="no_queue_decouple_microbenchmark.md",
        test_name="test_no_queue_decouple_microbenchmark.py",
    ),
    BaselineArtifact(
        method_name="PatchGranularityAblation",
        module_name="patch_granularity_ablation",
        public_symbols=("PatchGranularityAblationConfig", "run_patch_granularity_ablation_on_generated_dataset"),
        doc_name="patch_granularity_ablation.md",
        test_name="test_patch_granularity_ablation.py",
    ),
    BaselineArtifact(
        method_name="SlabProxyAblation",
        module_name="slab_proxy_ablation",
        public_symbols=("SlabProxyAblationConfig", "run_slab_proxy_ablation_on_generated_dataset"),
        doc_name="slab_proxy_ablation.md",
        test_name="test_slab_proxy_ablation.py",
    ),
    BaselineArtifact(
        method_name="RTDCDStyle",
        module_name="rt_style_reproduction",
        public_symbols=("RTDCDStyleConfig", "run_rt_dcd_style_on_generated_dataset"),
        doc_name="rt_style_reproduction.md",
        test_name="test_rt_style_reproduction.py",
    ),
    BaselineArtifact(
        method_name="RTCCDStyle",
        module_name="rt_style_reproduction",
        public_symbols=("RTCCDStyleConfig", "run_rt_ccd_style_on_generated_dataset"),
        doc_name="rt_style_reproduction.md",
        test_name="test_rt_style_reproduction.py",
    ),
    BaselineArtifact(
        method_name="NeuralSVCDStyle",
        module_name="learned_style_comparison",
        public_symbols=("NeuralSVCDStyleConfig", "run_neural_svcd_style_on_generated_dataset"),
        doc_name="learned_style_comparison.md",
        test_name="test_learned_style_comparison.py",
    ),
    BaselineArtifact(
        method_name="CabiNetStyle",
        module_name="learned_style_comparison",
        public_symbols=("CabiNetStyleConfig", "run_cabinet_style_on_generated_dataset"),
        doc_name="learned_style_comparison.md",
        test_name="test_learned_style_comparison.py",
    ),
    BaselineArtifact(
        method_name="CuRoboDownstream",
        module_name="curobo_downstream",
        public_symbols=("CuRoboDownstreamConfig", "run_curobo_downstream_on_generated_dataset"),
        doc_name="curobo_downstream.md",
        test_name="test_curobo_downstream.py",
    ),
)


def test_benchmarks_and_baselines_have_modules_docs_tests_and_public_exports() -> None:
    for artifact in ARTIFACTS:
        module = importlib.import_module(f"p2cccd.bench.{artifact.module_name}")
        assert module is not None, artifact.method_name
        assert (PROJECT_ROOT / "docs" / artifact.doc_name).exists(), artifact.method_name
        assert (PROJECT_ROOT / "tests" / "python" / artifact.test_name).exists(), artifact.method_name
        for symbol in artifact.public_symbols:
            assert hasattr(bench, symbol), f"{artifact.method_name} missing p2cccd.bench.{symbol}"


def test_benchmarks_and_baselines_chapter_audit_doc_exists() -> None:
    audit_doc = PROJECT_ROOT / "docs" / "benchmarks_and_baselines.md"
    content = audit_doc.read_text(encoding="utf-8")

    for artifact in ARTIFACTS:
        assert artifact.method_name in content
        assert artifact.module_name in content
    assert "not an official" in content
    assert "final_fn_zero" in content
