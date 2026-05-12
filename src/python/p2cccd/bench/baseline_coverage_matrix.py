from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence


RUN_NAME = "baseline_matrix_run_id"


@dataclass(frozen=True, slots=True)
class BaselineEntry:
    name: str
    source_path: str
    local_state: str
    build_state: str
    primitive_correctness: str
    primitive_wall_time: str
    dense_group_wall_time: str
    full_scene: str
    role: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    paper_usage: str


@dataclass(frozen=True, slots=True)
class ComparisonLevel:
    level: str
    purpose: str
    fair_input: str
    baselines: tuple[str, ...]
    metrics: tuple[str, ...]
    paper_table: str


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def _artifact(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    return {
        "path": rel.replace("\\", "/"),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def _status_from_artifacts(root: Path) -> dict[str, Any]:
    evidence_paths = [
        "src/baseline/Tight-Inclusion/build-release/libtight_inclusion.a",
        "src/baseline/Tight-Inclusion/build-release/app/Tight_Inclusion_bin.exe",
        "src/baseline/CCD-Wrapper/build-release/libccd_wrapper.a",
        "src/baseline/CCD-Wrapper/build-release/ccd_benchmark.exe",
        "src/baseline/Scalable-CCD/build-p-cuda-full/Testing/Temporary/LastTest.log",
        "src/baseline/Exact-Root-Parity-CCD/build-p-mingw-rational-cmake326/Testing/Temporary/LastTest.log",
        "src/baseline/rigid-ipc/build-repro-noprofile/Testing/Temporary/LastTest.log",
        "src/benchmark/tight_inclusion_sota_comparison_v3_current_run_id.md",
        "src/benchmark/native_dense_group_walltime_run_id.md",
        "src/benchmark/learned_vs_random_ablation_run_id.md",
        "src/benchmark/scalable-ccd-full-data-tests.log",
        "src/benchmark/exact-root-parity-standalone.log",
    ]
    return {"artifacts": [_artifact(root, p) for p in evidence_paths]}


def _baseline_entries(root: Path) -> tuple[BaselineEntry, ...]:
    tight_built = _exists(root, "src/baseline/Tight-Inclusion/build-release/libtight_inclusion.a")
    ccd_wrapper_built = _exists(root, "src/baseline/CCD-Wrapper/build-release/libccd_wrapper.a")
    scalable_tested = _exists(
        root, "src/baseline/Scalable-CCD/build-p-cuda-full/Testing/Temporary/LastTest.log"
    )
    erp_tested = _exists(
        root,
        "src/baseline/Exact-Root-Parity-CCD/build-p-mingw-rational-cmake326/Testing/Temporary/LastTest.log",
    )
    rigid_built = _exists(root, "src/baseline/rigid-ipc/build-repro-noprofile")
    native_dense_report = _exists(root, "src/benchmark/native_dense_group_walltime_run_id.md")
    learned_random_report = _exists(root, "src/benchmark/learned_vs_random_ablation_run_id.md")

    return (
        BaselineEntry(
            name="Tight-Inclusion",
            source_path="src/baseline/Tight-Inclusion",
            local_state="present",
            build_state="release lib/bin present" if tight_built else "source only",
            primitive_correctness="yes",
            primitive_wall_time="yes",
            dense_group_wall_time="partial",
            full_scene="partial",
            role="Main conservative primitive CCD SOTA baseline.",
            evidence=(
                "src/benchmark/tight_inclusion_sota_comparison_v3_current_run_id.md",
                "src/baseline/Tight-Inclusion/build-release/libtight_inclusion.a",
                "src/baseline/Tight-Inclusion/build-release/app/Tight_Inclusion_bin.exe",
            ),
            limitations=(
                "The primitive heldout wall-time table is fair and uses the same C++ TI exact kernel.",
                "Dense group early-stop still needs a real TI primitive payload to replace the current native proxy exact payload for the final SOTA dense wall-time table.",
            ),
            paper_usage="Main L1 primitive correctness and wall-time baseline; L2 dense group TI payload remains a P0 follow-up.",
        ),
        BaselineEntry(
            name="CCD-Wrapper",
            source_path="src/baseline/CCD-Wrapper",
            local_state="present",
            build_state="release lib/bin present; CTest path requires repair"
            if ccd_wrapper_built
            else "source only",
            primitive_correctness="planned",
            primitive_wall_time="planned",
            dense_group_wall_time="no",
            full_scene="no",
            role="Multi-algorithm narrow-phase wrapper for breadth of comparison.",
            evidence=(
                "src/baseline/CCD-Wrapper/build-release/libccd_wrapper.a",
                "src/baseline/CCD-Wrapper/build-release/ccd_benchmark.exe",
            ),
            limitations=(
                "Current CTest metadata still contains a stale absolute CCD-Wrapper path, so normalized smoke tests are not yet accepted as evidence.",
                "Different wrapped algorithms may not share identical conservative semantics or TOI tolerance.",
            ),
            paper_usage="Supplementary L0/L1 small-query comparison after path repair and tolerance normalization.",
        ),
        BaselineEntry(
            name="Scalable-CCD",
            source_path="src/baseline/Scalable-CCD",
            local_state="present",
            build_state="CUDA full build tested" if scalable_tested else "source/build present",
            primitive_correctness="partial",
            primitive_wall_time="partial",
            dense_group_wall_time="partial",
            full_scene="supplementary",
            role="Broad+narrow scalable CCD pipeline and parallel-friendly reference.",
            evidence=(
                "src/benchmark/scalable-ccd-full-data-tests.log",
                "src/baseline/Scalable-CCD/build-p-cuda-full/Testing/Temporary/LastTest.log",
                "src/baseline/Sample-Scalable-CCD-Data",
            ),
            limitations=(
                "Its scene/data representation is not identical to the P2CCCD dense group format.",
                "Use as scene-level and pipeline-level supplementary evidence unless converted candidate groups are generated.",
            ),
            paper_usage="L3 supplementary scalability/full-scene comparison; selected converted cases can be L2 if candidate groups are exported.",
        ),
        BaselineEntry(
            name="Sample-Scalable-CCD-Data",
            source_path="src/baseline/Sample-Scalable-CCD-Data",
            local_state="present",
            build_state="dataset only",
            primitive_correctness="not applicable",
            primitive_wall_time="not applicable",
            dense_group_wall_time="dataset source",
            full_scene="dataset source",
            role="Official/sample analytic scenes for Scalable-CCD style experiments.",
            evidence=("src/baseline/Sample-Scalable-CCD-Data",),
            limitations=("This is not an algorithm baseline; it only supplies scenes/data.",),
            paper_usage="Dataset source for supplementary scene-level conversion experiments.",
        ),
        BaselineEntry(
            name="Exact-Root-Parity-CCD",
            source_path="src/baseline/Exact-Root-Parity-CCD",
            local_state="present",
            build_state="small CTest passed" if erp_tested else "source/build present",
            primitive_correctness="yes-small",
            primitive_wall_time="small-only",
            dense_group_wall_time="no",
            full_scene="no",
            role="Exact narrow-phase reference for degeneracy, grazing, and root parity cases.",
            evidence=(
                "src/benchmark/exact-root-parity-standalone.log",
                "src/benchmark/exact-root-parity-handcrafted-fprp.log",
                "src/baseline/Exact-Root-Parity-CCD/build-p-mingw-rational-cmake326/Testing/Temporary/LastTest.log",
            ),
            limitations=(
                "Best used for correctness cross-checks rather than large wall-time tables.",
                "Standalone app exit status is not a clean benchmark result; CTest smoke passed.",
            ),
            paper_usage="L0 exact correctness smoke and hard-case appendix reference.",
        ),
        BaselineEntry(
            name="rigid-ipc",
            source_path="src/baseline/rigid-ipc",
            local_state="present",
            build_state="build tree present" if rigid_built else "source only",
            primitive_correctness="no",
            primitive_wall_time="no",
            dense_group_wall_time="converted-only",
            full_scene="supplementary",
            role="Real rigid-body trajectories and complex contact scenes.",
            evidence=(
                "src/baseline/rigid-ipc/meshes",
                "src/baseline/rigid-ipc/fixtures",
                "src/baseline/rigid-ipc/build-repro-noprofile/Testing/Temporary/LastTest.log",
            ),
            limitations=(
                "It is a simulation/IPC pipeline, not a pure CCD kernel.",
                "Do not compare IPC step time directly against primitive CCD kernel time.",
            ),
            paper_usage="Supplementary real-trajectory source; convert selected scenes to P2CCCD candidate groups for qualitative/visual evidence.",
        ),
        BaselineEntry(
            name="P2CCCD RTSTPFExact",
            source_path="src",
            local_state="present",
            build_state="native dense group report present" if native_dense_report else "implementation present",
            primitive_correctness="yes",
            primitive_wall_time="yes",
            dense_group_wall_time="yes",
            full_scene="partial",
            role="Proposed learned scheduling/proposal method with exact certificate fallback.",
            evidence=(
                "src/benchmark/native_dense_group_walltime_run_id.md",
                "src/benchmark/learned_vs_random_ablation_run_id.md",
                "src/benchmark/complete_benchmark_vs_baselines_run_id.md",
            ),
            limitations=(
                "The strongest current evidence is dense/high-cost candidate groups with zero-FN conservative scheduling.",
                "Sparse primitive global-threshold benchmark is not the main speed claim.",
                "Learned-vs-random ablation shows learned signal exists, but default cost-aware ranking is not uniformly better than random.",
            )
            if learned_random_report
            else ("Learned-vs-random ablation report missing.",),
            paper_usage="Main proposed method rows in L1/L2; dense group table is the current primary advantage evidence.",
        ),
    )


def _comparison_levels() -> tuple[ComparisonLevel, ...]:
    return (
        ComparisonLevel(
            level="L0 correctness smoke",
            purpose="Check conservative collision agreement on small primitive VF/EE cases.",
            fair_input="unit-tests, erleben, selected golf-ball primitive query files",
            baselines=("Tight-Inclusion", "Exact-Root-Parity-CCD", "CCD-Wrapper after CTest path repair"),
            metrics=("TP", "TN", "FP", "FN", "recall", "TOI/agreement when available"),
            paper_table="appendix correctness smoke",
        ),
        ComparisonLevel(
            level="L1 primitive CCD wall time",
            purpose="Fair SOTA primitive kernel comparison.",
            fair_input="same NYU/Tight-Inclusion primitive CSV queries, same tolerance and max iteration",
            baselines=("Tight-Inclusion", "NoProposal+TI", "RTExact+TI", "RTSTPFExact+TI"),
            metrics=("wall_ms", "exact_ms", "proposal_ms", "QPS", "p50/p90/p99", "FN"),
            paper_table="primitive SOTA wall-time and correctness table",
        ),
        ComparisonLevel(
            level="L2 dense candidate group",
            purpose="Measure whether learned scheduling reduces exact certificate work and native wall time under dense/high-cost candidate groups.",
            fair_input="same conservative candidate groups and same final exact certificate policy",
            baselines=("NoProposal", "RandomUniform", "Heuristic", "Learned RTSTPFExact", "Oracle upper bound"),
            metrics=("exact_calls", "exact_work", "early_stop_wall_ms", "FN", "first_positive_rank"),
            paper_table="main dense-group advantage table",
        ),
        ComparisonLevel(
            level="L3 scene-level supplementary",
            purpose="Demonstrate external validity on broader scene/trajectory pipelines.",
            fair_input="converted Scalable-CCD scenes or rigid-ipc trajectories when possible; otherwise documented native scene inputs",
            baselines=("Scalable-CCD", "rigid-ipc", "P2CCCD converted scene groups"),
            metrics=("candidate_count", "total_time", "qualitative contact agreement", "visualization"),
            paper_table="supplementary scene-level table",
        ),
    )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _render_markdown(payload: dict[str, Any]) -> str:
    entries = payload["baselines"]
    levels = payload["comparison_levels"]
    artifacts = payload["status"]["artifacts"]

    coverage_rows = [
        [
            f"`{entry['name']}`",
            entry["build_state"],
            entry["primitive_correctness"],
            entry["primitive_wall_time"],
            entry["dense_group_wall_time"],
            entry["full_scene"],
            entry["paper_usage"],
        ]
        for entry in entries
    ]
    level_rows = [
        [
            level["level"],
            level["fair_input"],
            ", ".join(level["baselines"]),
            ", ".join(level["metrics"]),
            level["paper_table"],
        ]
        for level in levels
    ]
    artifact_rows = [
        [
            f"`{artifact['path']}`",
            "yes" if artifact["exists"] else "no",
            "" if artifact["bytes"] is None else artifact["bytes"],
        ]
        for artifact in artifacts
    ]
    detail_rows = [
        [
            f"`{entry['name']}`",
            entry["role"],
            "<br>".join(entry["evidence"]),
            "<br>".join(entry["limitations"]),
        ]
        for entry in entries
    ]

    return "\n".join(
        [
            "# Baseline description(run_id)",
            "",
            "## 1. Conclusion",
            "",
            "current baseline descriptionsplitlayerdescription, rather thanalldescriptionsamedescription wall-time description. ",
            "",
            "- `Tight-Inclusion` isdescription primitive-level SOTA wall-time description baseline. ",
            "- `P2CCCD RTSTPFExact` descriptionadvantagedescriptionin dense/high-cost candidate group: descriptionuse native scheduling + exact early-stop, report exact-call/work reduction, wall time and `FN=0`. ",
            "- `Scalable-CCD` and `rigid-ipc` description scene-level / pipeline-level supplementary; descriptionconvertassame candidate group, descriptionand primitive CCD kernel time direct comparison. ",
            "- `CCD-Wrapper` and `Exact-Root-Parity-CCD` description correctness breadth; among them CCD-Wrapper current release build descriptionin, but CTest Pathdescriptionafterdescriptionas normalized smoke description. ",
            "",
            "## 2. Baseline Coverage Matrix",
            "",
            _md_table(
                [
                    "Baseline",
                    "Build / local state",
                    "Primitive correctness",
                    "Primitive wall time",
                    "Dense group wall time",
                    "Full scene",
                    "Paper usage",
                ],
                coverage_rows,
            ),
            "",
            "## 3. descriptionlayerlevel",
            "",
            _md_table(
                ["Level", "Fair input", "Baselines", "Metrics", "Paper table"],
                level_rows,
            ),
            "",
            "## 4. descriptionanddescription",
            "",
            _md_table(["Baseline", "Role", "Evidence", "Limitations"], detail_rows),
            "",
            "## 5. descriptionhasdescription",
            "",
            _md_table(["Artifact", "Exists", "Bytes"], artifact_rows),
            "",
            "## 6. descriptionwritedescription",
            "",
            "descriptionusedescription: ",
            "",
            "1. `Primitive SOTA Table`: description Tight-Inclusion / NoProposal+TI / RTExact+TI / RTSTPFExact+TI, descriptionsame C++ TI exact kernel, samedescription, `FN=0`. currentConclusionis RTSTPFExact+TI inordinary sparse primitive heldout ondescriptionhas wall-time advantage, this isdescriptionrather thandescription. ",
            "2. `Dense Candidate Group Table`: description NoProposal / Random / Heuristic / Learned / Oracle, description dense/high-cost group under exact-work reduction, native early-stop wall time, `FN=0`. this iscurrentthis paperdescriptionadvantagedescription. ",
            "3. `Supplementary External Coverage Table`: description CCD-Wrapper, Scalable-CCD, Exact-Root-Parity-CCD, rigid-ipc, Noteseach baseline descriptionlayerlevel, descriptionanddescriptiondirect comparisondescription. ",
            "",
            "descriptionavoiddescription: `RTSTPFExact faster than all baselines on all workloads`. ",
            "",
            "currentdescription: `RTSTPFExact is a correctness-preserving learned scheduling layer that reduces exact certificate workload and native dense-group wall time on high-candidate/high-cost workloads, while retaining exact-certificate fallback for zero false negatives.`",
            "",
            "## 7. afterdescription",
            "",
            "- `CCD-Wrapper`: description build in stale absolute CCD-Wrapper CTest Path, description unit-tests / erleben / golf-ball normalized smoke. ",
            "- `Exact-Root-Parity-CCD`: descriptionthroughdescription CTest, description grazing/root parity hard cases as appendix correctness reference. ",
            "- `Scalable-CCD`: CUDA broad/narrow tests descriptionthrough, description converted candidate group description; descriptionconvertscenedescription supplementary. ",
            "- `rigid-ipc`: asrealdescriptiontrajectorydescription, underdescription 2~3  scene generate P2CCCD candidate groups andvisualization. ",
            "- `Tight-Inclusion dense group`: description native proxy exact payload descriptionreal TI/CUDA primitive exact payload. ",
            "",
        ]
    )


def run(root: Path, output_dir: Path, run_name: str = RUN_NAME) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run_name": run_name,
        "root": str(root),
        "baselines": [asdict(entry) for entry in _baseline_entries(root)],
        "comparison_levels": [asdict(level) for level in _comparison_levels()],
        "status": _status_from_artifacts(root),
    }

    json_path = output_dir / f"{run_name}.json"
    md_path = output_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    payload["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("src/benchmark"))
    parser.add_argument("--run-name", default=RUN_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(root=args.root, output_dir=args.output_dir, run_name=args.run_name)
    print(json.dumps(payload["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
