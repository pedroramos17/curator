from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.domain.core import CoherenceMetrics, ExecutionMode, MultiscaleProfile, NarrativeState, RenderPlan, SequenceState, StructuredData
from src.domain.system_state import SystemState, TransitionResult
from src.pipeline.bootstrap import BootstrapConfig, build_pipeline_dependencies
from src.pipeline.orchestrator import PipelineDependencies, PipelineOrchestrator


class _StubCutter:
    def cut(self, plan: RenderPlan) -> dict[str, object]:
        return {"module": "cutter", "segment_id": plan.segment_id}


def test_orchestrator_accepts_injected_dependencies() -> None:
    state = SystemState()
    structured = StructuredData(segment_id="seg-1", start_time=0.0, end_time=1.0)
    sequence = SequenceState(segment_id="seg-1", latent_state=(0.1, 0.2), transition_score=0.95)
    narrative = NarrativeState(segment_id="seg-1")
    coherence = CoherenceMetrics(segment_id="seg-1", kl_divergence=0.1, wasserstein_distance=0.1, cosine_distance=0.1)
    multiscale = MultiscaleProfile(
        segment_id="seg-1",
        frame_level={"density": 0.5},
        shot_level={"cadence": 0.6},
        scene_level={"continuity": 0.7},
        act_level={"arc": 0.8},
        regime_shift_score=0.1,
    )
    plan = RenderPlan(segment_id="seg-1", mode=ExecutionMode.CUT_ONLY, selected_actions=("cut",))

    phase2 = Mock()
    phase2.execute.return_value = TransitionResult(output=structured, state=state)
    phase3 = Mock()
    phase3.execute.return_value = TransitionResult(output=sequence, state=state)
    phase4 = Mock()
    phase4.execute.return_value = TransitionResult(output=narrative, state=state)
    phase5 = Mock()
    phase5.execute.return_value = TransitionResult(output=coherence, state=state)
    phase6 = Mock()
    phase6.execute.return_value = TransitionResult(output=multiscale, state=state)
    router = Mock()
    router.execute.return_value = ExecutionMode.CUT_ONLY
    phase7 = Mock()
    phase7.execute.return_value = TransitionResult(output=plan, state=state)

    dependencies = PipelineDependencies(
        phase2=phase2,
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        phase6=phase6,
        router=router,
        phase7=phase7,
        cutter=_StubCutter(),
        audio=None,
        video=None,
        subtitles=None,
        haptic=None,
    )

    orchestrator = PipelineOrchestrator(dependencies)
    result = orchestrator.run("video.mp4", "audio.wav", [{"speaker": "narrator", "start": 0.0, "end": 1.0, "text": "hello"}])

    assert result.render_plan.mode is ExecutionMode.CUT_ONLY
    assert result.execution_output["steps"][0]["module"] == "cutter"


def test_bootstrap_validates_execution_backend_before_adapter_build() -> None:
    with pytest.raises(ValueError, match="Unsupported execution backend"):
        build_pipeline_dependencies(BootstrapConfig(execution_backend="remote"))
