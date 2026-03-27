from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.application.use_cases import (
    CostAwareRoutingUseCase,
    Phase2IngestUseCase,
    Phase3TemporalEncodingUseCase,
    Phase4ContextReasoningUseCase,
    Phase5CoherenceUseCase,
    Phase6MultiscaleUseCase,
    Phase7DecisionExecutionUseCase,
)
from src.domain.core import RenderPlan
from src.domain.system_state import SystemState
from src.ports.execution import AudioGenerationPort, CutterPort, HapticPort, SubtitlePort, VideoGenerationPort
from src.utils.execution import execute_plan


@dataclass(frozen=True)
class PipelineExecutionResult:
    render_plan: RenderPlan
    execution_output: dict[str, object]


@dataclass(frozen=True)
class PipelineDependencies:
    phase2: Phase2IngestUseCase
    phase3: Phase3TemporalEncodingUseCase
    phase4: Phase4ContextReasoningUseCase
    phase5: Phase5CoherenceUseCase
    phase6: Phase6MultiscaleUseCase
    router: CostAwareRoutingUseCase
    phase7: Phase7DecisionExecutionUseCase
    cutter: CutterPort
    audio: AudioGenerationPort | None
    video: VideoGenerationPort | None
    subtitles: SubtitlePort | None
    haptic: HapticPort | None


class PipelineOrchestrator:
    """Orchestrates phase ordering and explicit state transitions."""

    def __init__(self, dependencies: PipelineDependencies) -> None:
        self._dependencies = dependencies

    def run(
        self,
        input_video: str,
        audio_wav_path: str,
        transcript_records: Sequence[dict[str, object]],
        *,
        budget: float = 0.5,
        frame_stride: int = 5,
    ) -> PipelineExecutionResult:
        state = SystemState()
        ingest_transition = self._dependencies.phase2.execute(
            state,
            input_video=input_video,
            audio_wav_path=audio_wav_path,
            transcript_records=transcript_records,
            frame_stride=frame_stride,
        )
        state = ingest_transition.state
        structured = ingest_transition.output

        sequence_transition = self._dependencies.phase3.execute(state, structured)
        state = sequence_transition.state
        sequence_current = sequence_transition.output

        narrative_transition = self._dependencies.phase4.execute(state, sequence_current)
        state = narrative_transition.state
        narrative = narrative_transition.output

        coherence_transition = self._dependencies.phase5.execute(state, sequence_current)
        state = coherence_transition.state
        coherence = coherence_transition.output

        multiscale_transition = self._dependencies.phase6.execute(state, sequence_current, narrative)
        state = multiscale_transition.state
        multiscale = multiscale_transition.output

        mode = self._dependencies.router.execute(budget)
        decision_transition = self._dependencies.phase7.execute(state, narrative, coherence, multiscale, mode)
        plan = decision_transition.output

        execution_output = execute_plan(
            plan,
            self._dependencies.cutter,
            self._dependencies.audio,
            self._dependencies.video,
            self._dependencies.subtitles,
            self._dependencies.haptic,
        )
        return PipelineExecutionResult(render_plan=plan, execution_output=execution_output)
