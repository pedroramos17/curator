from __future__ import annotations

from dataclasses import dataclass

from src.adapters.encoding.sequence_encoder_adapter import SequenceEncoderAdapter
from src.adapters.execution.local_execution import (
    LocalAudioGenerator,
    LocalCutter,
    LocalHapticModule,
    LocalSubtitleGenerator,
    LocalVideoGenerator,
)
from src.adapters.graph.ruvector_graph_repository import InMemoryRuVectorClient, RuVectorClientProtocol, RuVectorGraphRepository
from src.adapters.reasoning.context_reasoning_adapter import ContextReasoningAdapter
from src.application.use_cases import (
    CostAwareRoutingUseCase,
    Phase2IngestUseCase,
    Phase3TemporalEncodingUseCase,
    Phase4ContextReasoningUseCase,
    Phase5CoherenceUseCase,
    Phase6MultiscaleUseCase,
    Phase7DecisionExecutionUseCase,
)
from src.pipeline.orchestrator import PipelineDependencies, PipelineOrchestrator


@dataclass(frozen=True)
class BootstrapConfig:
    graph_schema_path: str = "data_structures/graph_schema.json"
    ruvector_client: str = "in_memory"
    execution_backend: str = "local"


def _build_ruvector_client(client_type: str) -> RuVectorClientProtocol:
    if client_type == "in_memory":
        return InMemoryRuVectorClient()
    raise ValueError(f"Unsupported RuVector client type: {client_type}")


def build_pipeline_dependencies(config: BootstrapConfig | None = None) -> PipelineDependencies:
    cfg = config or BootstrapConfig()
    graph = RuVectorGraphRepository(client=_build_ruvector_client(cfg.ruvector_client))

    if cfg.execution_backend != "local":
        raise ValueError(f"Unsupported execution backend: {cfg.execution_backend}")

    from src.adapters.media.pipeline_ingest_adapter import PipelineIngestAdapter

    return PipelineDependencies(
        phase2=Phase2IngestUseCase(PipelineIngestAdapter(graph_schema_path=cfg.graph_schema_path), graph),
        phase3=Phase3TemporalEncodingUseCase(SequenceEncoderAdapter()),
        phase4=Phase4ContextReasoningUseCase(ContextReasoningAdapter(graph)),
        phase5=Phase5CoherenceUseCase(),
        phase6=Phase6MultiscaleUseCase(),
        router=CostAwareRoutingUseCase(),
        phase7=Phase7DecisionExecutionUseCase(),
        cutter=LocalCutter(),
        audio=LocalAudioGenerator(),
        video=LocalVideoGenerator(),
        subtitles=LocalSubtitleGenerator(),
        haptic=LocalHapticModule(),
    )


def build_pipeline_orchestrator(config: BootstrapConfig | None = None) -> PipelineOrchestrator:
    return PipelineOrchestrator(dependencies=build_pipeline_dependencies(config))
