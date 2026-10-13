"""Pydantic schemas for ARBITER inference endpoints.

Compliant with vLLM Structured Decisions specification (/v1/systemone)
and ARBITER native decision interface (/decide).
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SystemOneQuestion(BaseModel):
    """Specification of a typed question in /v1/systemone.
    
    Per vLLM documentation: Any field other than type, instructions, and criteria
    is rejected with a 400.
    """
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score"] = Field(
        ..., description="Decision type: choice (categorical), noul (boolean), score (ordinal)"
    )
    instructions: Optional[str] = Field(
        None, description="Natural language prompt / question text"
    )
    instruction: Optional[str] = Field(
        None, description="Alias for instructions"
    )
    criteria: Optional[Union[dict[str, Optional[str]], list[str], None]] = Field(
        None, description="Options dictionary, levels list, or boolean criteria mapping"
    )

    @model_validator(mode="after")
    def validate_instructions(self) -> "SystemOneQuestion":
        if self.instructions is None and self.instruction is None:
            raise ValueError("Question must provide either 'instructions' or 'instruction'")
        if (
            self.instructions is not None
            and self.instruction is not None
            and self.instructions != self.instruction
        ):
            raise ValueError("Conflicting values between 'instructions' and 'instruction'")
        return self

    @property
    def prompt_instruction(self) -> str:
        return self.instructions if self.instructions is not None else (self.instruction or "")


class SystemOneRequest(BaseModel):
    """Request payload for /v1/systemone.
    
    Standardized by vLLM for Structured Decisions (System 1 non-causal evaluation).
    """
    model: Optional[str] = Field(
        None, description="Target model ID. If omitted, uses default loaded ARBITER model."
    )
    state: Union[str, dict[str, Any], list[Any]] = Field(
        ..., description="Context for decisions (string or structured JSON object)"
    )
    questions: dict[str, SystemOneQuestion] = Field(
        ..., description="Dictionary of typed questions keyed by non-empty question ID"
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        None, description="Chat template kwargs (ignored for non-causal pooling models)"
    )
    cache_salt: Optional[str] = Field(
        None, description="Optional cache salt identifier"
    )

    # Multimodal rejection fields: text-only models must reject these with 400
    images: Optional[Any] = Field(
        None, description="Image inputs (not supported by text-only model)"
    )
    videos: Optional[Any] = Field(
        None, description="Video inputs (not supported by text-only model)"
    )
    audio: Optional[Any] = Field(
        None, description="Audio inputs (not supported by text-only model)"
    )

    # Top-level instruction rejection: per vLLM doc, top-level instructions returns 400
    instructions: Optional[Any] = Field(
        None, description="Top-level instructions not permitted"
    )
    instruction: Optional[Any] = Field(
        None, description="Top-level instruction not permitted"
    )

    @model_validator(mode="after")
    def validate_request(self) -> "SystemOneRequest":
        if self.instructions is not None or self.instruction is not None:
            raise ValueError(
                "Top-level 'instructions' or 'instruction' is not supported in /v1/systemone. Place instructions inside individual question objects."
            )
        if self.images is not None or self.videos is not None or self.audio is not None:
            raise ValueError(
                "Multimodal inputs (images, videos, audio) are not supported by the text-only ARBITER model."
            )
        if not self.questions:
            raise ValueError("The 'questions' dictionary must not be empty.")
        for qid in self.questions:
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("Question IDs must be non-empty strings.")
        return self


class SystemOneAnswer(BaseModel):
    """Individual answer block in /v1/systemone."""
    type: Literal["choice", "noul", "score"]
    choice: Optional[str] = None
    noul: Optional[float] = None
    score: Optional[float] = None
    legend: Optional[dict[str, str]] = None
    probabilities: dict[str, float]
    confidence: float


class SystemOneUsage(BaseModel):
    """Token usage tracking for /v1/systemone."""
    input_tokens: int
    output_tokens: int = 0


class SystemOneDiagnostic(BaseModel):
    """Diagnostics block per question in /v1/systemone."""
    label_mass: float = 1.0
    argmax_is_label: bool = True


class SystemOneResponse(BaseModel):
    """Full OpenAI-style response for /v1/systemone."""
    id: str
    object: str = "structured_decision"
    created: int
    model: str
    answers: dict[str, SystemOneAnswer]
    usage: SystemOneUsage
    diagnostics: dict[str, SystemOneDiagnostic]


class DecideRequest(BaseModel):
    """Native flat ARBITER decision request for /decide."""
    context: str = Field(..., description="Context passage or document")
    instruction: str = Field(..., description="Decision instruction / question")
    criteria: Optional[Union[dict[str, str], list[str], None]] = Field(
        None, description="Option candidates or rubric"
    )
    question_type: Literal["choice", "noul", "score"] = Field(
        "choice", description="Decision type"
    )
    max_length: int = Field(
        4096, gt=0, description="Maximum total sequence length in tokens"
    )
    allow_truncation: bool = Field(
        False, description="Whether to allow progressive context truncation"
    )


class DecideResponse(BaseModel):
    """Native flat ARBITER decision response for /decide."""
    selected_key: str
    probabilities: dict[str, float]
    confidence: float
    input_tokens: int
    truncated: bool
    score: Optional[float] = None
    noul: Optional[float] = None


class HealthResponse(BaseModel):
    """Service health response."""
    status: str = "healthy"
    model_id: str
    device: str
    loaded: bool = True
    version: str


class MetricsResponse(BaseModel):
    """Service operational metrics."""
    total_requests: int
    total_decisions: int
    total_tokens: int
    avg_batch_size: float
    p50_latency_ms: float
    p95_latency_ms: float
    queue_size: int
    queue_saturation: float
