"""Structured adapter specification extracted from docs or provided by user."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class InstallSpec(BaseModel):
    """How to install the migration library and its dependencies."""

    commands: list[str] = Field(default_factory=list, description="Shell commands, e.g. pip install")
    pip_packages: list[str] = Field(default_factory=list)
    notes: str = ""


class EnableSpec(BaseModel):
    """Code that must run before/around torch to enable the migration layer."""

    preamble: str = Field(
        ...,
        description="Python source prepended to test harness (imports + enable calls)",
    )
    import_order_note: str = ""


class DeviceSpec(BaseModel):
    """Target device and how to place tensors / models."""

    target_device: str = Field(
        "cpu",
        description="e.g. cpu, cuda:0, jax, or library-specific",
    )
    tensor_device_code: str | None = Field(
        None,
        description="Optional: expression for default device, e.g. 'jax' for torchax",
    )
    model_to_device: str | None = Field(
        None,
        description="Optional line to move model, e.g. m = m.to('jax')",
    )


class ConstraintItem(BaseModel):
    """Known limitation from docs."""

    api_or_area: str
    reason: str = ""


class ExampleSnippet(BaseModel):
    """Example from documentation with traceability."""

    title: str = ""
    code: str
    source_ref: str = ""


class Evidence(BaseModel):
    """Quote from docs supporting a field."""

    field: str
    quote: str
    source: str = ""


class ConfidenceSpec(BaseModel):
    """LLM self-assessment; does not replace schema validation."""

    overall: float = 0.0
    by_field: dict[str, float] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)


class AdapterSpec(BaseModel):
    """Full adapter description for a migration library."""

    library_name: str = Field(..., description="Short name, e.g. torchax, mindtorch")
    version_hint: str | None = None
    install: InstallSpec = Field(default_factory=InstallSpec)
    enable: EnableSpec
    device: DeviceSpec = Field(default_factory=DeviceSpec)
    constraints: list[ConstraintItem] = Field(default_factory=list)
    examples: list[ExampleSnippet] = Field(default_factory=list)
    confidence: ConfidenceSpec | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("library_name")
    @classmethod
    def non_empty_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("library_name is required")
        return v.strip()

    @field_validator("enable")
    @classmethod
    def preamble_non_empty(cls, v: EnableSpec) -> EnableSpec:
        if not v.preamble or not v.preamble.strip():
            raise ValueError("enable.preamble is required")
        return v


def validate_adapter_dict(data: dict[str, Any]) -> AdapterSpec:
    """Parse and validate user or LLM output."""
    return AdapterSpec.model_validate(data)


def adapter_spec_to_dict(spec: AdapterSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json")
