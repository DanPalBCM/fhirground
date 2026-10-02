"""Backends. The core never parses FHIR out of a model; it parses mentions."""

from fhirground.backends.base import Backend, RawResponse
from fhirground.backends.bedrock import TOOL_NAME, BedrockBackend
from fhirground.backends.mock import MockBackend, extract_by_rules, record_fixtures
from fhirground.backends.openai_compatible import CONSTRAINT_MODES, OpenAICompatible
from fhirground.backends.prompts import (
    PROMPT_VERSION,
    RenderedPrompt,
    render,
    render_repair,
)

__all__ = [
    "CONSTRAINT_MODES",
    "PROMPT_VERSION",
    "TOOL_NAME",
    "Backend",
    "BedrockBackend",
    "MockBackend",
    "OpenAICompatible",
    "RawResponse",
    "RenderedPrompt",
    "extract_by_rules",
    "record_fixtures",
    "render",
    "render_repair",
]
