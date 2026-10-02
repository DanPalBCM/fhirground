"""Exception hierarchy for fhirground.

Errors carry enough context to locate the problem: a note id, a chunk id, or a
file path and line number. None of them carry note **text**, because log and
error streams are exactly where note text should not end up.
"""

from __future__ import annotations


class FhirgroundError(Exception):
    """Base class for every error raised by fhirground."""


class InputError(FhirgroundError):
    """An input note or file could not be read or validated."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        note_id: str | None = None,
    ) -> None:
        self.path = path
        self.line = line
        self.note_id = note_id
        bits = []
        if path is not None:
            bits.append(f"{path}:{line}" if line is not None else path)
        if note_id is not None:
            bits.append(f"note_id={note_id}")
        suffix = f" [{', '.join(bits)}]" if bits else ""
        super().__init__(f"{message}{suffix}")


class ConfigError(FhirgroundError):
    """The pipeline configuration is missing, malformed, or self-contradictory."""


class BackendError(FhirgroundError):
    """A backend failed to produce a usable response.

    Distinguished from :class:`SchemaValidationError` because a transport or
    auth failure is not the model's fault and should not consume repair
    attempts. The CLI maps this to exit code 4.
    """


class SchemaValidationError(FhirgroundError):
    """A backend response did not parse or did not satisfy the mention schema.

    This is the error fed back into the repair loop, so its message is written
    to be useful *to a model*: it names the offending field and what was
    expected.
    """

    def __init__(self, message: str, *, raw_excerpt: str | None = None) -> None:
        self.raw_excerpt = raw_excerpt
        super().__init__(message)


class GroundingError(FhirgroundError):
    """A mention's evidence could not be located in the source text.

    Normally handled as a drop rather than raised; it exists for callers that
    want grounding failures to be fatal.
    """


class TerminologyError(FhirgroundError):
    """A terminology resolver could not be constructed or queried."""


class MappingError(FhirgroundError):
    """A mention could not be mapped to a FHIR resource."""


class ValidationRuleError(FhirgroundError):
    """A produced resource violated a project validation rule.

    Carries the rule name so the failure is actionable rather than a generic
    "invalid resource".
    """

    def __init__(self, message: str, *, rule: str, resource_type: str | None = None) -> None:
        self.rule = rule
        self.resource_type = resource_type
        where = f" [{resource_type}]" if resource_type else ""
        super().__init__(f"{rule}: {message}{where}")


class CalibrationError(FhirgroundError):
    """A calibrator could not be fitted, loaded, or applied."""
