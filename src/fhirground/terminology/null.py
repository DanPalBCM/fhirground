"""The null resolver, re-exported for the SPEC §4.1 layout.

:class:`~fhirground.terminology.local.NullResolver` lives beside
``LocalValueSet`` because they share the resolver contract and are read
together; this module is the import path SPEC §4.1 names.
"""

from fhirground.terminology.local import NullResolver

__all__ = ["NullResolver"]
