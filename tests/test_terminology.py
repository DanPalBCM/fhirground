"""Terminology resolvers: local value sets, chains, null, and RxNav via respx."""

from __future__ import annotations

import json
from typing import ClassVar

import httpx
import pytest
import respx

from fhirground.exceptions import TerminologyError
from fhirground.schemas.mention import MentionType
from fhirground.terminology import (
    RXNORM_SYSTEM,
    Chain,
    LocalValueSet,
    NullResolver,
    Resolution,
    ResolverConfig,
    RxNavResolver,
)
from tests.conftest import CONDITION_ROWS, SNOMED


@pytest.fixture
def conditions() -> LocalValueSet:
    return LocalValueSet.from_entries(CONDITION_ROWS, name="conds")


class TestLocalValueSet:
    def test_exact_display_scores_one(self, conditions):
        r = conditions.resolve("Type 2 diabetes mellitus", MentionType.CONDITION)
        assert r.coding.code == "44054006"
        assert r.method == "exact"
        assert r.score == 1.0

    def test_synonym_scores_slightly_lower(self, conditions):
        r = conditions.resolve("T2DM", MentionType.CONDITION)
        assert r.coding.code == "44054006"
        assert r.method == "synonym"
        assert r.score == 0.95

    def test_fuzzy_handles_word_reorder(self, conditions):
        r = conditions.resolve("diabetes mellitus type 2", MentionType.CONDITION)
        assert r.coding.code == "44054006"
        assert r.method == "fuzzy"

    def test_miss_returns_none_so_a_chain_continues(self, conditions):
        assert conditions.resolve("fractured femur", MentionType.CONDITION) is None

    def test_display_comes_from_the_value_set_not_the_input(self, conditions):
        """Otherwise a resource carries a real code beside invented wording."""
        r = conditions.resolve("T2DM", MentionType.CONDITION)
        assert r.coding.display == "Type 2 diabetes mellitus"
        assert r.text == "T2DM"

    def test_fuzzy_can_be_disabled(self):
        vs = LocalValueSet.from_entries(CONDITION_ROWS, fuzzy=False)
        assert vs.resolve("diabetes mellitus type 2", MentionType.CONDITION) is None

    def test_threshold_is_respected(self):
        strict = LocalValueSet.from_entries(CONDITION_ROWS, fuzzy_threshold=99.9)
        assert strict.resolve("diabetes mellitus ty", MentionType.CONDITION) is None

    def test_candidates_recorded_for_fuzzy_hits(self, conditions):
        r = conditions.resolve("diabetes mellitus type 2", MentionType.CONDITION)
        assert isinstance(r.candidates, tuple)

    def test_known_codes(self, conditions):
        codes = conditions.known_codes()
        assert (SNOMED, "44054006") in codes
        assert len(codes) == len(CONDITION_ROWS)

    def test_empty_value_set_rejected(self):
        with pytest.raises(TerminologyError, match="is empty"):
            LocalValueSet([])

    def test_blank_input(self, conditions):
        assert conditions.resolve("", MentionType.CONDITION) is None
        assert conditions.resolve("   ", MentionType.CONDITION) is None

    def test_punctuation_tolerant(self, conditions):
        assert conditions.resolve("hypertension.", MentionType.CONDITION).coding.code == "38341003"


class TestCsvAndJsonLoading:
    def test_csv_round_trip(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text(
            "system,code,display,synonyms\n"
            f"{SNOMED},44054006,Type 2 diabetes mellitus,type 2 diabetes|T2DM\n"
        )
        vs = LocalValueSet.from_csv(path)
        assert len(vs) == 1
        assert vs.resolve("T2DM", MentionType.CONDITION).coding.code == "44054006"
        assert vs.name == "local:c"

    def test_csv_missing_column_names_what_it_needs(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("system,code\nx,y\n")
        with pytest.raises(TerminologyError, match="missing column"):
            LocalValueSet.from_csv(path)

    def test_csv_blank_required_cell_reports_the_line(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("system,code,display\nx,,z\n")
        with pytest.raises(TerminologyError, match=":2:"):
            LocalValueSet.from_csv(path)

    def test_csv_missing_file(self, tmp_path):
        with pytest.raises(TerminologyError, match="not found"):
            LocalValueSet.from_csv(tmp_path / "nope.csv")

    def test_json_round_trip(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps(CONDITION_ROWS))
        vs = LocalValueSet.from_json(path)
        assert len(vs) == len(CONDITION_ROWS)

    def test_json_must_be_an_array(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(TerminologyError, match="array"):
            LocalValueSet.from_json(path)

    def test_json_malformed(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{oops")
        with pytest.raises(TerminologyError, match="malformed JSON"):
            LocalValueSet.from_json(path)


class TestNullResolver:
    def test_never_resolves(self):
        n = NullResolver()
        assert n.resolve("type 2 diabetes", MentionType.CONDITION) is None

    def test_emits_no_codes(self):
        assert NullResolver().known_codes() == set()


class TestChain:
    def test_first_hit_wins(self, conditions):
        second = LocalValueSet.from_entries(
            [{"system": "http://example.org", "code": "X", "display": "Type 2 diabetes mellitus"}],
            name="second",
        )
        chain = Chain([conditions, second])
        r = chain.resolve("Type 2 diabetes mellitus", MentionType.CONDITION)
        assert r.coding.code == "44054006"
        assert r.resolver == "conds"

    def test_fallthrough_to_the_next_resolver(self, conditions):
        second = LocalValueSet.from_entries(
            [{"system": "http://example.org", "code": "FEM", "display": "Fractured femur"}],
            name="second",
        )
        chain = Chain([conditions, second])
        r = chain.resolve("Fractured femur", MentionType.CONDITION)
        assert r.coding.code == "FEM"
        assert r.resolver == "second"

    def test_best_strategy_picks_the_higher_score(self):
        weak = LocalValueSet.from_entries(
            [{"system": "http://a", "code": "A", "display": "diabetes mellitus ty 2"}], name="weak"
        )
        strong = LocalValueSet.from_entries(
            [{"system": "http://b", "code": "B", "display": "type 2 diabetes"}], name="strong"
        )
        chain = Chain([weak, strong], strategy="best")
        r = chain.resolve("type 2 diabetes", MentionType.CONDITION)
        assert r.coding.code == "B"

    def test_min_score_skips_weak_hits(self, conditions):
        backup = LocalValueSet.from_entries(
            [{"system": "http://b", "code": "B", "display": "diabetes mellitus type 2"}],
            name="backup",
        )
        chain = Chain([conditions, backup], min_score=0.99)
        r = chain.resolve("Type 2 diabetes mellitus", MentionType.CONDITION)
        assert r.score >= 0.99

    def test_all_miss_returns_none(self, conditions):
        assert Chain([conditions]).resolve("pulmonary embolism", MentionType.CONDITION) is None

    def test_empty_chain_rejected(self):
        with pytest.raises(TerminologyError, match="at least one resolver"):
            Chain([])

    def test_unknown_strategy_rejected(self, conditions):
        with pytest.raises(TerminologyError, match="unknown chain strategy"):
            Chain([conditions], strategy="vibes")

    def test_known_codes_is_the_union(self, conditions):
        other = LocalValueSet.from_entries(
            [{"system": "http://b", "code": "B", "display": "x"}], name="other"
        )
        assert Chain([conditions, other]).known_codes() == conditions.known_codes() | {
            ("http://b", "B")
        }


class TestResolverConfig:
    def test_per_type_routing(self, conditions):
        cfg = ResolverConfig(condition=conditions)
        assert cfg.for_type(MentionType.CONDITION) is conditions
        assert cfg.for_type(MentionType.MEDICATION) is None

    def test_resolve_always_returns_a_resolution(self, conditions):
        cfg = ResolverConfig(condition=conditions)
        hit = cfg.resolve("T2DM", MentionType.CONDITION)
        miss = cfg.resolve("T2DM", MentionType.MEDICATION)
        assert isinstance(hit, Resolution) and hit.is_coded
        assert isinstance(miss, Resolution) and not miss.is_coded
        assert miss.score == 0.0

    def test_text_only_codeable_concept(self, conditions):
        cfg = ResolverConfig(condition=conditions)
        cc = cfg.resolve("fractured femur", MentionType.CONDITION).as_codeable_concept()
        assert cc == {"text": "fractured femur"}

    def test_coded_codeable_concept(self, conditions):
        cfg = ResolverConfig(condition=conditions)
        cc = cfg.resolve("T2DM", MentionType.CONDITION).as_codeable_concept()
        assert cc["coding"][0]["code"] == "44054006"
        assert cc["text"] == "T2DM"

    def test_describe(self, conditions):
        cfg = ResolverConfig(condition=conditions)
        assert cfg.describe() == {"condition": "conds", "medication": "none", "observation": "none"}


class TestRxNav:
    """Network is mocked with respx; no test here reaches NLM."""

    PAYLOAD: ClassVar[dict] = {
        "approximateGroup": {
            "candidate": [
                {"rxcui": "860975", "name": "metformin 500 MG", "score": "85"},
                {"rxcui": "860974", "name": "metformin", "score": "70"},
            ]
        }
    }

    @respx.mock
    def test_resolves_a_medication(self, tmp_path):
        route = respx.get(
            url__startswith="https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
        ).mock(return_value=httpx.Response(200, json=self.PAYLOAD))
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite")
        r = resolver.resolve("metformin 500 mg", MentionType.MEDICATION)
        assert route.called
        assert r.coding.system == RXNORM_SYSTEM
        assert r.coding.code == "860975"
        assert r.method == "remote"
        assert r.score == pytest.approx(0.85)

    @respx.mock
    def test_second_lookup_hits_the_cache(self, tmp_path):
        route = respx.get(
            url__startswith="https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
        ).mock(return_value=httpx.Response(200, json=self.PAYLOAD))
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite")
        resolver.resolve("metformin", MentionType.MEDICATION)
        resolver.resolve("metformin", MentionType.MEDICATION)
        assert route.call_count == 1

    @respx.mock
    def test_non_medication_types_are_not_queried(self, tmp_path):
        route = respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST").mock(
            return_value=httpx.Response(200, json=self.PAYLOAD)
        )
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite")
        assert resolver.resolve("type 2 diabetes", MentionType.CONDITION) is None
        assert not route.called

    @respx.mock
    def test_low_score_is_rejected(self, tmp_path):
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST").mock(
            return_value=httpx.Response(
                200,
                json={
                    "approximateGroup": {"candidate": [{"rxcui": "1", "name": "x", "score": "10"}]}
                },
            )
        )
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite", min_score=50.0)
        assert resolver.resolve("zzz", MentionType.MEDICATION) is None

    @respx.mock
    def test_no_candidates(self, tmp_path):
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST").mock(
            return_value=httpx.Response(200, json={"approximateGroup": {}})
        )
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite")
        assert resolver.resolve("zzz", MentionType.MEDICATION) is None

    @respx.mock
    def test_http_error_is_a_terminology_error(self, tmp_path):
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST").mock(
            return_value=httpx.Response(503)
        )
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite")
        with pytest.raises(TerminologyError, match="RxNav request failed"):
            resolver.resolve("metformin", MentionType.MEDICATION)

    @respx.mock
    def test_offline_mode_never_calls_out(self, tmp_path):
        route = respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST").mock(
            return_value=httpx.Response(200, json=self.PAYLOAD)
        )
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite", offline=True)
        assert resolver.resolve("metformin", MentionType.MEDICATION) is None
        assert not route.called

    @respx.mock
    def test_known_codes_reflects_the_cache(self, tmp_path):
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST").mock(
            return_value=httpx.Response(200, json=self.PAYLOAD)
        )
        resolver = RxNavResolver(cache_path=tmp_path / "c.sqlite")
        resolver.resolve("metformin", MentionType.MEDICATION)
        assert (RXNORM_SYSTEM, "860975") in resolver.known_codes()

    @pytest.mark.live_rxnav
    def test_live_lookup(self):
        with RxNavResolver(cache_path=None) as resolver:
            r = resolver.resolve("metformin", MentionType.MEDICATION)
            assert r is not None
            assert r.coding.system == RXNORM_SYSTEM
