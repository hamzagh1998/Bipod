from app.services.brain_service import BrainService
import app.services.brain_service as brain_module


def test_extract_web_search_signal_from_exact_marker():
    brain = BrainService()

    result = brain._extract_web_search_signal(
        "[[BIPOD_WEB_SEARCH: current president of france]]",
        "who is the current president of france",
    )

    assert result == "current president of france"


def test_extract_web_search_signal_from_fallback_phrase_uses_user_input():
    brain = BrainService()

    result = brain._extract_web_search_signal(
        "I need current information before I can answer this.",
        "who is the current president of france",
    )

    assert result == "who is the current president of france"


def test_extract_web_search_signal_from_knowledge_cutoff_disclaimer_uses_user_input():
    brain = BrainService()

    result = brain._extract_web_search_signal(
        (
            "As of my knowledge cutoff in October 2023, the current Secretary of Defense "
            "of the United States is Lloyd Austin. However, there may have been a change "
            "in leadership. To ensure accuracy, I recommend checking official U.S. "
            "Department of Defense sources or recent news updates."
        ),
        "who is the current secretary of defense of the united states",
    )

    assert result == "who is the current secretary of defense of the united states"


def test_extract_web_search_signal_ignores_normal_answers():
    brain = BrainService()

    result = brain._extract_web_search_signal(
        "I am Bipod, a local AI assistant running on your machine.",
        "who are you?",
    )

    assert result is None


def test_extract_current_role_lookup_understands_current_role_questions():
    brain = BrainService()

    role_lookup = brain._extract_current_role_lookup(
        "who's the current minister of war of the united state of America?"
    )

    assert role_lookup == ("secretary of defense", "the united states")


def test_run_web_search_prefers_rewritten_query_results(monkeypatch):
    brain = BrainService()

    class FakeDDGS:
        def __init__(self, timeout=20):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def _search(self, query, **kwargs):
            q = query.lower()
            if "minister of war" in q:
                return [
                    {
                        "title": "E-imza Services",
                        "href": "https://example.com/e-imza",
                        "body": "Certificates and unrelated services.",
                    }
                ]
            if "secretary of defense" in q:
                return [
                    {
                        "title": "Secretary of Defense",
                        "href": "https://www.defense.gov/About/Biographies/Secretary-of-Defense/",
                        "body": "Official Department of Defense leadership page.",
                    },
                    {
                        "title": "U.S. Department of Defense",
                        "href": "https://www.defense.gov/",
                        "body": "Official defense information.",
                    },
                ]
            return []

        def _text_html(self, query, **kwargs):
            return self._search(query, **kwargs)

        def _text_lite(self, query, **kwargs):
            return []

        def _text_bing(self, query, **kwargs):
            return []

    monkeypatch.setattr(brain_module, "DDGS", FakeDDGS)

    async def fake_expand(query, base_candidates):
        assert any("secretary of defense" in candidate for candidate in base_candidates)
        return [
            "current united states secretary of defense official",
            *base_candidates,
        ]

    monkeypatch.setattr(brain, "_expand_web_search_candidates_with_model", fake_expand)
    async def fake_enrichment(query, results):
        return ""
    monkeypatch.setattr(brain, "_build_search_result_enrichment", fake_enrichment)
    
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(brain_module.asyncio, "to_thread", fake_to_thread)

    result = brain_module.asyncio.run(
        brain._run_web_search("who's the current minister of war of the united state of America?")
    )

    assert "best query: '" in result
    assert "current united states secretary of defense official" in result.lower()
    assert "https://www.defense.gov/About/Biographies/Secretary-of-Defense/" in result
    assert "E-imza Services" not in result


def test_build_web_search_candidates_for_current_role_avoids_wikipedia_and_keeps_official_focus():
    brain = BrainService()

    candidates = brain._build_web_search_candidates(
        "who is the current secretary of defense of the united states"
    )

    assert any("site:defense.gov" in candidate for candidate in candidates)
    assert not any("wikipedia" in candidate for candidate in candidates)


def test_normalize_web_search_query_maps_historical_war_title_to_secretary_of_defense():
    brain = BrainService()

    normalized = brain._normalize_web_search_query(
        "who is the current minister of war of the united states?"
    )

    assert "secretary of defense" in normalized
    assert "minister of war" not in normalized


def test_extract_result_date_parses_relative_dates():
    brain = BrainService()

    extracted = brain._extract_result_date(
        {
            "title": "Secretary of Defense",
            "href": "https://www.defense.gov/",
            "body": "Official update 2 days ago from the Department of Defense.",
        }
    )

    assert extracted is not None


def test_select_search_results_prefers_newer_official_results_for_current_queries():
    brain = BrainService()
    query = "who is the current secretary of defense of the united states"
    candidates = [query]
    candidate_results = {
        query: [
            {
                "title": "Secretary of Defense",
                "href": "https://www.defense.gov/About/Biographies/Secretary-of-Defense/",
                "body": "Official update 2 years ago from the Department of Defense.",
            },
            {
                "title": "Secretary of Defense",
                "href": "https://www.defense.gov/News/Releases/Release/Article/123456/",
                "body": "Official update 2 days ago from the Department of Defense.",
            },
        ]
    }

    _, results = brain._select_search_results(query, candidates, candidate_results)

    assert results[0]["href"] == "https://www.defense.gov/News/Releases/Release/Article/123456/"


def test_select_search_results_drops_negative_relevance_matches():
    brain = BrainService()
    query = "who is the current secretary of defense of the united states"
    candidates = [query]
    candidate_results = {
        query: [
            {
                "title": "Saudi Press Agency",
                "href": "https://www.spa.gov.sa/en/N2441682",
                "body": "A minister of war meeting abroad.",
            },
            {
                "title": "Secretary of Defense",
                "href": "https://www.defense.gov/About/Biographies/Secretary-of-Defense/",
                "body": "Official Department of Defense leadership page.",
            },
        ]
    }

    _, results = brain._select_search_results(query, candidates, candidate_results)

    assert len(results) == 1
    assert results[0]["href"] == "https://www.defense.gov/About/Biographies/Secretary-of-Defense/"


def test_build_search_result_enrichment_prefers_official_result_pages(monkeypatch):
    brain = BrainService()

    async def fake_fetch(url, query):
        if "defense.gov" in url:
            return "Secretary of Defense official page excerpt."
        return None

    monkeypatch.setattr(brain, "_fetch_search_result_excerpt", fake_fetch)

    enrichment = brain_module.asyncio.run(
        brain._build_search_result_enrichment(
            "who is the current secretary of defense of the united states",
            [
                {
                    "title": "Secretary of Defense",
                    "href": "https://www.defense.gov/About/Biographies/Secretary-of-Defense/",
                    "body": "Official Department of Defense leadership page.",
                },
                {
                    "title": "Unrelated summary",
                    "href": "https://example.com/summary",
                    "body": "Some other snippet.",
                },
            ],
        )
    )

    assert "Fetched page excerpts:" in enrichment
    assert "defense.gov/About/Biographies/Secretary-of-Defense/" in enrichment
    assert "official page excerpt" in enrichment


def test_build_search_result_enrichment_filters_irrelevant_gov_domains_for_us_defense_queries(monkeypatch):
    brain = BrainService()

    async def fake_fetch(url, query):
        return f"Fetched from {url}"

    monkeypatch.setattr(brain, "_fetch_search_result_excerpt", fake_fetch)

    enrichment = brain_module.asyncio.run(
        brain._build_search_result_enrichment(
            "who is the current secretary of defense of the united states",
            [
                {
                    "title": "Saudi Press Agency",
                    "href": "https://www.spa.gov.sa/en/N2441682",
                    "body": "A minister of war meeting abroad.",
                },
                {
                    "title": "War Department",
                    "href": "https://www.war.gov/News/",
                    "body": "Historical military information.",
                },
                {
                    "title": "Secretary of Defense",
                    "href": "https://www.defense.gov/About/Biographies/Secretary-of-Defense/",
                    "body": "Official Department of Defense leadership page.",
                },
            ],
        )
    )

    assert "defense.gov/About/Biographies/Secretary-of-Defense/" in enrichment
    assert "spa.gov.sa" not in enrichment
    assert "war.gov" not in enrichment


def test_filter_grounding_results_keeps_only_preferred_official_domains_for_current_role_queries():
    brain = BrainService()

    results = brain._filter_grounding_results(
        "who is the current secretary of defense of the united states",
        [
            {
                "title": "United States Secretary of Defense - Wikipedia",
                "href": "https://en.wikipedia.org/wiki/United_States_Secretary_of_Defense",
                "body": "Leadership overview and history.",
            },
            {
                "title": "Secretary of Defense",
                "href": "https://www.defense.gov/About/Biographies/Secretary-of-Defense/",
                "body": "Official Department of Defense biography page.",
            },
            {
                "title": "Restoring names that honor American greatness",
                "href": "https://www.whitehouse.gov/presidential-actions/2025/09/restoring-names-that-honor-american-greatness/",
                "body": "Official White House order.",
            },
        ],
    )

    assert len(results) == 2
    assert all(
        any(domain in item["href"] for domain in ("defense.gov", "whitehouse.gov"))
        for item in results
    )
    assert all("wikipedia.org" not in item["href"] for item in results)


def test_filter_grounding_results_keeps_original_results_when_no_preferred_official_match():
    brain = BrainService()

    results = brain._filter_grounding_results(
        "latest league of legends patch notes",
        [
            {
                "title": "Patch Notes",
                "href": "https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-5-notes/",
                "body": "Official patch notes.",
            },
            {
                "title": "Patch summary",
                "href": "https://example.com/lol-patch-summary",
                "body": "Community summary.",
            },
        ],
    )

    assert len(results) == 2
    assert results[0]["href"] == "https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-5-notes/"


def test_complete_with_web_search_uses_clean_grounded_handoff(monkeypatch):
    brain = BrainService()
    seen = {}

    async def fake_run_web_search(query):
        assert query == "who is the current secretary of defense of the united states"
        return "Search results for 'who is the current secretary of defense of the united states':\n1. **Secretary of Defense**"

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "Grounded answer."}}

    class FakeClient:
        async def post(self, url, json):
            seen["messages"] = json["messages"]
            return FakeResponse()

    monkeypatch.setattr(brain, "_run_web_search", fake_run_web_search)

    result = brain_module.asyncio.run(
        brain._complete_with_web_search(
            client=FakeClient(),
            target_model="test-model",
            messages=[
                {"role": "system", "content": "old system"},
                {"role": "assistant", "content": "Lloyd Austin is current."},
            ],
            user_input="who is the current secretary of defense of the united states",
            search_query="who is the current secretary of defense of the united states",
        )
    )

    assert result == "Grounded answer."
    assert len(seen["messages"]) == 3
    assert seen["messages"][0]["role"] == "system"
    assert "Ignore any prior assistant statements" in seen["messages"][1]["content"]
    assert all("Lloyd Austin is current." not in message["content"] for message in seen["messages"])
