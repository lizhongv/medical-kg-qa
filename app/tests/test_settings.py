from app.settings import load_settings


def test_defaults_when_env_empty():
    s = load_settings({})
    assert s.accept_threshold == 0.8
    assert s.deny_threshold == 0.4
    assert s.neo4j_uri == "bolt://127.0.0.1:7687"
    assert s.llm_api_key is None


def test_reads_from_env():
    s = load_settings({
        "KBQA_ACCEPT_THRESHOLD": "0.7",
        "KBQA_LLM_API_KEY": "sk-x",
        "KBQA_LLM_MODEL": "gpt-4o-mini",
    })
    assert s.accept_threshold == 0.7
    assert s.llm_api_key == "sk-x"
    assert s.llm_model == "gpt-4o-mini"
