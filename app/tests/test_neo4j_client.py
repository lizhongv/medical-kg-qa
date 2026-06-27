from app.settings import Settings
from app.kg.neo4j_client import KGClient


class FakeResult(list):
    def data(self):
        return list(self)


class FakeSession:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, cypher, **params):
        return FakeResult([{"name": "高血压"}])


class FakeDriver:
    def session(self): return FakeSession()


def test_unavailable_returns_empty():
    c = KGClient(Settings())
    c._driver = None
    assert c.available is False
    assert c.query("MATCH (n) RETURN n") == []


def test_query_returns_rows():
    c = KGClient(Settings())
    c._driver = FakeDriver()
    rows = c.query("MATCH (p) RETURN p.name", {})
    assert rows == [{"name": "高血压"}]
