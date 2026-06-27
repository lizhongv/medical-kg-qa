# -*- coding: utf-8 -*-
class KGClient:
    def __init__(self, settings):
        self.settings = settings
        self._driver = None
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        except Exception:
            self._driver = None

    @property
    def available(self) -> bool:
        return self._driver is not None

    def query(self, cypher, params=None):
        if not self.available:
            return []
        params = params or {}
        try:
            with self._driver.session() as session:
                return list(session.run(cypher, **params).data())
        except Exception:
            return []
