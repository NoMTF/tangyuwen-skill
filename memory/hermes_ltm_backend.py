#!/usr/bin/env python3
"""
Hermes-style long-term memory backend (high-availability local edition).

This version stores AGENT SELF-MEMORY (not user preference profiling):
- persona invariants
- successful playbooks
- failure postmortems
- project continuity
- tool knowledge
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("LTM_DB_PATH", "./memory/ltm.db"))
EVENT_LOG_PATH = Path(os.getenv("LTM_EVENT_LOG", "./memory/ltm.events.jsonl"))
SNAPSHOT_PATH = Path(os.getenv("LTM_SNAPSHOT", "./memory/ltm.snapshot.json"))

VALID_CLASSES = {
    "persona_invariants",
    "successful_playbooks",
    "failure_postmortems",
    "project_continuity",
    "tool_knowledge",
}


@dataclass
class MemoryItem:
    agent_id: str
    memory_class: str
    key: str
    value: str
    source: str
    impact: float
    confidence: float
    tags: list[str]


class LTMSelfMemoryStore:
    """Thread-safe, SQLite-backed self-memory store with basic HA primitives."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=FULL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_class TEXT NOT NULL,
                    mkey TEXT NOT NULL,
                    mvalue TEXT NOT NULL,
                    source TEXT NOT NULL,
                    impact REAL NOT NULL,
                    confidence REAL NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at INTEGER,
                    UNIQUE(agent_id, memory_class, mkey)
                );

                CREATE TABLE IF NOT EXISTS memory_events (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
                CREATE INDEX IF NOT EXISTS idx_memories_class ON memories(agent_id, memory_class);
                """
            )

    def healthcheck(self) -> dict[str, Any]:
        start = time.time()
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1")
            status = "ok"
            error = None
        except Exception as exc:
            status = "degraded"
            error = str(exc)
        return {
            "status": status,
            "latency_ms": round((time.time() - start) * 1000, 2),
            "db_path": str(self.db_path),
            "event_log_exists": EVENT_LOG_PATH.exists(),
            "snapshot_exists": SNAPSHOT_PATH.exists(),
            "error": error,
        }

    def upsert_memory(self, item: MemoryItem) -> None:
        if item.memory_class not in VALID_CLASSES:
            raise ValueError(f"invalid memory_class: {item.memory_class}")

        now = int(time.time())
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, agent_id, memory_class, mkey, mvalue, source, impact, confidence,
                        tags_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id, memory_class, mkey) DO UPDATE SET
                        mvalue = excluded.mvalue,
                        source = excluded.source,
                        impact = excluded.impact,
                        confidence = excluded.confidence,
                        tags_json = excluded.tags_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(uuid.uuid4()),
                        item.agent_id,
                        item.memory_class,
                        item.key,
                        item.value,
                        item.source,
                        max(0.0, min(1.0, item.impact)),
                        max(0.0, min(1.0, item.confidence)),
                        json.dumps(item.tags, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            self._append_event(item.agent_id, "upsert", item.__dict__)

    def retrieve(self, agent_id: str, task_query: str, limit: int = 6) -> list[dict[str, Any]]:
        """
        Hermes-like ranking for self-memory:
        score = 0.55 * impact + 0.30 * recency + 0.15 * lexical_match

        Persona invariants are always prioritized as hard constraints.
        """
        now = int(time.time())
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE agent_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 300
                    """,
                    (agent_id,),
                ).fetchall()

        query_terms = [t.lower() for t in task_query.split() if t.strip()]
        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            age_days = max(0.0, (now - row["updated_at"]) / 86400)
            recency = 1.0 / (1.0 + age_days / 14.0)
            blob = f"{row['memory_class']} {row['mkey']} {row['mvalue']} {row['tags_json']}".lower()
            lexical = 0.0
            if query_terms:
                hits = sum(1 for t in query_terms if t in blob)
                lexical = hits / len(query_terms)

            base = 0.55 * float(row["impact"]) + 0.30 * recency + 0.15 * lexical
            hard_boost = 1.0 if row["memory_class"] == "persona_invariants" else 0.0
            ranked.append((base + hard_boost, row))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:limit]

        results: list[dict[str, Any]] = []
        for score, row in top:
            results.append(
                {
                    "memory_class": row["memory_class"],
                    "key": row["mkey"],
                    "value": row["mvalue"],
                    "impact": row["impact"],
                    "confidence": row["confidence"],
                    "tags": json.loads(row["tags_json"]),
                    "updated_at": row["updated_at"],
                    "score": round(score, 4),
                }
            )

        self._touch_access(agent_id, [(r["memory_class"], r["key"]) for r in results])
        self._append_event(agent_id, "retrieve", {"task_query": task_query, "limit": limit, "hits": len(results)})
        return results

    def _touch_access(self, agent_id: str, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        now = int(time.time())
        with self._conn() as conn:
            for memory_class, key in pairs:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed_at = ?
                    WHERE agent_id = ? AND memory_class = ? AND mkey = ?
                    """,
                    (now, agent_id, memory_class, key),
                )

    def _append_event(self, agent_id: str, event_type: str, payload: dict[str, Any]) -> None:
        now = int(time.time())
        event = {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": now,
        }
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memory_events (id, agent_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (event["id"], agent_id, event_type, json.dumps(payload, ensure_ascii=False), now),
            )
        with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def snapshot(self) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT agent_id, memory_class, mkey, mvalue, source, impact, confidence, tags_json, updated_at FROM memories"
            ).fetchall()

        data = [
            {
                "agent_id": r["agent_id"],
                "memory_class": r["memory_class"],
                "key": r["mkey"],
                "value": r["mvalue"],
                "source": r["source"],
                "impact": r["impact"],
                "confidence": r["confidence"],
                "tags": json.loads(r["tags_json"]),
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
        payload = {"generated_at": int(time.time()), "count": len(data), "items": data}
        SNAPSHOT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


if __name__ == "__main__":
    store = LTMSelfMemoryStore()

    store.upsert_memory(
        MemoryItem(
            agent_id="laotang_core",
            memory_class="persona_invariants",
            key="response_structure",
            value="默认短句+多条连发，除非用户明确触发/long",
            source="style_guardrail",
            impact=1.0,
            confidence=1.0,
            tags=["style", "hard_constraint"],
        )
    )

    store.upsert_memory(
        MemoryItem(
            agent_id="laotang_core",
            memory_class="failure_postmortems",
            key="ltm_scope_error",
            value="长期记忆应优先存储模型经验，不做用户偏好画像",
            source="review_feedback",
            impact=0.95,
            confidence=0.98,
            tags=["scope", "correction"],
        )
    )

    print("health:", json.dumps(store.healthcheck(), ensure_ascii=False, indent=2))
    print("retrieve:", json.dumps(store.retrieve("laotang_core", "风格稳定 失败复盘"), ensure_ascii=False, indent=2))
    print("snapshot_count:", store.snapshot()["count"])
