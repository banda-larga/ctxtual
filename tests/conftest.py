"""
Shared test fixtures for ctx.
"""

import pytest

from ctxtual import Ctx, MemoryStore
from ctxtual.store.sqlite import SQLiteStore

# Sample data

SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani", "Shazeer", "Parmar"],
        "year": 2017,
        "citations": 90000,
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
        "year": 2019,
        "citations": 70000,
        "abstract": "We introduce a new language representation model called BERT.",
    },
    {
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "authors": ["Brown", "Mann", "Ryder"],
        "year": 2020,
        "citations": 25000,
        "abstract": "We demonstrate that scaling up language models greatly improves task-agnostic few-shot performance.",
    },
    {
        "title": "Scaling Laws for Neural Language Models",
        "authors": ["Kaplan", "McCandlish", "Henighan"],
        "year": 2020,
        "citations": 5000,
        "abstract": "We study empirical scaling laws for language model performance.",
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Wei", "Wang", "Schuurmans"],
        "year": 2022,
        "citations": 8000,
        "abstract": "We explore how chain-of-thought prompting improves reasoning capabilities.",
    },
]


@pytest.fixture
def sample_papers() -> list[dict]:
    """A reusable list of fake research paper dicts."""
    return list(SAMPLE_PAPERS)


# Store fixtures


@pytest.fixture
def memory_store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteStore:
    """An on-disk SQLiteStore in a temporary directory."""
    return SQLiteStore(tmp_path / "test.db")


@pytest.fixture
def sqlite_memory_store() -> SQLiteStore:
    """An in-memory SQLiteStore (no disk I/O)."""
    return SQLiteStore(":memory:")


# Ctx fixtures


@pytest.fixture
def ctx(memory_store: MemoryStore) -> Ctx:
    """Ctx backed by MemoryStore."""
    return Ctx(store=memory_store)


@pytest.fixture
def forge_sqlite(sqlite_memory_store: SQLiteStore) -> Ctx:
    """Ctx backed by an in-memory SQLiteStore."""
    return Ctx(store=sqlite_memory_store)
