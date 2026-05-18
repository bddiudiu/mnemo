"""Unit tests for working memory."""

import pytest

from mnemo.core.working_memory import WorkingMemory


class TestWorkingMemory:
    def test_keyword_score_exact_match(self):
        score = WorkingMemory._keyword_score(
            "User prefers Python 3.12",
            "python",
        )
        assert score == 1.0

    def test_keyword_score_partial_match(self):
        score = WorkingMemory._keyword_score(
            "User prefers Python 3.12 with asyncio",
            "python version",
        )
        assert 0.0 < score < 1.0

    def test_keyword_score_no_match(self):
        score = WorkingMemory._keyword_score(
            "User prefers Python 3.12",
            "javascript",
        )
        assert score == 0.0
