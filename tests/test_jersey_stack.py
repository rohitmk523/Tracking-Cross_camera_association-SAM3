"""Jersey stack voting: one confident wrong read must never name a player."""
from __future__ import annotations

from uball_cc.tracking.jersey_stack import vote_number


def test_agreeing_reads_commit():
    assert vote_number(["23", "23", "23"]) == 23


def test_single_read_abstains():
    assert vote_number(["7"]) is None                 # one read is never enough


def test_split_reads_abstain():
    assert vote_number(["7", "1", "7", "1"]) is None  # 50% share < 60% -> abstain


def test_majority_with_outlier_commits():
    assert vote_number(["22", "22", "22", "2"]) == 22


def test_empty_abstains():
    assert vote_number([]) is None
