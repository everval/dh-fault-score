"""
A physically informed changepoint score for fault prioritisation in
district heating.

    decision_scores  the score of one channel, for changepoints from any detector
    detect_beast     BEAST changepoints as in the paper
    score_window     detect, score and date the changes in one window
    rank_windows     the two-tier ranking of windows
"""

from .beast import detect_beast
from .ranking import rank_windows
from .score import UNFAVOURABLE, decision_scores
from .window import WindowResult, score_window

__version__ = "0.1.0"
__all__ = ["decision_scores", "detect_beast", "score_window", "rank_windows", "WindowResult", "UNFAVOURABLE"]
