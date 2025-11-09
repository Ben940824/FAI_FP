"""
logging_teacher_agent.py
------------------------
A wrapper that

  • delegates every decision to *teacher_ai()*  (baseline .so)
  • logs **ONLY the teacher’s own actions & state** into JSON-L file
"""

import uuid, json
from pathlib import Path
from typing import Any, Tuple
from agents.utils.monte_carlo_utils import mc_equity

import torch    # 只是示範，可刪
from game.players import BasePokerPlayer
from baseline4 import setup_ai as teacher_ai   # ← 換成你要學的 baseline

class LoggingTeacherMCAgent(BasePokerPlayer):
    def __init__(self, log_dir: str = "logs"):
        super().__init__()
        self.teacher = teacher_ai()           # baseline 內部 AI
        self.session_id = str(uuid.uuid4())[:8]
        self.buffer = []
        self.log_path = Path(log_dir) / f"{self.session_id}.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)

    # ---------------- delegate DECISION ----------------

    def declare_action(self, valid_actions, hole_card, round_state):
        action, amount = self.teacher.declare_action(valid_actions, hole_card, round_state)
        print("\n\n\nAAA")
        print(hole_card, round_state["community_card"])
        equity = mc_equity(hole_card, round_state["community_card"], n=600)
        print("\n\nPat")

        self.buffer.append({
            "type": "declare_action",
            "round_count": round_state["round_count"],
            "hole_card": hole_card,
            "equity": equity,                 # ← 新增
            "valid_actions": valid_actions,
            "round_state": round_state,
            "chosen": [action, amount],
        })
        return action, amount

    # ---------------- mirror other callbacks -----------
    def receive_game_start_message(self, game_info):
        self.teacher.receive_game_start_message(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.teacher.receive_round_start_message(round_count, hole_card, seats)

    def receive_street_start_message(self, street, round_state):
        self.teacher.receive_street_start_message(street, round_state)

    def receive_game_update_message(self, new_action, round_state):
        self.teacher.receive_game_update_message(new_action, round_state)


    def receive_round_result_message(self, winners, hand_info, round_state):
        """
        1. 把老師自己的行為 buffer 先 flush
        2. 另外加一條 'round_result' 事件，留 stack / winners
        """
        # ----- flush declare_action buffer -----
        with self.log_path.open("a", encoding="utf-8") as fp:
            for evt in self.buffer:
                fp.write(json.dumps(evt, default=str) + "\n")
            # ----- 新增 round_result 行 -----
            fp.write(json.dumps({
                "type": "round_result",
                "winners": winners,          # list of {uuid, name, stack, state}
                "seats": round_state["seats"],  # 全桌最終 stack；想節省可只留 uuid+stack
                "round_count": round_state["round_count"],
            }, default=str) + "\n")
        self.buffer.clear()

        # 保持 baseline 內部邏輯
        self.teacher.receive_round_result_message(winners, hand_info, round_state)

# framework hook
def setup_ai():
    return LoggingTeacherMCAgent()