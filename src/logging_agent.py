from pathlib import Path
import json
import uuid
import random

from game.players import BasePokerPlayer


class LoggingAgent(BasePokerPlayer):
    """
    A lightweight agent that:
      • Always performs the 'call' action (baseline behaviour).
      • Logs every relevant event it can observe into a JSON‑Lines file.
        Each game session gets its own file named <8‑char‑session_id>.jsonl
        under the folder specified by ``log_dir`` (default: ``logs``).

    The JSON object schema for each line:
        {
          "type":                # event tag, e.g. "declare_action", "game_update", ...
          ...                    # payload fields (see below for each tag)
        }

    Logged events
    -------------
    * game_start        → {"type": "game_start",        "game_info": {...}}
    * round_start       → {"type": "round_start",       "round_count": int, "hole_card": [...], "seats": [...]}
    * street_start      → {"type": "street_start",      "street": str,      "round_state": {...}}
    * declare_action    → {"type": "declare_action",    "hole_card": [...], "valid_actions": [...], "round_state": {...}, "chosen": [action, amount]}
    * game_update       → {"type": "game_update",       "action": {...},    "round_state": {...}}
    * round_result      → {"type": "round_result",      "winners": [...],   "hand_info": [...],    "round_state": {...}}
    """
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.session_id = str(uuid.uuid4())[:8]
        self.buffer = []
        self.log_path = Path(log_dir) / f"{self.session_id}.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)


    # ---------- internal helper ----------
    def _push(self, event: dict) -> None:
        """Append *event* to the in‑memory buffer."""
        self.buffer.append(event)



    def declare_action(self, valid_actions, hole_card, round_state):
        """
        永遠選擇 'call'；並完整紀錄觀測。
        """
        call_info = next(a for a in valid_actions if a["action"] == "call")
        self.buffer.append({
            "type": "declare_action",
            "hole_card": hole_card,
            "valid_actions": valid_actions,
            "round_state": round_state,
            "chosen": [call_info["action"], call_info["amount"]],
        })
        return call_info["action"], call_info["amount"]

    def receive_game_start_message(self, game_info):
        self._push({"type": "game_start", "game_info": game_info})

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._push({
            "type": "round_start",
            "round_count": round_count,
            "hole_card": hole_card,
            "seats": seats,
        })

    def receive_street_start_message(self, street, round_state):
        self._push({
            "type": "street_start",
            "street": street,
            "round_state": round_state,
        })

    def receive_game_update_message(self, action, round_state):
        self._push({
            "type": "game_update",
            "action": action,
            "round_state": round_state,
        })

    def receive_round_result_message(self, winners, hand_info, round_state):
        """
        When a hand finishes, dump the buffered events to disk and clear the buffer.
        This guarantees we only hit the filesystem once per hand (keeps I/O cheap).
        """
        self._push({
            "type": "round_result",
            "winners": winners,
            "hand_info": hand_info,
            "round_state": round_state,
        })

        # ---- flush to file ----
        with self.log_path.open("a", encoding="utf-8") as fp:
            for evt in self.buffer:
                # default=str prevents serialization errors for UUID / numpy types etc.
                fp.write(json.dumps(evt, default=str) + "\n")

        self.buffer.clear()

# -------- factory required by the framework --------
def setup_ai():
    return LoggingAgent()