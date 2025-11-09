"""
decision_agent.py
-----------------
A *battle‑ready* agent that loads the TorchScript model produced
by `train_model.py` and chooses between **fold / call / raise**
in real time (≪ 5 s).

• Inherits `BasePokerPlayer` – so it can be registered exactly
  like other agents:
      from agents.decision_agent import setup_ai as decision_ai
      config.register_player(name="p1", algorithm=decision_ai())

• Model must be a TorchScript file with the same output head:
      logits[0] = fold
      logits[1] = call
      logits[2] = raise
"""
from pathlib import Path
from typing import List, Union

import torch
from game.players import BasePokerPlayer

# --------------------------------------------------------------------- #
#  Utils  – keep in sync with `train_model.py`
# --------------------------------------------------------------------- #
SUITS = "CDHS"                       # index 0~3
RANKS = "23456789TJQKA"              # index 0~12
CARD2IDX = {r + s: 13 * si + ri
            for si, s in enumerate(SUITS)
            for ri, r in enumerate(RANKS)}
PAD_IDX = 52                         # padding index in embedding


def encode_card(card: str) -> int:
    """'C9' → integer ∈[0,52], 52 denotes PAD/None."""
    return CARD2IDX.get(card, PAD_IDX)


def pad(seq: List[int], size: int, val: int = PAD_IDX) -> List[int]:
    return (seq + [val] * size)[:size]


def encode_numeric(round_state: dict, to_call: int) -> torch.Tensor:
    pot = round_state["pot"]["main"]["amount"] / 2000           # ↯粗略 normalise
    sb = round_state["small_blind_amount"] / 10
    street_id = ["preflop", "flop", "turn", "river",
                 "showdown"].index(round_state["street"]) / 4
    to_call_n = to_call / 500
    return torch.tensor([pot, sb, street_id, to_call_n],
                        dtype=torch.float32).unsqueeze(0)       # shape [1,4]

# --------------------------------------------------------------------- #
#  Main Agent
# --------------------------------------------------------------------- #
class DecisionAgent(BasePokerPlayer):
    def __init__(self,
                 model_path: Union[str, Path] = "poker_policy.pt",
                 device: str = "cpu"):
                 
        super().__init__()
        self.device = torch.device(device)
        self.net = torch.jit.load(str(model_path), map_location=self.device)
        self.net.eval()                     # just in case
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0 / 3

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    # ---------------- mandatory callback ----------------
    def declare_action(self, valid_actions, hole_card, round_state):
        """
        1. Encode current observation exactly like during training
        2. Run model → logits → choose the best VALID action
        3. Map label back to (action, amount) tuple required by engine
        """
        # --- encode observation ---
        hole_idx = torch.tensor(
            [encode_card(c) for c in hole_card],
            dtype=torch.long).unsqueeze(0).to(self.device)      # [1,2]

        board_idx = torch.tensor(
            pad([encode_card(c) for c in round_state["community_card"]], 5),
            dtype=torch.long).unsqueeze(0).to(self.device)      # [1,5]

        to_call_amt = next(a["amount"] for a in valid_actions
                           if a["action"] == "call")
        num_feat = encode_numeric(round_state, to_call_amt).to(self.device)

        with torch.no_grad():
            logits = self.net(hole_idx, board_idx, num_feat)
            probs = torch.softmax(logits, dim=-1).squeeze(0)    # [3]

        # ----------------------------------------------------------------
        # Mask INVALID logits (e.g., 'raise' absent) by setting -inf
        # ----------------------------------------------------------------
        action_map = {"fold": 0, "call": 1, "raise": 2}
        mask = torch.zeros_like(probs) - float("inf")
        for a in valid_actions:
            mask[action_map[a["action"]]] = 0
        masked_logits = probs + mask
        label = int(torch.argmax(masked_logits).item())

        # ---------------- map back to engine action ------------------
        if label == action_map["fold"]:
            return "fold", 0
        if label == action_map["call"]:
            amt = next(a["amount"] for a in valid_actions
                       if a["action"] == "call")
            return "call", amt

        # ---- raise ----
        raise_info = next(a for a in valid_actions if a["action"] == "raise")
        min_r, max_r = raise_info["amount"]["min"], raise_info["amount"]["max"]
        # simplest strategy: raise min (can tweak later)
        return "raise", min_r

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

# factory the framework expects
def setup_ai():
    return DecisionAgent()