"""
agent.py

Final Agent for competition.
Original file name: MC_agent.py
"""

from dataclasses import dataclass
import random
from game.players import BasePokerPlayer
from game.engine.card import Card
from utils.monte_carlo_utils import mc_equity

# ---------- Monte‑Carlo player ---------- #
class MonteCarloPlayer(BasePokerPlayer):
    NB_SIMULATIONS = 1000  # number of random roll‑outs per decision


    # ---- helpers to convert 'SA' / ('S',14) / Card into engine.Card ----
    _SUIT_MAP = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    _RANK_MAP = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}

    def declare_action(self, valid_actions, hole_card, round_state):
        print("\n My turn NOWWWWW!\n")
        print(round_state)
        me = round_state["next_player"]
        opponent = 0 if me else 1
        my_money = round_state["seats"][me]["stack"]
        opponent_money = round_state["seats"][opponent]["stack"]

        print("price", my_money, opponent_money)
        print("hole_card", hole_card)
        community_card = round_state["community_card"]
        print("Community_card", community_card)

        small_blind = round_state["small_blind_amount"]
        big_blind = small_blind * 2
        is_big_blind = 1 if round_state["big_blind_pos"] == me else 0

        # 穩住的小賤招！
        total_rounds = 20
        rounds_left = total_rounds - round_state["round_count"]
        blind_exposure = rounds_left * small_blind * 1.53
        pot = round_state["pot"]["main"]["amount"] if "pot" in round_state else 0
        lead = my_money - (opponent_money + pot) # 假設獎池的錢也被對手贏走

        if lead > blind_exposure:
            print("Gonna win!")
            return fold_act["action"], fold_act["amount"]

        # Example `valid_actions` layout:
        # [{"action":"fold","amount":0},
        #  {"action":"call","amount":X},
        #  {"action":"raise","amount":{"min":Y,"max":Z}}]
        fold_act, call_act, raise_act = valid_actions

        rank_threshold_1 = 10
        rand_threshold_2 = 12
        a, b = [int(card[1]) if card[1].isdigit() else MonteCarloPlayer._RANK_MAP[card[1]] for card in hole_card]
        print(a, b)
        if a == b :
            print("SAME! raise 50 or min")
            target_amount = max(50, raise_act["amount"]["min"])
            if target_amount >= 150: 
                print("BUT: too high, decide later")
            else:
                return raise_act["action"], target_amount
        if a >= rand_threshold_2 and b >= rand_threshold_2:
            print("BOTH Pass 2, raise 50 or min")
            target_amount = max(30, raise_act["amount"]["min"])
            if target_amount >= 150: 
                print("BUT: too high, decide later")
            else:
                return raise_act["action"], target_amount
        if a >= rand_threshold_2 or b >= rand_threshold_2:
            print("One Pass 2, raise 20 or min")
            target_amount = max(20, raise_act["amount"]["min"])
            if target_amount >= 70: 
                print("BUT: too high, decide later")
            else:            
                return raise_act["action"], target_amount
        elif not (a + b >= rank_threshold_1):
            print("Hole cards too weak, folding:", a, b)
            return fold_act["action"], fold_act["amount"]


        # Very simple heuristic – feel free to tweak later
        win_rate = self._estimate_win_rate(hole_card, community_card)
        print("Win_rate", win_rate)

        if win_rate >= 0.85 and raise_act["amount"]["max"] > 0:
            target_amount = max(300, raise_act["amount"]["min"])
            print(f"AAA raise {target_amount}")
            return raise_act["action"], target_amount
        
        if win_rate >= 0.70 and raise_act["amount"]["max"] > 0:
            target_amount = max(100, raise_act["amount"]["min"])
            print(f"AAA raise {target_amount}")
            return raise_act["action"], target_amount

        call_amount = call_act["amount"]
        if win_rate > 0.6 and call_amount <= 100:
            print("Moderate win_rate > 0.65, calling up to 100: AAA", call_amount)
            return call_act["action"], call_amount
        elif win_rate > 0.4 and call_amount <= 50:
            print("Mild win_rate > 0.4, calling up to 50: AAA", call_amount)
            return call_act["action"], call_amount
        elif win_rate >= 0.25 and call_amount <= 10:
            print("Low win_rate > 0.25, calling up tp 10: AAA", call_amount)
            return call_act["action"], call_amount
        elif win_rate >= 0.1 and call_amount <= 5:
            print("Mini win_rate > 0.1, calling up tp 5: AAA", call_amount)
            return call_act["action"], call_amount
        elif call_amount == 0:
            return call_act["action"], call_amount


        print("AAA fold")
        return fold_act["action"], fold_act["amount"]

    # ------------------------------------------------ #
    def _estimate_win_rate(self, hole_card, community_card):
        """
        Estimate win probability using the shared mc_equity() helper.
        Returns a float in [0,1].
        """
        hole = hole_card
        board = community_card
        return mc_equity(hole, board)

    # Callbacks (optional) ----------------------------------------------- #
    def receive_game_start_message(self, game_info):          pass
    def receive_round_start_message(self, round_count,
                                    hole_card, seats):        pass
    def receive_street_start_message(self, street, state):    pass
    def receive_game_update_message(self, action, state):     pass
    def receive_round_result_message(self, winners, hand_info, state):
        print("\n=== Round Result ===")
        print("Round Winner:", winners)
        print("Hand Info:", hand_info)
        print("State:")
        print(state)
        print("====================\n")


def setup_ai():
    return MonteCarloPlayer()