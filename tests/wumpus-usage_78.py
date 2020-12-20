import os, sys
sys.path.insert(1, os.path.abspath("modules/"))

import wumpus as wws

from typing import Iterable
from wumpus import Hunter
from modules.hybrid_agent_model_78 import HuntWumpusHybridAgent

class OnlineDFSPlayer(wws.UninformedPlayer):
    """Uninformed player demostrating the implementation of an online search"""
    
    def _say(self, text: str):
        print(self.name + ' says: ' + text)

    def start_episode(self):
        """Method called when the game starts"""

        self.total_reward = 0
        self.hybrid_wumpus_agent = HuntWumpusHybridAgent()
        
    def end_episode(self, outcome: int, alive: bool, success: bool):
        """Method called the when an episode is completed."""

    def play(self, turn: int, percept: wws.Hunter.Percept, actions: Iterable[wws.Hunter.Actions]) -> wws.Hunter.Actions:
        """Return the action the agent should perform at the requested turn"""
        return self.hybrid_wumpus_agent.get_next_action_from(percept=percept)

    def feedback(self, action: wws.Hunter.Actions, reward: int, percept: wws.Hunter.Percept):
        """Receive in input the reward of the last action and the resulting state. 
           The function is called right after the execution of the action."""

        self.total_reward += reward

    

