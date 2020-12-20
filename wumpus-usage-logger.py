import logging
import os, sys
sys.path.insert(1, os.path.abspath("modules/"))

import wumpus as wws

from logging.handlers import TimedRotatingFileHandler
from typing import Iterable
from wumpus import Hunter
from modules.hybrid_agent_model_logger import HuntWumpusHybridAgent

#logging.disable()

# Logs are saved in logs/hunt_the_wumpus.log
logger = logging.getLogger("HuntWumpusLogger")
logger.setLevel(logging.DEBUG)
logger.propagate = False
    
time_handler = TimedRotatingFileHandler(r"logs/hunt_the_wumpus.log",
                                        when="s",
                                        interval=30,
                                        backupCount=5)
time_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(time_handler)

class OnlineDFSPlayer(wws.UninformedPlayer):
    """Uninformed player demostrating the implementation of an online search"""
    
    def _say(self, text: str):
        print(self.name + ' says: ' + text)

    def start_episode(self):
        """Method called when the game starts"""

        self.total_reward = 0
        logger.info("Initializing the world environment of the game")

        self.hybrid_wumpus_agent = HuntWumpusHybridAgent()
        
    def end_episode(self, outcome: int, alive: bool, success: bool):
        """Method called the when an episode is completed."""

        logger.info(f"Successfully Climbed out, the total reward is {self.total_reward}" if alive 
                    else f"You DIED, the total reward is {self.total_reward}")

    def play(self, turn: int, percept: wws.Hunter.Percept, actions: Iterable[wws.Hunter.Actions]) -> wws.Hunter.Actions:
        """Return the action the agent should perform at the requested turn"""

        logger.info("")
        logger.info("*" * 3 + " " * 2 + f"Turn [{turn}]: Current percept {percept}\n")

        return self.hybrid_wumpus_agent.get_next_action_from(percept=percept)

    def feedback(self, action: wws.Hunter.Actions, reward: int, percept: wws.Hunter.Percept):
        """Receive in input the reward of the last action and the resulting state. 
           The function is called right after the execution of the action."""

        self.total_reward += reward

        logger.info("*" * 15 + " " * 2 + f"Executing [{action}] with reward ({reward}) -> total reward: {self.total_reward}")

