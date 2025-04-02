from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
import random


class SlipperyCliffWalkingEnv(CliffWalkingEnv):
    def __init__(self, slip_chance=0.2):
        super(SlipperyCliffWalkingEnv, self).__init__()
        self.slip_chance = slip_chance
        
    def step(self, action):
        # Con una certa probabilità, esegui un'azione casuale invece di quella scelta
        if random.random() < self.slip_chance:
            action = random.randint(0, 3)
        
        return super().step(action)
    

make = lambda **kwargs: SlipperyCliffWalkingEnv(**kwargs)