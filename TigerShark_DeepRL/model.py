# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment
from stable_baselines3.common import env_checker
# Import the DQN algorithm
from stable_baselines3 import DQN
# Import WebGame environment from client
from client import WebGame

# Gather environment from client
env = WebGame()

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "best_model_{}".format(self.n_calls))
            self.model.save(model_path)
        
        return True

CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# Create the DQN model
model = DQN("CnnPolicy", env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=100000, learning_starts=1000)

# Kick off training
model.learn(total_timesteps=44000, callback=callback)
