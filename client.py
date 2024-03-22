# PIL for screen capture
from PIL import ImageGrab
# Send keyinput commands
import pydirectinput
# Process frames
import cv2
# Transformational framework
import numpy as np
# OCR for game over extraction
import pytesseract
# Use time for pause and variable key presses
import time
# Environmental components
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
# Import the Rainbow DQN algorithm
from stable_baselines3 import DQN
import threading
from matplotlib import pyplot as plt
# OS
import os

class WebGame(gym.Env):
    # Setup environment action and observation shapes
    def __init__(self):
        super().__init__()
        # Setup space
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(6)
        # Area of the game where we want to extract
        self.game_location = {"top":32, "left":1, "width":670, "height":360}
        self.done_location = {"top":405, "left":630, "width":660, "height":70}
        self.score_location = {"top":57, "left":439, "width":115, "height":17}

    # Step, an action called in order to move the game
    def step(self, action):
        # Action key - 0 = no_op, 1 = right, 2 = jump
        action_map = {
            0: [],            # No operation
            1: ['d'],         # Move right
            2: ['space'],     # Jump
            3: ['d', 'space'],# Move right and jump simultaneously
            4: ['z'],         # Spin
            5: ['z', 'd']     # Move while spinning
        }
        duration_map = {
            0: 0.05,  # Duration for no_op
            1: 0.05,  # Duration for d (right movement)
            2: 0.1,   # Duration for space (jump)
            3: 0.1,    # Duration for d and space (simultaneous)
            4: 0.05,
            5: 0.1
        }

        if action != 0:
            for key in action_map[action]:
                pydirectinput.keyDown(key)    # Press the key

            time.sleep(duration_map[action]) # Keep the keys pressed for a certain duration

            for key in action_map[action]:
                pydirectinput.keyUp(key)      # Release the key

        # Checking whether the game is done
        done, _ = self.get_done()
        # Get the new observation
        new_observation = self.get_observation()
        # Give reward for staying alive
        reward = 1
        # Terminated and truncated signals
        terminated = bool(done)  # Convert done to boolean
        truncated = False         # Assuming no truncation
        # Info dictionary
        info = {}

        return new_observation, reward, terminated, truncated, info


    # Visualize the game
    def render(self):
        cv2.imshow("Game", np.array(ImageGrab.grab(bbox=(
            self.game_location["left"],
            self.game_location["top"],
            self.game_location["left"] + self.game_location["width"],
            self.game_location["top"] + self.game_location["height"]
        ))))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()
    # Restart the game
    def reset(self, **kwargs):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        time.sleep(4)
        pydirectinput.press("space")
        time.sleep(7)
        pydirectinput.keyDown("d")
        time.sleep(0.5)
        pydirectinput.keyUp("d")
        observation = self.get_observation()
        info = {}  # You can populate this info dictionary if needed
        return observation, info


    # Closes observations
    def close(self):
        cv2.destroyAllWindows()
    # Get part of observation that we want
    def get_observation(self):
        # Get screen capture of game
        raw = np.array(ImageGrab.grab(bbox=(
            self.game_location["left"],
            self.game_location["top"],
            self.game_location["left"] + self.game_location["width"],
            self.game_location["top"] + self.game_location["height"]
        )))[:,:,:3]
        # Greyscale
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, (100,83))
        # Add channels first
        channel = np.reshape(resized, (1,83,100))
        return channel
    # Get game over screen
    def get_done(self):
        # Get done screen
        black_threshold = 20
        screenshot = self.get_observation()
        is_black = np.all(screenshot < black_threshold)

        return is_black, screenshot

    def get_score(self):
        score_cap = np.array(ImageGrab.grab(bbox=(
            self.score_location["left"],
            self.score_location["top"],
            self.score_location["left"] + self.score_location["width"],
            self.score_location["top"] + self.score_location["height"]
        )))[:,:,:3]
        score = pytesseract.image_to_string(score_cap, config=r'--oem 3 --psm 6')
        return score, score_cap

env = WebGame()

