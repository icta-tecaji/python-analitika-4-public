import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if (
        snake_head[0] >= 500
        or snake_head[0] < 0
        or snake_head[1] >= 500
        or snake_head[1] < 0
    ):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnekEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(SnekEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        # vvvv    HERE HERE HERE    vvvv
        self.render_training = False
        # ^^^^    HERE HERE HERE    ^^^^

    def step(self, action):
        # vvvv    HERE HERE HERE    vvvv
        self.render() if self.render_training else None  # If i want to see training or not

        # Change the head position based on the button direction
        if action == 1:  # move RIGHT
            self.snake_head[0] += 10
        elif action == 0:  # move LEFT
            self.snake_head[0] -= 10
        elif action == 2:  # move DOWN
            self.snake_head[1] += 10
        elif action == 3:  # move UP
            self.snake_head[1] -= 10

        # Calculating reward
        self.reward = 0

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if (
            collision_with_boundaries(self.snake_head) == 1
            or collision_with_self(self.snake_position) == 1
        ):
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype="uint8")
            cv2.putText(
                self.img,
                "Your Score is {}".format(self.score),
                (140, 250),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self.done = True  # if sneak hits itself it dies

        self.reward += len(self.snake_position)

        info = {}

        observation = self.constructObservation()

        return (
            observation,
            self.reward,
            self.done,
            self.truncated,
            info,
        )  # observation, reward, terminated, truncated, info
        # ^^^^    HERE HERE HERE    ^^^^

    def reset(self, seed=None, options=None):
        self.img = np.zeros((500, 500, 3), dtype="uint8")

        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]

        self.score = 0
        self.snake_head = self.snake_position[0].copy()

        self.done = False
        self.truncated = False

        observation = self.constructObservation()

        return observation, {}  # observation, info

    def render(self, mode="human"):
        cv2.imshow("Snake", self.img)  # Show the current frame
        cv2.waitKey(
            50
        )  # wait 50ms. Otherwise a person can't see anything because it is too fast

        # Create the next frame to be rendered
        self.img = np.zeros((500, 500, 3), dtype="uint8")  # create empty board

        # Display Apple
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
        )
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                3,
            )

    def close(self):
        pass

    def constructObservation(self):
        # Used to create a current observation
        """Observation holds: (all 1 or 0 values)
        * is apple to the left of head
        * is apple to the right of head
        * is apple above head
        * is apple below head
        * is wall or tail to the left of head
        * is wall or tail to the right of head
        * is wall or tail above head
        * is wall or tail below head
        """
        observation = []

        if self.apple_position[0] < self.snake_head[0]:  # to the left
            observation.extend([1, 0])
        else:
            observation.extend([0, 1])

        if self.apple_position[1] < self.snake_head[1]:  # above
            observation.extend([1, 0])
        else:
            observation.extend([0, 1])

        positions = self.snake_position.copy()
        head = positions[0].copy()

        # Check left
        head_left = [head[0] - 10, head[1]]
        positions[0] = head_left
        o = collision_with_boundaries(head_left) or collision_with_self(positions)
        observation.append(o)

        # Check right
        head_right = [head[0] + 10, head[1]]
        positions[0] = head_right
        o = collision_with_boundaries(head_right) or collision_with_self(positions)
        observation.append(o)

        # Check above
        head_above = [head[0], head[1] - 10]
        positions[0] = head_above
        o = collision_with_boundaries(head_above) or collision_with_self(positions)
        observation.append(o)

        # Check below
        head_below = [head[0], head[1] + 10]
        positions[0] = head_below
        o = collision_with_boundaries(head_below) or collision_with_self(positions)
        observation.append(o)

        observation = np.array(observation, dtype=np.float32)
        return observation
