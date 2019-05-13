from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pommerman.graphics import PommeViewer, ResourceManager
import pommerman.constants as constants
import time, math
import pyglet
import numpy as np
import cv2
FILE_PREFIX = 'openaigym'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'

class PommeVideoRecorder(PommeViewer):
    def __init__(self,
                 display=None,
                 board_size=11,
                 agents=[],
                 partially_observable=False,
                 agent_view_size=None,
                 game_type=None):
        super().__init__()
        self.display = pyglet.canvas.Display(display)
        board_height = constants.TILE_SIZE * board_size
        height = math.ceil(board_height + (constants.BORDER_SIZE * 2) +
                           (constants.MARGIN_SIZE * 3))
        width = math.ceil(board_height + board_height / 4 +
                          (constants.BORDER_SIZE * 2) + constants.MARGIN_SIZE)

        self._height = height
        self._width = width
        self.window = pyglet.window.Window(
            width=width, height=height, display=None)
        self.window.set_caption('Pommerman')
        self.isopen = True
        self._board_size = board_size
        self._resource_manager = ResourceManager(game_type)
        self._tile_size = constants.TILE_SIZE
        self._agent_tile_size = (board_height / 4) / board_size
        self._agent_count = len(agents)
        self._agents = agents
        self._game_type = game_type
        self._is_partially_observable = partially_observable
        self._agent_view_size = agent_view_size


    def get_frame(self):
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self._height, self._width, 4)
        return np.array(arr[::-1,:,0:3])

class Monitor(MultiAgentEnv):
    def __init__(self, env):
        self.env = env
        self.width = 738
        self.height = 620
        self.video_after_episodes = 100
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.start_recording = False
        self.video_recorded = False
        self.episode = 0
        self._viewer = PommeVideoRecorder(
                display=str.encode(":" + str("99")),
                board_size=self.env.env._board_size,
                agents=self.env.env._agents,
                partially_observable=self.env.env._is_partially_observable,
                agent_view_size=self.env.env._agent_view_size,
                game_type=self.env.env._game_type)

    def render(self, mode=None):
        self._viewer.set_board(self.env.env._board)
        self._viewer.set_agents(self.env.env._agents)
        self._viewer.set_step(self.env.env._step_count)
        self._viewer.set_bombs(self.env.env._bombs)
        self._viewer.render()
        frame = self._viewer.get_frame()
        self.video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.start_recording = True
        return frame

    def set_phase(self, phase):
        self.env.set_phase(phase)

    def step(self, action):
        obs = self.env.step(action)
        if self.episode % self.video_after_episodes == 0:
            self.render()
        return obs

    def reset(self):
        self.episode = self.episode + 1
        if self.start_recording == True:
            self.start_recording = False
            self.video.release()
        if self.episode % self.video_after_episodes == 0:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            self.video = cv2.VideoWriter('./pommer_ep_' + str(self.episode) + '.avi', fourcc, float(1.5), (self.width, self.height))
        return self.env.reset()
