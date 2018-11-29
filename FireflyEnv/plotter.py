import gym
import torch
from .env_utils import *
from .env_variables import *
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt

AGENT_SIZE, GOAL_SIZE = 10, 10

GREEN = (0.6274, 0.8313, 0.4078)
ORANGE = (0.9882,  0.4313,  0.3176)
GREY = (0.5, 0.5, 0.5)

class Plotter():
    def __init__(self):
        self.trajectory = None
        return

    def plot(self, x):
        if self.trajectory is None:
            self.trajectory = x.view(1, -1)
        else:
            self.trajectory = torch.cat([self.trajectory, x.view(1, -1)])

    def show(self):
        x = self.trajectory[:, 0].data.numpy()
        y = self.trajectory[:, 1].data.numpy()
        plt.plot(x, y)
        plt.show()

def translate(point, center, scale):
    return (point - center) * scale

class Render(gym.Env):
    def __init__(self):
        self.viewer = None

    def render(self, goal_pos, x, P):
        goal_pos = goal_pos.detach().numpy()
        x = x.detach().numpy()
        P = P.detach().numpy()

        screen_width, screen_height = 500, 500
        xyBox = WORLD_SIZE
        world_width = 2 * xyBox
        scale = screen_width/world_width

        center = translate(torch.zeros(2),  -xyBox, scale)
        goal_pos = translate(goal_pos, -xyBox, scale)

        agent_size = AGENT_SIZE
        goal_size = GOAL_SIZE

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal = rendering.make_circle(goal_size, res=30)
            goal_ring = rendering.make_circle(GOAL_RADIUS * scale, res=30, filled=False)

            goal.set_color(*GREEN) #green
            goal_ring.set_color(*GREEN) #green

            self.goal_motion = rendering.Transform(translation=center)
            goal.add_attr(self.goal_motion)
            goal_ring.add_attr(self.goal_motion)

            agent = rendering.make_circle(agent_size, res=30)
            head = rendering.make_polygon([(0,5),(0,-5),(5,0)])

            agent.set_color(*ORANGE) #orange
            head.set_color(*GREY)
            head.add_attr(rendering.Transform(translation=(10,0))) #offset

            self.agent_motion = rendering.Transform(translation=(0,0))
            self.head_motion = rendering.Transform() # for rotation (turning)

            agent.add_attr(self.agent_motion)
            head.add_attr(self.head_motion)
            head.add_attr(self.agent_motion)

            self.viewer.add_geom(goal)
            self.viewer.add_geom(goal_ring)
            self.viewer.add_geom(agent)
            self.viewer.add_geom(head)
            self.viewer.add_geom(agent) # dummy addition to replace with cov

        self.goal_motion.set_translation(goal_pos[0], goal_pos[1])
        position, ang = x[:2], x[2]
        move = translate(position, -xyBox, scale)

        self.agent_motion.set_translation(move[0], move[1])
        self.head_motion.set_rotation(ang)

        pts = np.vstack(ellipse(np.zeros(2), P[:2,:2], conf_int=5.991*scale**2)).T
        pts = [tuple(v) for v in pts]

        cov = rendering.make_polygon(pts, False)
        cov.set_color(*ORANGE)
        cov.add_attr(rendering.Transform(translation=(move[0], move[1])))
        self.viewer.geoms[-1] = cov

        return self.viewer.render(return_rgb_array=False)
