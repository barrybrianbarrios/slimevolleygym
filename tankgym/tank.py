"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gym
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import cv2 # installed with gym anyways
from collections import deque

# np.set_printoptions(threshold=50, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER = False

MAX_MOVE = 5.0
BACKGROUND_COLOR = (255, 255, 255)
PLAYER_RAD = 3.0
COOLDOWN = 30
MAX_SPEED = 10.0
MAX_SPEED_SQUARED = MAX_SPEED * MAX_SPEED
BULLET_RADIUS = 1.0

AGENT_LEFT_COLOR = (255, 0, 0)
AGENT_RIGHT_COLOR = (0,232,255)

FENCE_COLOR = (255, 255, 255)
COIN_COLOR = FENCE_COLOR
GROUND_COLOR = (116, 114, 117)

ACTION_SPACE = spaces.Box(low = np.array([-5 * math.pi,-MAX_MOVE,-5 * math.pi,-1]), high = np.array([5 * math.pi,MAX_MOVE,5 * math.pi,1]), dtype = np.float32)

REF_W = 24*2
REF_H = REF_W
BALL_SPEED = 40
MIN_BALL_SPEED = BALL_SPEED - MAX_SPEED
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION

MAX_DIST = math.sqrt(2) * REF_W

MAX_BULLETS = math.ceil((BULLET_RADIUS * 2 + MAX_DIST)/(MIN_BALL_SPEED * TIMESTEP * COOLDOWN))
BULLET_SPREAD = 0

MAXLIVES = 3 # game ends when one agent loses this many games

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = False
PIXEL_SCALE = 4 # first render at multiple of Pixel Obs resolution, then downscale. Looks better.

PIXEL_WIDTH = 84*2*1
PIXEL_HEIGHT = 84*1

# by default, don't load rendering (since we want to use it in headless cloud machines)
rendering = None
def checkRendering():
  global rendering
  if rendering is None:
    from gym.envs.classic_control import rendering as rendering

def setPixelObsMode():
  """
  used for experimental pixel-observation mode
  note: new dim's chosen to be PIXEL_SCALE (2x) as Pixel Obs dims (will be downsampled)

  also, both agent colors are identical, to potentially facilitate multiagent
  """
  global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR, PIXEL_MODE
  PIXEL_MODE = True
  WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
  WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
  FACTOR = WINDOW_WIDTH / REF_W

def upsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH * PIXEL_SCALE, PIXEL_HEIGHT * PIXEL_SCALE), interpolation=cv2.INTER_NEAREST)
def downsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH, PIXEL_HEIGHT), interpolation=cv2.INTER_AREA)

# conversion from space to pixels (allows us to render to diff resolutions)
def toX(x):
  return (x+REF_W/2)*FACTOR
def toP(x):
  return (x)*FACTOR
def toY(y):
  return y*FACTOR

def _add_attrs(geom, color):
  """ help scale the colors from 0-255 to 0.0-1.0 (pyglet renderer) """
  r = color[0]
  g = color[1]
  b = color[2]
  geom.set_color(r/255., g/255., b/255.)

def create_canvas(canvas, c):
  if PIXEL_MODE:
    result = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
      result[:, :, channel] *= c[channel]
    return result
  else:
    rect(canvas, 0, 0, WINDOW_WIDTH, -WINDOW_HEIGHT, color=BACKGROUND_COLOR)
    return canvas

def rect(canvas, x, y, width, height, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    canvas = cv2.rectangle(canvas, (round(x), round(WINDOW_HEIGHT-y)),
      (round(x+width), round(WINDOW_HEIGHT-y+height)),
      color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas
  else:
    box = rendering.make_polygon([(0,0), (0,-height), (width, -height), (width,0)])
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(box, color)
    box.add_attr(trans)
    canvas.add_onetime(box)
    return canvas

def circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    return cv2.circle(canvas, (round(x), round(WINDOW_HEIGHT-y)), round(r),
      color, thickness=-1, lineType=cv2.LINE_AA)
  else:
    geom = rendering.make_circle(r, res=40)
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(geom, color)
    geom.add_attr(trans)
    canvas.add_onetime(geom)
    return canvas

class Bullet:
  """ used for the ball, and also for the round stub above the fence """
  def __init__(self, x, y, vx, vy, r, c):
    self.x = x
    self.y = y
    self.prev_x = self.x
    self.prev_y = self.y
    self.vx = vx
    self.vy = vy
    self.r = r
    self.c = c
  def display(self, canvas):
    return circle(canvas, toX(self.x), toY(self.y), toP(self.r), color=self.c)
  def move(self):
    self.prev_x = self.x
    self.prev_y = self.y
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  def outOfBounds(self):
    if (self.x<=(-self.r-REF_W/2)):
      return True

    if (self.x >= (REF_W/2+self.r)):
      return True

    if (self.y<=(-self.r)):
      return True
    if (self.y >= (REF_H+self.r)):
      return True
    return False
  def getDist2(self, p): # returns distance squared from p
    dy = p.y - self.y
    dx = p.x - self.x
    return (dx*dx+dy*dy)
  def isColliding(self, p): # returns true if it is colliding w/ p
    r = self.r+p.r
    return (r*r > self.getDist2(p)) # if distance is less than total radius, then colliding.

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self):
    # agent
    self.x = 0
    self.y = 0
    self.vx = 0
    self.vy = 0
    self.life = 0
    self.cooldown = 0
    # opponent
    self.ox = 0
    self.oy = 0
    self.ovx = 0
    self.ovy = 0
    self.olife = 0
    self.ocooldown = 0
    self.myballs = []
    self.oppballs = []

  def convertTheta(self, theta):
    theta = math.fmod(theta, math.pi * 2)
    if theta > math.pi:
        theta -= math.pi * 2
    elif theta < -math.pi:
        theta += math.pi * 2
    return theta / math.pi

  def getObservation(self):
    dy = self.oy - self.y
    dx = self.ox - self.x
    degreeToFace = math.atan2(dy, dx)
    dvy = (self.ovy - self.vy)
    dvx = (self.ovx - self.vx)
    l = math.sqrt(dx * dx + dy * dy)
    L = dx * dvy - dy * dvx
    S = dx * dvx + dy * dvy
    sqs = dvx * dvx + dvy * dvy
    C = -L/sqs if sqs > 0.1 else 0
    t = (dvx * C - dy) / dvy if sqs > 0.1 else 0
    r = abs(L) / math.sqrt(sqs) if sqs > 0.1 else 0
    result = [self.x/REF_W * 2, self.y / REF_H - 0.5, self.vx/MAX_SPEED, self.vy/MAX_SPEED, self.life/MAXLIVES - 0.5, self.cooldown / COOLDOWN - 0.5,
              l / MAX_DIST, L /MAX_DIST / MAX_SPEED, S / MAX_DIST / MAX_SPEED, max(min(t, 2), -2) / 2, max(min(r, 10), -10) / 10 - 0.5, self.olife/MAXLIVES -0.5, self.ocooldown / COOLDOWN - 0.5]
    myballs = []
    for i in range(MAX_BULLETS):
        if len(self.myballs) == 0:
            myballs += [[0, 0, 0, 0, 0, 0]]
        else:
            ball = self.myballs[i % len(self.myballs)]
            dy = ball[1] - self.y
            dx = ball[0] - self.x
            dvy = ball[3] - self.vy
            dvx = ball[2] - self.vx
            theta = math.atan2(dy, dx) - degreeToFace
            l = math.sqrt(dx * dx + dy * dy)
            L = dx * dvy - dy * dvx
            S = dx * dvx + dy * dvy
            sqs = dvx * dvx + dvy * dvy
            C = -L/sqs if sqs > 0.1 else 0
            t = (dvx * C - dy) / dvy if sqs > 0.1 else 0
            r = abs(L) / math.sqrt(sqs) if sqs > 0.1 else 0
            myballs += [[self.convertTheta(theta), l / MAX_DIST, L /MAX_DIST / BALL_SPEED, S / MAX_DIST / BALL_SPEED, max(min(t, 2), -2) / 2, max(min(r, 10), -10) / 10 - 0.5]]
    for ball in myballs:    result += ball
    oppballs = []
    for i in range(MAX_BULLETS):
        if len(self.oppballs) == 0:
            oppballs += [[0, 0, 0, 0, 0, 0]]
        else:
            ball = self.oppballs[i % len(self.oppballs)]
            dy = ball[1] - self.y
            dx = ball[0] - self.x
            dvy = ball[3] - self.vy
            dvx = ball[2] - self.vx
            theta = math.atan2(dy, dx) - degreeToFace
            l = math.sqrt(dx * dx + dy * dy)
            L = dx * dvy - dy * dvx
            S = dx * dvx + dy * dvy
            sqs = dvx * dvx + dvy * dvy
            C = -L/sqs if sqs > 0.1 else 0
            t = (dvx * C - dy) / dvy if sqs > 0.1 else 0
            r = abs(L) / math.sqrt(sqs) if sqs > 0.1 else 0
            oppballs += [[self.convertTheta(theta), l / MAX_DIST, L /MAX_DIST / BALL_SPEED, S / MAX_DIST / BALL_SPEED, max(min(t, 2), -2) / 2, max(min(r, 10), -10) / 10 - 0.5]]
    for ball in oppballs:    result += ball
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) * scaleFactor
    # print(result, np.shape(result))
    return result

class Agent:
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, dir, x, y, c):
    self.dir = dir
    self.x = x
    self.y = y
    self.r = PLAYER_RAD
    self.c = c
    self.vx = 0
    self.vy = 0
    self.state = RelativeState()
    self.life = MAXLIVES
    self.cooldown = COOLDOWN
    self.primedbullet = None
    self.dirpt = 0
  def lives(self):
    return self.life
  def normalizeSpeed(self):
      speed = self.vx**2 + self.vy**2
      if speed >= MAX_SPEED_SQUARED:
          toDivide = speed / MAX_SPEED_SQUARED
          self.vx /= toDivide
          self.vy /= toDivide
  def setAction(self, action):
      # print(action)
      dy = self.state.oy - self.state.y
      dx = self.state.ox - self.state.x
      degreeToFace = math.atan2(dy, dx)
      self.vx += math.cos(action[0]/5 + degreeToFace) * action[1]
      self.vy += math.sin(action[0]/5 + degreeToFace) * action[1]
      self.dirpt = degreeToFace + action[2]/5
      self.normalizeSpeed()
      if self.cooldown == 0 and action[3] > 0:
          self.cooldown = COOLDOWN
          dx = math.cos(self.dirpt)
          dy = math.sin(self.dirpt)
          x = self.x + (BULLET_RADIUS + self.r + .1) * dx
          y = self.y + (BULLET_RADIUS + self.r + .1) * dy
          vx = self.vx + dx * BALL_SPEED
          vy = self.vy + dy * BALL_SPEED
          self.primedbullet = Bullet(x, y, vx, vy, BULLET_RADIUS, self.c)
          # self.vx -= vx
          # self.vy -= vy


  def move(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.cooldown = max(0, self.cooldown - 1)

  def step(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP

  def checkEdges(self):
    if (self.x<=(self.r-REF_W/2)):
      self.vx *= -FRICTION
      self.x = self.r-REF_W/2+NUDGE*TIMESTEP

    if (self.x >= (REF_W/2-self.r)):
      self.vx *= -FRICTION;
      self.x = REF_W/2-self.r-NUDGE*TIMESTEP

    if (self.y<=(self.r)):
      self.vy *= -FRICTION
      self.y = self.r+NUDGE*TIMESTEP
    if (self.y >= (REF_H-self.r)):
      self.vy *= -FRICTION
      self.y = REF_H-self.r-NUDGE*TIMESTEP

  def getDist2(self, p): # returns distance squared from p
    dy = p.y - self.y
    dx = p.x - self.x
    return (dx*dx+dy*dy)
  def isColliding(self, p): # returns true if it is colliding w/ p
    r = self.r+p.r
    return (r*r > self.getDist2(p)) # if distance is less than total radius, then colliding.

  def update(self):
    self.checkEdges()
    self.move()

  def updateState(self, opponent, myballs, oppballs):
    """ normalized to side, appears different for each agent's perspective"""
    # agent's self
    self.state.x = self.x
    self.state.y = self.y
    self.state.vx = self.vx
    self.state.vy = self.vy
    self.state.life = self.life
    self.state.cooldown = self.cooldown
    # opponent
    self.state.ox = opponent.x
    self.state.oy = opponent.y
    self.state.ovx = opponent.vx
    self.state.ovy = opponent.vy
    self.state.olife = opponent.life
    self.state.ocooldown = opponent.cooldown
    self.state.myballs = []
    for ball in myballs:
        self.state.myballs += [(ball.x, ball.y, ball.vx, ball.vy)]
    self.state.oppballs = []
    for ball in oppballs:
        self.state.oppballs += [(ball.x, ball.y, ball.vx, ball.vy)]
  def getObservation(self):
    return self.state.getObservation()

  def display(self, canvas):
    x = self.x
    y = self.y
    r = self.r

    canvas = circle(canvas, toX(x), toY(y), toP(r), color=self.c)

    # draw coins (lives) left
    for i in range(1, self.life+1):
      canvas = circle(canvas, toX(x+PLAYER_RAD-i*1), toY(y), toP(0.5), color=COIN_COLOR)

    return canvas

class BaselinePolicy:
  """ Tiny RNN policy with only 120 parameters of otoro.net/slimevolley agent """
  def __init__(self):
    self.inputState = None
    pass
  def reset(self):
    pass
  def _forward(self):
    pass
  def _setInputState(self, obs):
    self.inputState = obs
  def _getAction(self):
    return [0, 0, 0, 1]
  def predict(self, obs):
    """ take obs, update rnn state, return action """
    self._setInputState(obs)
    self._forward()
    return self._getAction()
  def getName(self):
    return "Aim_no_move"

class BaselineRand(BaselinePolicy):
  def _getAction(self):
    values = ACTION_SPACE.sample()
    return values
  def getName(self):
    return "Random_actions"

class BaselineRandWAim(BaselinePolicy):
  def _getAction(self):
    values = ACTION_SPACE.sample()
    values[2] = 0
    return values
  def getName(self):
      return "Aim_w_Randmove"

class Game:
  """
  the main slime volley game.
  can be used in various settings, such as ai vs ai, ai vs human, human vs human
  """
  def __init__(self, train_rewards, np_random=np.random):
    self.bullets_good = None
    self.bullets_bad = None
    self.ground = None
    self.agent_bad = None
    self.agent_good = None
    self.np_random = np_random
    self.train_rewards = train_rewards
    self.reset()
  def randRange(self, low, high):
    return self.np_random.uniform(low, high)
  def reset(self):
    self.bullets_good = []
    self.bullets_bad = []
    self.agent_bad = Agent(-1, self.randRange(PLAYER_RAD-REF_W/2, REF_W/2 - PLAYER_RAD), self.randRange(PLAYER_RAD, REF_H - PLAYER_RAD), c=AGENT_LEFT_COLOR)
    self.agent_good = Agent(1, self.randRange(PLAYER_RAD-REF_W/2, REF_W/2 - PLAYER_RAD), self.randRange(PLAYER_RAD, REF_H - PLAYER_RAD), c=AGENT_RIGHT_COLOR)
    while self.agent_good.isColliding(self.agent_bad):
        self.agent_good = Agent(1, self.randRange(PLAYER_RAD-REF_W/2, REF_W/2 - PLAYER_RAD), self.randRange(PLAYER_RAD, REF_H - PLAYER_RAD), c=AGENT_RIGHT_COLOR)

  def newMatch(self):
    reset()

  def fix_bullet_intersection(self):
    for i in range(len(self.bullets_good)):
        goodbullet = self.bullets_good[i]
        for j in range(len(self.bullets_bad)):
            badbullet = self.bullets_bad[j]
            if badbullet.isColliding(goodbullet):
                if self.np_random.rand() < 0.5:
                    del self.bullets_bad[j]
                else:
                    del self.bullets_good[i]
                return False
    return True

  def step(self):
    """ main game loop """
    self.agent_good.update()
    self.agent_bad.update()
    extrareward = 0

    truedir = math.atan2(self.agent_bad.y - self.agent_good.y, self.agent_bad.x - self.agent_good.x)
    # extrareward += 0.5-abs(truedir-self.agent_good.dirpt)/(2*math.pi)

    if self.agent_good.primedbullet is not None:
        self.bullets_good.append(self.agent_good.primedbullet)
        self.agent_good.primedbullet = None

    if self.agent_bad.primedbullet is not None:
        self.bullets_bad.append(self.agent_bad.primedbullet)
        self.agent_bad.primedbullet = None

    for bullet in self.bullets_good:
        bullet.move()
    self.bullets_good = [bullet for bullet in self.bullets_good if not bullet.outOfBounds()]

    for bullet in self.bullets_bad:
        bullet.move()
    self.bullets_bad = [bullet for bullet in self.bullets_bad if not bullet.outOfBounds()]

    while True:
        if self.fix_bullet_intersection():
            break
        else:
            extrareward += 3

    for i in range(len(self.bullets_good)):
        goodbullet = self.bullets_good[i]
        if goodbullet.isColliding(self.agent_bad):
            extrareward += 10
            self.agent_bad.life -= 1
            del self.bullets_good[i]
            break

    for i in range(len(self.bullets_bad)):
        badbullet = self.bullets_bad[i]
        if badbullet.isColliding(self.agent_good):
            extrareward -= 10
            self.agent_good.life -= 1
            del self.bullets_bad[i]
            break

    if self.agent_bad.isColliding(self.agent_good):
        minval = min(self.agent_good.life, self.agent_bad.life)
        self.agent_good.life -= minval
        self.agent_bad.life -= minval
        extrareward += minval * 10 * (-1 if self.agent_bad.life > 0 else 1 if self.agent_good.life > 0 else 0)

    self.agent_good.updateState(self.agent_bad, self.bullets_good, self.bullets_bad)
    self.agent_bad.updateState(self.agent_good, self.bullets_bad, self.bullets_good)

    isTie = self.agent_bad.life == 0 and self.agent_good.life == 0
    return (extrareward if self.train_rewards else 0) + (0 if isTie else -100 if self.agent_good.life == 0 else 100 if self.agent_bad.life == 0 else 0)
  def display(self, canvas):
    # background color
    # if PIXEL_MODE is True, canvas is an RGB array.
    # if PIXEL_MODE is False, canvas is viewer object
    canvas = create_canvas(canvas, c=BACKGROUND_COLOR)
    canvas = self.agent_good.display(canvas)
    canvas = self.agent_bad.display(canvas)
    for bullet in self.bullets_good:
        canvas = bullet.display(canvas)
    for bullet in self.bullets_bad:
        canvas = bullet.display(canvas)
    return canvas

class TankGymEnv(gym.Env):
  """
  Gym wrapper for Slime Volley game.

  By default, the agent you are training controls the right agent
  on the right. The agent on the left is controlled by the baseline
  RNN policy.

  Game ends when an agent loses 5 matches (or at t=3000 timesteps).

  Note: Optional mode for MARL experiments, like self-play which
  deviates from Gym env. Can be enabled via supplying optional action
  to override the default baseline agent's policy:

  obs1, reward, done, info = env.step(action1, action2)

  the next obs for the right agent is returned in the optional
  fourth item from the step() method.

  reward is in the perspective of the right agent so the reward
  for the left agent is the negative of this number.
  """
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  train_rewards = False
  from_pixels = False
  multiagent = True # optional args anyways

  def __init__(self):
    """
    Reward modes:

    net score = right agent wins minus left agent wins

    0: returns net score (basic reward)
    1: returns 0.01 x number of timesteps (max 3000) (survival reward)
    2: sum of basic reward and survival reward

    0 is suitable for evaluation, while 1 and 2 may be good for training

    Setting multiagent to True puts in info (4th thing returned in stop)
    the otherObs, the observation for the other agent. See multiagent.py

    Setting self.from_pixels to True makes the observation with multiples
    of 84, since usual atari wrappers downsample to 84x84
    """

    self.t = 0
    self.t_limit = 3000
    self.epnum = 0

    #self.action_space = spaces.Box(0, 1.0, shape=(3,))
    self.action_space = ACTION_SPACE

    if self.from_pixels:
      setPixelObsMode()
      self.observation_space = spaces.Box(low=0, high=255,
        shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
    else:
      high = np.array([np.finfo(np.float32).max] * (13 + 12 * MAX_BULLETS))
      self.observation_space = spaces.Box(-high, high)
    self.canvas = None
    self.previous_rgbarray = None

    self.game = Game(self.train_rewards)
    self.policy = BaselinePolicy() # the “bad guy”

    self.viewer = None

    # another avenue to override the built-in AI's action, going past many env wraps:
    self.otherAction = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.game = Game(self.train_rewards, np_random=self.np_random)
    return [seed]

  def getObs(self):
    if self.from_pixels:
      obs = self.render(mode='state')
      self.canvas = obs
    else:
      obs = self.game.agent_good.getObservation()
    return obs

  def step(self, action, otherAction=None):
    """
    baseAction is only used if multiagent mode is True
    note: although the action space is multi-binary, float vectors
    are fine (refer to setAction() to see how they get interpreted)
    """
    done = False
    self.t += 1

    if self.otherAction is not None:
      otherAction = self.otherAction

    if otherAction is None: # override baseline policy
      obs = self.game.agent_bad.getObservation()
      otherAction = self.policy.predict(obs)

    self.game.agent_bad.setAction(otherAction)
    self.game.agent_good.setAction(action) # external agent is agent_right

    reward = self.game.step()

    obs = self.getObs()

    if self.t >= self.t_limit:
      done = True

    if self.game.agent_good.life <= 0 or self.game.agent_bad.life <= 0:
      done = True
      self.epnum += 1

    otherObs = None
    if self.multiagent:
      if self.from_pixels:
        goodmask = np.all(obs == AGENT_LEFT_COLOR, axis=-1)
        badmask = np.all(obs == AGENT_RIGHT_COLOR, axis=-1)
        obs[goodmask] = AGENT_RIGHT_COLOR
        obs[badmask] = AGENT_LEFT_COLOR
      else:
        otherObs = self.game.agent_bad.getObservation()

    info = {
      'ale.lives': self.game.agent_bad.lives(),
      'ale.otherLives': self.game.agent_good.lives(),
      'otherObs': otherObs,
      'state': self.game.agent_good.getObservation(),
      'otherState': self.game.agent_bad.getObservation(),
    }

    if self.epnum % 100 == 0 and RENDER:
        self.render()
    return obs, reward, done, info

  def init_game_state(self):
    self.t = 0
    self.game.reset()

  def reset(self):
    self.init_game_state()
    return self.getObs()

  def checkViewer(self):
    # for opengl viewer
    if self.viewer is None:
      checkRendering()
      self.viewer = rendering.SimpleImageViewer(maxwidth=2160) # macbook pro resolution

  def render(self, mode='human', close=False):

    if PIXEL_MODE:
      if self.canvas is not None: # already rendered
        rgb_array = self.canvas
        self.canvas = None
        if mode == 'rgb_array' or mode == 'human':
          self.checkViewer()
          larger_canvas = upsize_image(rgb_array)
          self.viewer.imshow(larger_canvas)
          if (mode=='rgb_array'):
            return larger_canvas
          else:
            return

      self.canvas = self.game.display(self.canvas)
      # scale down to original res (looks better than rendering directly to lower res)
      self.canvas = downsize_image(self.canvas)

      if mode=='state':
        return np.copy(self.canvas)

      # upsampling w/ nearest interp method gives a retro "pixel" effect look
      larger_canvas = upsize_image(self.canvas)
      self.checkViewer()
      self.viewer.imshow(larger_canvas)
      if (mode=='rgb_array'):
        return larger_canvas

    else: # pyglet renderer
      if self.viewer is None:
        checkRendering()
        self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

      self.game.display(self.viewer)
      return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()

class TankGymPixelEnv(TankGymEnv):
  from_pixels = True

class TankGymTrainEnv(TankGymEnv):
  train_rewards = True

class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """Stack n_frames last frames.

    (don't use lazy frames)
    modified from:
    stable_baselines.common.atari_wrappers

    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                        dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_ob()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.n_frames
    return np.concatenate(list(self.frames), axis=2)

#####################
# helper functions: #
#####################

def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
  """
  play one agent vs the other in modified gym-style loop.
  important: returns the score from perspective of policy_right.
  """
  obs_right = env.reset()
  obs_left = obs_right # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  t = 0

  while not done:

    action_right = policy_right.predict(obs_right)
    action_left = policy_left.predict(obs_left)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']

    total_reward += reward
    t += 1

    if render_mode:
      env.render()

  return total_reward, t

####################
# Reg envs for gym #
####################

register(
    id='TankGym-v0',
    entry_point='tankgym.tank:TankGymEnv'
)

register(
    id='TankGymPixel-v0',
    entry_point='tankgym.tank:TankGymPixelEnv'
)

register(
    id='TankGymTrain-v0',
    entry_point='tankgym.tank:TankGymTrainEnv'
)
