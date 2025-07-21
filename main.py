import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import ale_py                              # register ALE ROMs
import tqdm

import imageio
import glob
import io
import base64
from IPython.display import HTML, display

# ─── Register all ALE ROMs under the Gymnasium API ─────────────────────────────
gym.register_envs(ale_py)

# ─── Neural Network ─────────────────────────────────────────────────────────────
class Network(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1     = nn.Conv2d(4, 32, 3, stride=2)
        self.conv2     = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3     = nn.Conv2d(64, 64, 3, stride=2)
        self.flatten   = nn.Flatten()
        self.fc_hidden = nn.Linear(64*4*4, 128)
        self.fc_policy = nn.Linear(128, action_size)
        self.fc_value  = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc_hidden(x))
        return self.fc_policy(x), self.fc_value(x).squeeze(-1)

# ─── Atari Preprocessing Wrapper ────────────────────────────────────────────────
class PreprocessAtari(gym.ObservationWrapper):
    def __init__(self, env, height=42, width=42, n_frames=4):
        super().__init__(env)
        self.img_size = (height, width)
        self.frames   = np.zeros((n_frames,) + self.img_size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=self.frames.shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        self.frames.fill(0)
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

    def observation(self, obs):
        return self._process(obs)

    def _process(self, obs):
        img = cv2.resize(obs, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = img
        return self.frames

# ─── Environment Factory ────────────────────────────────────────────────────────
def make_env():
    env = gym.make(
        "ALE/KungFuMaster-v5",
        render_mode="rgb_array",
        full_action_space=False
    )
    return PreprocessAtari(env, height=42, width=42, n_frames=4)

# ─── A2C‑style Agent ────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, n_actions, lr=1e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net    = Network(n_actions).to(self.device)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma  = gamma

    def act(self, states):
        # states already float32, shape (4,42,42) or (batch,4,42,42)
        if states.ndim == 3:
            states = states[None]
        t = torch.from_numpy(states).to(self.device)
        logits, _ = self.net(t)
        probs     = F.softmax(logits, dim=-1).detach().cpu().numpy()
        return np.array([np.random.choice(probs.shape[-1], p=p) for p in probs])

    def step(self, s, a, r, s2, dones):
        s   = torch.from_numpy(s).to(self.device)
        s2  = torch.from_numpy(s2).to(self.device)
        a   = torch.tensor(a, dtype=torch.int64, device=self.device)
        r   = torch.tensor(r, dtype=torch.float32, device=self.device)
        d   = torch.tensor(dones, dtype=torch.float32, device=self.device)

        logits, vals      = self.net(s)
        _,      vals_next = self.net(s2)

        targets   = r + self.gamma * vals_next * (1 - d)
        adv       = (targets.detach() - vals)

        logp      = F.log_softmax(logits, dim=-1)
        chosen_lp = logp[range(len(a)), a]
        entropy   = -(F.softmax(logits, dim=-1) * logp).sum(-1)

        loss_policy = -(chosen_lp * adv.detach()).mean() - 0.001 * entropy.mean()
        loss_value  = F.mse_loss(vals, targets.detach())

        self.opt.zero_grad()
        (loss_policy + loss_value).backward()
        self.opt.step()

# ─── Batch Manager & Training Loop ──────────────────────────────────────────────
class EnvBatch:
    def __init__(self, n_envs=8):
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        return np.stack([env.reset()[0] for env in self.envs], axis=0)

    def step(self, actions):
        next_s, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            obs, rew, term, trunc, info = env.step(int(a))
            done = term or trunc
            if done:
                obs, _ = env.reset()
            next_s.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(next_s),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=bool),
            infos
        )

def evaluate(agent, env, episodes=5):
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.act(s)[0]
            s, r, t, tr, _ = env.step(int(a))
            done = t or tr
            total += r
        returns.append(total)
    return returns

# ─── Video Display Utilities ───────────────────────────────────────────────────
def show_video_of_model(agent, seed=0):
    # use our wrapped env so `state` is preprocessed
    env = make_env()
    state, _ = env.reset(seed=seed)
    done = False
    frames = []
    while not done:
        # grab raw frame from the underlying ALE env
        raw = env.env.render()  
        frames.append(raw)
        action = int(agent.act(state)[0])
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30, codec='h264_mf')

def show_video():
    mp4list = glob.glob('video.mp4')
    if not mp4list:
        print("Could not find video")
        return
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    display(HTML(f'''
        <video alt="Agent Demo" autoplay loop controls style="height:400px;">
          <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4"/>
        </video>'''))

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BATCH_SIZE = 64
    STEPS      = 300000

    batch_env = EnvBatch(BATCH_SIZE)
    agent     = Agent(batch_env.envs[0].action_space.n)
    eval_env  = make_env()

    states = batch_env.reset()
    for step in tqdm.trange(1, STEPS+1):
        actions      = agent.act(states)
        next_states, rewards, dones, _ = batch_env.step(actions)
        agent.step(states, actions, rewards * 0.01, next_states, dones)
        states = next_states

        if step % 1000 == 0:
            avg_ret = np.mean(evaluate(agent, eval_env, episodes=10))
            tqdm.tqdm.write(f"[Step {step}] AvgEvalReward: {avg_ret:.2f}")

    # after training, record and display
    show_video_of_model(agent)
    show_video()




# meow :3 If the code doesn't run, please check if you have registered the ROMs. pip install autorom , then open the scripts file and run the autorom script to accept the licences. 



# :3


#Have a great day <3333 