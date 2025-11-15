import random
import torch
from DQN import DQN
import os
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl
import time

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

def get_latest_model():
    res = os.scandir("models")
    most_recent = None
    for i in res:
        if most_recent is None:
            most_recent = i
        elif i.stat().st_mtime > most_recent.stat().st_mtime:
            most_recent = i
    print(f"Loading model from {most_recent.path}")
    return most_recent.path

policy = torch.load(get_latest_model(), weights_only=False)

# make env
#env_name = 'puffer_squared'
env_name = 'puffer_connect4'
env_creator = pufferlib.ocean.env_creator(env_name)

vecenv = pufferlib.vector.make(env_creator, num_envs=1, num_workers=1, batch_size=1,
    backend=pufferlib.vector.Serial, env_kwargs={'num_envs': 1})
    #backend=pufferlib.vector.Serial, env_kwargs={'num_envs': 1, "size": 5})

obs, i = vecenv.reset()

done = False
while not done:
    batch_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    actions = torch.argmax(policy(batch_obs), dim=1).cpu()
    obs, rewards, terminals, truncations, infos = vecenv.step(actions)
    if rewards[0]:
        print(f"Reward was {rewards[0]}")
        obs, i = vecenv.reset(random.randint(1, 10000))

    vecenv.driver_env.render()
    time.sleep(0.5)