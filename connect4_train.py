import torch
from torch import nn
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl
import numpy as np
from collections import namedtuple
from collections import deque
import random
import pprint
import torch.optim as optim
import time


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'new_state', 'reward'))

class ReplayMemory():
    def __init__(self, size):
        self.history = deque([], maxlen=size)

    def add_experince(self, transition: Transition):
        assert type(transition) == Transition
        self.history.append(transition)

    def sample(self, n):
        return random.sample(self.history, n)

    def __len__(self):
        return len(self.history)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(7*6, 512), # 7*6 42 spaces avaiable
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 7), # 7 spaces to play
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x) # logits here because the values are straight from the model, not softmaxxed or normalized
        return logits

''' In these experiments, we used the RMSProp algorithm with minibatches of size 32. The behavior
policy during training was -greedy with  annealed linearly from 1 to 0.1 over the first million
frames, and fixed at 0.1 thereafter. We trained for a total of 10 million frames and used a replay
memory of one million most recent frames. '''

EPSILON = 1.0
EPOCHS = 10_000_000
MAX_REPLAY_SIZE = 1_000_000
MIN_EPSILON = 0.1
MINIBATCH_SIZE = 32
MAX_GAME_LEN = 42 # can't be more than 42 moves
NUM_ENVS = 4
LEARNING_RATE = 0.001 
GAMMA = 0.99 # looks forward 100 steps, which should be more than enough

'''
1. Initialize replay memory D to capacity N 
2. Initialize action-value function Q with random weights
3. for episode = 1, M do
4.    Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
5.    for t = 1, T do
6.        With probability  select a random action at
7.        otherwise select at = maxa Q∗(φ(st), a; θ)
8.        Execute action at in emulator and observe reward rt and image xt+1
        Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
        Store transition (φt, at, rt, φt+1) in D
        Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
        Set yj =
        { rj for terminal φj+1
        rj + γ maxa′ Q(φj+1, a′; θ) for non-terminal φj+1
        Perform a gradient descent step on (yj − Q(φj , aj ; θ))2 according to equation 3
    end for
end for
'''

if __name__ == '__main__':
    env_name = 'puffer_connect4'
    env_creator = pufferlib.ocean.env_creator(env_name)
    vecenv = pufferlib.vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
        #backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': NUM_ENVS})
        backend=pufferlib.vector.Serial, env_kwargs={'num_envs': NUM_ENVS})

    obs, _ = vecenv.reset()

    total_steps = 0
    # actions = [3] * NUM_ENVS
    # for i in range(2):
    #     obs, rewards, terminals, truncations, infos = vecenv.step(actions)
    # print(obs)
    # for row in obs[0]:
    #     print(row)
    # print(len(obs[0]))
    # exit()

    # 1.
    replay = ReplayMemory(MAX_REPLAY_SIZE)
    
    # 2.
    policy = DQN().to(device) # put on gpu
    optimizer = optim.AdamW(policy.parameters(), lr=LEARNING_RATE, amsgrad=True)
    criterion = nn.MSELoss()
    print(policy)

    # 3. 
    for _ in range(EPOCHS):
        obs, _ = vecenv.reset()
        #print(f"intial obs {obs}")

        # 6.
        for _ in range(MAX_GAME_LEN):
            if random.random() <= 0:
                actions = np.random.randint(0, 7, (NUM_ENVS))
                #print(actions.shape)
            else:
                # argmax Q 

                # (num_envs, obs_size)
                batch_obs = torch.tensor(obs, dtype=torch.float32, device=device)
                #print
                with torch.no_grad():
                    actions = torch.argmax(policy(batch_obs), dim=1).cpu() # index 0-6 is actual the action too
            
            old_obs = obs.copy()
            # 8.
            total_steps += 1
            obs1, rewards, terminals, truncations, infos = vecenv.step(actions)
            #time.sleep(1)

            # assumes one env
            for n in range(NUM_ENVS):
                new_state = obs[n]

                if terminals[n] or truncations[n]:
                    new_state = None

                trans = Transition(state=old_obs[n], action=actions[n], reward=rewards[n], new_state=obs[n])
                replay.add_experince(trans)
            

            state_plus_1 = torch.tensor(obs1, dtype=torch.float32, device=device) 

            # TODO, should be minibatch size
            transitions = replay.sample(2) 
            batch = Transition(*zip(*transitions)) # fun trick to go from list of Transitions to a single transition holding a list of rewds, obs, etc  

            with torch.no_grad():
                phi_j_plus_1 = torch.tensor(np.array(batch.new_state), dtype=torch.float32, device=device) 
                q_next = policy(phi_j_plus_1)
                #print(q_next.shape)
                max_next_q = q_next.max(1, keepdim=True)[0]
                print(max_next_q)
                print(batch.reward)
                reward = torch.tensor(batch.reward, device=device).unsqueeze(0)
                print(reward)
                y_j = reward + GAMMA * max_next_q # this probably breaks when there are terminals etc

            phi_j = torch.tensor(batch.state, dtype=torch.float32, device=device) 
            q_current = policy(phi_j)
            print(q_current)
            print(torch.tensor(batch.action, device=device))
            q_selected = q_current.gather(1, torch.tensor(batch.action, device=device).unsqueeze(0))
            
            loss = criterion(y_j, q_selected)
            optimizer.zero_grad() # don't want any gradients from the previous loop
            loss.backward() # do gradient magic
            optimizer.step() # apply gradients to variables 

            print(f"Loss: {loss.item():.4f}")
