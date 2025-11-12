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
import cProfile
import pstats
from io import StringIO


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
        if len(self.history) < n:
            return False
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
EPOCHS = 100_000_000
MAX_REPLAY_SIZE = 1_000_000
MIN_EPSILON = 0.1
MINIBATCH_SIZE = 32
MAX_GAME_LEN = 42 # can't be more than 42 moves
#NUM_ENVS = 2 ** 10
NUM_ENVS = 1
LEARNING_RATE = 0.1 
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

class ProfilingTimer:
    """Simple timing context manager and accumulator"""
    def __init__(self):
        self.timings = {}

    def __call__(self, name):
        return self._Timer(self, name)

    class _Timer:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            self.start = None

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            if self.name not in self.parent.timings:
                self.parent.timings[self.name] = []
            self.parent.timings[self.name].append(elapsed)

    def report(self):
        print("\n" + "="*60)
        print("PROFILING REPORT")
        print("="*60)
        for name, times in sorted(self.timings.items()):
            total = sum(times)
            avg = total / len(times)
            print(f"{name:40s}: {total:8.3f}s total | {avg*1000:8.3f}ms avg | {len(times):6d} calls")
        print("="*60 + "\n")

if __name__ == '__main__':
    # Create profiler and timer
    profiler = cProfile.Profile()
    timer = ProfilingTimer()

    #profiler.enable()

    env_name = 'puffer_connect4'
    env_creator = pufferlib.ocean.env_creator(env_name)

    with timer("env_initialization"):
        vecenv = pufferlib.vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
            #backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': NUM_ENVS})
            backend=pufferlib.vector.Serial, env_kwargs={'num_envs': NUM_ENVS})

    with timer("initial_reset"):
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
    with timer("replay_memory_init"):
        replay = ReplayMemory(MAX_REPLAY_SIZE)

    # 2.
    with timer("model_initialization"):
        policy = DQN().to(device) # put on gpu
        optimizer = optim.AdamW(policy.parameters(), lr=LEARNING_RATE, amsgrad=True)
        criterion = nn.MSELoss()
    #print(policy)
    success_rate = []
    # 3.
    for _ in range(EPOCHS):
        with timer("episode_reset"):
            obs, _ = vecenv.reset()
        #print(f"intial obs {obs}")
        wins = 0
        losses = 1
        success_rate.append(wins / losses)
        # 6.
        for _ in range(MAX_GAME_LEN):
            with timer("action_selection"):
                if random.random() <= EPSILON:
                    actions = np.random.randint(0, 7, (NUM_ENVS))
                    #print(actions.shape)
                else:
                    # argmax Q

                    # (num_envs, obs_size)
                    batch_obs = torch.tensor(obs, dtype=torch.float32, device=device)
                    #print
                    with torch.no_grad():
                        actions = torch.argmax(policy(batch_obs), dim=1).cpu() # index 0-6 is actual the action too

            # lower over 1m frames
            EPSILON = max(MIN_EPSILON, EPSILON - (1.0 - MIN_EPSILON) / 1_000_000)
            old_obs = obs.copy()
            # 8.
            total_steps += 1
            if total_steps % 10000 == 0:
                print(f"Total steps {total_steps}")
                print(f"Success rate {np.mean(success_rate[-50:])}")
                timer.report()

            with timer("env_step"):
                obs1, rewards, terminals, truncations, infos = vecenv.step(actions)
            #time.sleep(1)

            # assumes one env
            with timer("replay_buffer_update"):
                for n in range(NUM_ENVS):
                    new_state = obs[n]

                    if terminals[n] or truncations[n]:
                        new_state = None

                        if rewards[n] == 1.0:
                            wins += 1
                        else:
                            losses += 1


                    trans = Transition(state=old_obs[n], action=actions[n], reward=rewards[n], new_state=obs[n])
                    replay.add_experince(trans)

            with timer("replay_sampling"):
                transitions = replay.sample(MINIBATCH_SIZE)
                if not transitions: # size too small
                    continue
                batch = Transition(*zip(*transitions)) # fun trick to go from list of Transitions to a single transition holding a list of rewds, obs, etc  

            with timer("q_target_computation"):
                with torch.no_grad():
                    phi_j_plus_1 = torch.as_tensor(np.array(batch.new_state), dtype=torch.float32, device=device)
                    with timer("q_target_computation_policy"):
                        q_next = policy(phi_j_plus_1)
                    #print(q_next.shape)
                    max_next_q = q_next.max(1, keepdim=True)[0]
                    #print(max_next_q)
                    #print(batch.reward)
                    reward = torch.tensor(batch.reward, device=device).unsqueeze(1)

                    #print(f"Reward {reward}")
                    #print(f"Max next q {max_next_q}")
                    y_j = reward + GAMMA * max_next_q # this probably breaks when there are terminals etc
                    #print(y_j)

            with timer("q_current_computation"):
                phi_j = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
                q_current = policy(phi_j)
                #print(q_current)
                #print(torch.tensor(batch.action, device=device))

                action_mask = torch.tensor(batch.action, device=device).unsqueeze(1)
                q_selected = q_current.gather(1, action_mask)
                #print(f"Action mask {action_mask}")
                #print(f"Q selected {q_selected}")

            with timer("loss_and_backprop"):
                loss = criterion(y_j, q_selected)
                optimizer.zero_grad() # don't want any gradients from the previous loop
                loss.backward() # do gradient magic
                optimizer.step() # apply gradients to variables 

            #print(f"Loss: {loss.item():.4f}")

    # Stop profiler and print results
    profiler.disable()

    # Print final timing report
    print("\n" + "="*60)
    print("FINAL PROFILING REPORT")
    print("="*60)
    timer.report()

    # Print cProfile statistics
    print("\n" + "="*60)
    print("DETAILED PROFILING STATISTICS (Top 30 by cumulative time)")
    print("="*60)
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Save detailed profile to file
    profiler.dump_stats('connect4_profile.prof')
    print("\nDetailed profile saved to 'connect4_profile.prof'")
    print("View with: python -m pstats connect4_profile.prof")

