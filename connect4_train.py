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
import atexit
import datetime
import signal
from sys import exit
from DQN import DQN

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


Transition = namedtuple('Transition', ('state', 'action', 'new_state', 'reward'))

# don't use random.sample on deque! this is what SB3 uses (i think)
class ReplayMemoryPrior():
    def __init__(self, size, obs_shape):
        self.capacity = size
        self.states = np.zeros((size, obs_shape), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, obs_shape), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add_experince(self, transition: Transition):
        idx = self.position
        self.states[idx] = transition.state
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_states[idx] = transition.new_state if transition.new_state is not None else 0
        self.dones[idx] = 0 if transition.new_state is None else 1
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        #print(f"Slow impl New position {self.position}")
        #print(f"Slow Replay size before add {self.size}")

    def sample(self, n):
        if self.size < n:
            return False, False, False, False, False
        indices = np.random.randint(0, self.size, size=n)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


# don't use random.sample on deque! this is what SB3 uses (i think)
class ReplayMemory():
    def __init__(self, size, obs_shape):
        self.capacity = size
        self.states = np.zeros((size, obs_shape), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, obs_shape), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add_experince(self, states, actions, rewards, new_states, terminals, truncations):
        starting_idx = self.position
        ending_idx = (self.position + len(states) - 1) % self.capacity
        if starting_idx > ending_idx:
            first_part = self.capacity - self.position
            second_part = len(states) - first_part
            print("Got wrap around")
            #print(f"First part {first_part} second part {second_part}")
            #print(f"Capacity {self.capacity} size {self.size}")
            #print(f"Trying to add {len(states)} experiences")
            self.states[self.position:] = states[:first_part]
            self.states[:second_part] = states[first_part:]
            self.actions[self.position:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[self.position:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
            done_mask = np.logical_not(np.logical_or(terminals, truncations)) # we want 0 to mean there was NO next state
            self.dones[self.position:] = done_mask[:first_part]
            self.dones[:second_part] = done_mask[first_part:]
            new_states[np.logical_or(terminals, truncations)] = 0 # set next state
            self.next_states[self.position:] = new_states[:first_part]
            self.next_states[:second_part] = new_states[first_part:]

            #part1 = a[start_index:]
            #part2 = a[:end_index]
            #result = np.concatenate([part1, part2])
        else:
            #result = a[start_index:end_index]
            # print(self.states)
            # print(self.states.shape)
            # print(states)
            # print(states.shape)
            #print(f"Starting idx {starting_idx} ending {ending_idx}")

            self.states[starting_idx:ending_idx+1] = states
            #print(self.states)
            self.actions[starting_idx:ending_idx+1] = actions
            #print(f"Actions stored {self.actions[starting_idx:ending_idx+1]}")
            self.rewards[starting_idx:ending_idx+1] = rewards

            done_mask = np.logical_not(np.logical_or(terminals, truncations)) # we want 0 to mean there was NO next state
            #print(f"Done mask {done_mask}")
            self.dones[starting_idx:ending_idx+1] = done_mask
            new_states[np.logical_or(terminals, truncations)] = 0 # set next state to 0 if done
            self.next_states[starting_idx:ending_idx+1] = new_states

        self.position = (self.position + len(states) - 1) % self.capacity + 1
        #print(f"Fast impl New position {self.position}")
        self.size = min(self.size + len(states), self.capacity)
        #print(f"Fast Replay size before add {self.size}")

    def sample(self, n):
        if self.size < n:
            return False, False, False, False, False
        indices = np.random.randint(0, self.size, size=n)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


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


''' In these experiments, we used the RMSProp algorithm with minibatches of size 32. The behavior
policy during training was -greedy with  annealed linearly from 1 to 0.1 over the first million
frames, and fixed at 0.1 thereafter. We trained for a total of 10 million frames and used a replay
memory of one million most recent frames. '''

EPSILON = 1.0
EPOCHS = 100_000_000
MAX_REPLAY_SIZE = 1_000_0
MIN_EPSILON = 0.1
MINIBATCH_SIZE = 64
MAX_GAME_LEN = 42 # can't be more than 42 moves
#NUM_ENVS = 2 ** 10
NUM_ENVS = 256
LEARNING_RATE = 1e-4 
GAMMA = 0.99 # looks forward 100 steps, which should be more than enough

np.random.seed(42)
torch.random.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed(42)

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

def recent_reward(rewards: deque):
    return round(np.mean([success_rate[i] for i in range(0, 50)]), 3) 

def save_model(_, __):
    model_path = f"models/{env_name}_{recent_reward(success_rate)}_{datetime.datetime.now()}.pt"
    print(f"Saving model to {model_path}")
    torch.save(policy, model_path)
    exit(0)

if __name__ == '__main__':
    # Create profiler and timer
    profiler = cProfile.Profile()
    timer = ProfilingTimer()

    signal.signal(signal.SIGINT, save_model) # ctlr + c
    #profiler.enable()

    # connect 4!! 
    env_name = 'puffer_squared'
    env_creator = pufferlib.ocean.env_creator(env_name)

    with timer("env_initialization"):
        vecenv = pufferlib.vector.make(env_creator, num_envs=10, num_workers=10, batch_size=1,
            #backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': NUM_ENVS})
            backend=pufferlib.vector.Serial, env_kwargs={'num_envs': NUM_ENVS, "size": 5})

    # vecenv = pufferlib.ocean.make_bandit()
    # print(vecenv)
    # print(type(vecenv))
    # print(vecenv.__dict__)
    with timer("initial_reset"):
        obs, _ = vecenv.reset()

    total_steps = 0

    # 1.
    with timer("replay_memory_init"):
        replay = ReplayMemory(MAX_REPLAY_SIZE, vecenv.single_observation_space.shape[0])
        replay_old = ReplayMemoryPrior(MAX_REPLAY_SIZE, vecenv.single_observation_space.shape[0])
    # 2.
    with timer("model_initialization"):
        policy = DQN(obs_size=vecenv.single_observation_space.shape[0], action_size=vecenv.single_action_space.n).to(device) # put on gpu
        optimizer = optim.AdamW(policy.parameters(), lr=LEARNING_RATE, amsgrad=True)
        criterion = nn.MSELoss()
    with timer("initial_reset"):
        obs, _ = vecenv.reset()

    obs_shape = vecenv.single_observation_space.shape[0]

    #print(policy)
    success_rate = deque(maxlen=1000)
    for i in range(50):
        success_rate.append(0)
    # 3.
    wins = 0
    losses = 1
    for _ in range(EPOCHS):
        with timer("episode_reset"):
            obs, _ = vecenv.reset()
        #print(f"intial obs {obs}")

        success_rate.appendleft(wins / (wins + losses))
        wins = 0
        losses = 1
        # 6.
        for _ in range(MAX_GAME_LEN):
            with timer("action_selection"):
                if random.random() <= EPSILON:
                    actions = np.random.randint(0, vecenv.single_action_space.n, (NUM_ENVS))
                    #print(actions.shape)
                else:
                    # argmax Q

                    # (num_envs, obs_size)
                    batch_obs = torch.tensor(obs, dtype=torch.float32, device=device)
                    #print
                    with torch.no_grad():
                        actions = torch.argmax(policy(batch_obs), dim=1).cpu() # index 0-6 is actual the action too

            # lower over 1m frames
            EPSILON = max(MIN_EPSILON, EPSILON - (1.0 - MIN_EPSILON) / 1_000_0)
            if recent_reward(success_rate) > 0.8:
                EPSILON = 0.0
            old_obs = obs.copy()
            # 8.
            total_steps += 1
            if total_steps % 100 == 0:
                print(f"Total steps {total_steps}")
                print(f"Success rate {recent_reward(success_rate)}")
                print(f"Epsilon {EPSILON}")
                timer.report()

            with timer("env_step"):
                #print("Step")
                obs_new, rewards, terminals, truncations, infos = vecenv.step(actions)
                #obs_new, rewards, terminals, truncations, infos = [obs_new], [rewards], [terminals], [truncations], [infos]
            #time.sleep(1)

            # assumes one env
            with timer("replay_buffer_update"):
                for n in range(NUM_ENVS):
                    new_state = obs_new[n]

                    if terminals[n] or truncations[n]:
                        new_state = None
                        if rewards[n] == 1.0:
                            #print("Postitive reward!!!")
                            wins += 1
                        else:
                            losses += 1


                    trans = Transition(state=old_obs[n], action=actions[n], reward=rewards[n], new_state=new_state)
                    replay_old.add_experince(trans)
                replay.add_experince(old_obs.copy(), actions.copy(), rewards.copy(), obs_new.copy(), terminals.copy(), truncations.copy())

            #print(replay_old.actions)
            #print(replay.actions)
            #assert np.array_equal(replay_old.actions, replay.actions)
            if not np.array_equal(replay_old.states, replay.states):
                diff_mask = replay_old.states != replay.states
                diff_indices = np.where(diff_mask)[0]
                print(f"State Arrays differ at indices: {diff_indices}")
                print(f"Old values at those indices: {replay_old.states[diff_indices]}")
                print(f"New values at those indices: {replay.states[diff_indices]}")
                exit()
            assert np.array_equal(replay_old.states, replay.states)
            assert np.array_equal(replay_old.rewards, replay.rewards)
            #print(f"Replay old {replay_old.dones}")
            #print(f"Replay new {replay.dones}")
            assert np.array_equal(replay_old.dones, replay.dones)
            #print(f"Replay old next states {replay_old.next_states}")
            #print(f"Replay new next states {replay.next_states}")
            # assert np.array_equal(replay_old.next_states, replay.next_states)
            if not np.array_equal(replay_old.next_states, replay.next_states):
                diff_mask = replay_old.next_states != replay.next_states
                diff_indices = np.where(diff_mask)[0]
                print(f"Next State Arrays differ at indices: {diff_indices}")
                print(f"Old values at those indices: {replay_old.next_states[diff_indices]}")
                print(f"New values at those indices: {replay.next_states[diff_indices]}")
                exit()

            

            with timer("replay_sampling"):
                with timer("replay_sampling_p1"):
                    
                    # self.states[indices],
                    # self.actions[indices],
                    # self.rewards[indices],
                    # self.next_states[indices],
                    # self.dones[indices]
                    #states, actions, rewards, next_states, done_mask = replay.sample(MINIBATCH_SIZE)
                    states, actions, rewards, next_states, done_mask = replay_old.sample(MINIBATCH_SIZE)

                    if states is False: # size too small
                        continue


            with timer("q_target_computation"):
                with torch.no_grad():
                    non_terminal_mask = torch.from_numpy(done_mask).unsqueeze(1).to(device)
                    phi_j_plus_1 = torch.from_numpy(next_states).to(device)

                    q_next = policy(phi_j_plus_1)
                    max_next_q = q_next.max(1, keepdim=True)[0]

                    reward = torch.from_numpy(rewards).unsqueeze(1).to(device)
                    y_j = reward + GAMMA * max_next_q * non_terminal_mask

            with timer("q_current_computation"):
                phi_j = torch.from_numpy(states).to(device)
                q_current = policy(phi_j)

                action_mask = torch.from_numpy(actions).unsqueeze(1).to(device)
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
