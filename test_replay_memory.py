import numpy as np
import pytest
from collections import namedtuple
from connect4_train import ReplayMemory, ReplayMemoryPrior, Transition
from tqdm import trange

class TestReplayMemoryComparison:
    """Unit tests to ensure ReplayMemory and ReplayMemoryPrior are bit-for-bit identical"""

    @pytest.fixture
    def setup_replays(self):
        """Setup both replay buffers with the same configuration"""
        capacity = 1000
        obs_shape = 42  # Connect 4 observation shape
        replay_new = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)
        return replay_new, replay_old, obs_shape

    def assert_replays_equal(self, replay_old, replay_new, msg=""):
        """Helper to assert all fields are identical"""
        assert np.array_equal(replay_old.actions, replay_new.actions), f"Actions differ {msg}"
        assert np.array_equal(replay_old.states, replay_new.states), f"States differ {msg}"
        assert np.array_equal(replay_old.rewards, replay_new.rewards), f"Rewards differ {msg}"
        assert np.array_equal(replay_old.dones, replay_new.dones), f"Dones differ {msg}"
        assert np.array_equal(replay_old.next_states, replay_new.next_states), f"Next states differ {msg}"
        assert replay_old.position == replay_new.position, f"Position differs {msg}"
        assert replay_old.size == replay_new.size, f"Size differs {msg}"

    def test_eq_at_start(self, setup_replays):
        """Test that both replays are equal at initialization"""
        replay_new, replay_old, _ = setup_replays
        self.assert_replays_equal(replay_old, replay_new, "at initialization")
    
    def test_single_transition_non_terminal(self, setup_replays):
        """Test adding a single non-terminal transition"""
        replay_new, replay_old, obs_shape = setup_replays

        # Create test data for single transition
        state = np.random.randn(obs_shape).astype(np.float32)
        action = 3
        reward = 0.0
        next_state = np.random.randn(obs_shape).astype(np.float32)
        terminal = False
        truncation = False

        # Add to old replay (single transition)
        trans = Transition(state=state, action=action, reward=reward, new_state=next_state)
        replay_old.add_experince(trans)

        # Add to new replay (batch of 1)
        replay_new.add_experince(
            states=np.array([state]),
            actions=np.array([action]),
            rewards=np.array([reward]),
            new_states=np.array([next_state]),
            terminals=np.array([terminal]),
            truncations=np.array([truncation])
        )


        self.assert_replays_equal(replay_old, replay_new, "after single non-terminal transition")

    def test_single_transition_terminal(self, setup_replays):
        """Test adding a single terminal transition"""
        replay_new, replay_old, obs_shape = setup_replays

        # Create test data for terminal transition
        state = np.random.randn(obs_shape).astype(np.float32)
        action = 5
        reward = 1.0
        terminal = True
        truncation = False

        # Add to old replay (terminal means new_state=None)
        trans = Transition(state=state, action=action, reward=reward, new_state=None)
        replay_old.add_experince(trans)

        # Add to new replay
        next_state = np.random.randn(obs_shape).astype(np.float32)  # Will be overwritten
        replay_new.add_experince(
            states=np.array([state]),
            actions=np.array([action]),
            rewards=np.array([reward]),
            new_states=np.array([next_state]),
            terminals=np.array([terminal]),
            truncations=np.array([truncation])
        )

        self.assert_replays_equal(replay_old, replay_new, "after single terminal transition")

    def test_large_batch_256_envs(self):
        """Test with 256 environments like in actual training - this catches the wraparound bug"""
        capacity = 10000
        obs_shape = 25  # From the error message
        num_envs = 256

        replay_new = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)

        np.random.seed(42)

        # Simulate multiple steps like real training
        for step in range(50):  # 50 steps * 256 envs = 12,800 transitions
            states = np.random.randn(num_envs, obs_shape).astype(np.float32)
            actions = np.random.randint(0, 5, size=num_envs)  # 5 actions from error message
            rewards = np.random.randn(num_envs).astype(np.float32)
            next_states = np.random.randn(num_envs, obs_shape).astype(np.float32)
            terminals = np.random.rand(num_envs) < 0.1  # 10% terminal
            truncations = np.random.rand(num_envs) < 0.05  # 5% truncation

            # Add to new replay (batch)

            # Add to old replay (one by one)
            for i in range(num_envs):
                new_state = next_states[i]
                if terminals[i] or truncations[i]:
                    new_state = None

                trans = Transition(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    new_state=new_state
                )
                replay_old.add_experince(trans)
                
            replay_new.add_experince(states, actions, rewards, next_states.copy(), terminals, truncations)

            # Check after each batch
            if not np.array_equal(replay_old.states, replay_new.states):
                diff_mask = replay_old.states != replay_new.states
                diff_indices = np.where(diff_mask)[0]
                pytest.fail(f"Step {step}: States differ at {len(diff_indices)} indices: {diff_indices[:50]}\n"
                           f"Old position: {replay_old.position}, New position: {replay_new.position}\n"
                           f"Old size: {replay_old.size}, New size: {replay_new.size}")

            self.assert_replays_equal(replay_old, replay_new, f"after step {step} with 256 envs")

    def test_single_transition_truncation(self, setup_replays):
        """Test adding a single truncated transition"""
        replay_new, replay_old, obs_shape = setup_replays

        # Create test data for truncated transition
        state = np.random.randn(obs_shape).astype(np.float32)
        action = 2
        reward = -1.0
        terminal = False
        truncation = True

        # Add to old replay (truncation means new_state=None)
        trans = Transition(state=state, action=action, reward=reward, new_state=None)
        replay_old.add_experince(trans)

        # Add to new replay
        next_state = np.random.randn(obs_shape).astype(np.float32)  # Will be overwritten
        replay_new.add_experince(
            states=np.array([state]),
            actions=np.array([action]),
            rewards=np.array([reward]),
            new_states=np.array([next_state]),
            terminals=np.array([terminal]),
            truncations=np.array([truncation])
        )

        self.assert_replays_equal(replay_old, replay_new, "after single truncation transition")

    def test_batch_transitions_mixed(self, setup_replays):
        """Test adding a batch of mixed transitions (non-terminal, terminal, truncation)"""
        replay_new, replay_old, obs_shape = setup_replays
        num_envs = 8

        # Create batch data
        states = np.random.randn(num_envs, obs_shape).astype(np.float32)
        actions = np.random.randint(0, 7, size=num_envs)
        rewards = np.random.randn(num_envs).astype(np.float32)
        next_states = np.random.randn(num_envs, obs_shape).astype(np.float32)

        # Mixed: some terminal, some truncated, some neither
        terminals = np.array([False, True, False, False, True, False, False, False])
        truncations = np.array([False, False, True, False, False, True, False, False])

        # Add to new replay (batch)
        replay_new.add_experince(states, actions, rewards, next_states.copy(), terminals, truncations)

        # Add to old replay (one by one)
        for i in range(num_envs):
            new_state = next_states[i]
            if terminals[i] or truncations[i]:
                new_state = None

            trans = Transition(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                new_state=new_state
            )
            replay_old.add_experince(trans)

        self.assert_replays_equal(replay_old, replay_new, "after batch mixed transitions")

    def test_multiple_batches(self, setup_replays):
        """Test adding multiple batches over time"""
        replay_new, replay_old, obs_shape = setup_replays
        num_envs = 16
        num_batches = 10

        np.random.seed(42)  # For reproducibility

        for batch_idx in range(num_batches):
            # Create batch data
            states = np.random.randn(num_envs, obs_shape).astype(np.float32)
            actions = np.random.randint(0, 7, size=num_envs)
            rewards = np.random.randn(num_envs).astype(np.float32)
            next_states = np.random.randn(num_envs, obs_shape).astype(np.float32)
            terminals = np.random.rand(num_envs) < 0.2  # 20% terminal
            truncations = np.random.rand(num_envs) < 0.1  # 10% truncation

            # Add to new replay
            replay_new.add_experince(states, actions, rewards, next_states.copy(), terminals, truncations)

            # Add to old replay
            for i in range(num_envs):
                new_state = next_states[i]
                if terminals[i] or truncations[i]:
                    new_state = None

                trans = Transition(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    new_state=new_state
                )
                replay_old.add_experince(trans)

            # Check equality after each batch
            self.assert_replays_equal(replay_old, replay_new, f"after batch {batch_idx}")

    def test_buffer_wraparound(self, setup_replays):
        """Test that both buffers handle wraparound identically"""
        capacity = 100
        obs_shape = 42
        replay_new = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)

        num_envs = 8
        # Add enough to cause wraparound (more than capacity)
        num_batches = 20  # 20 * 8 = 160 > 100

        np.random.seed(123)

        for batch_idx in range(num_batches):
            states = np.random.randn(num_envs, obs_shape).astype(np.float32)
            actions = np.random.randint(0, 7, size=num_envs)
            rewards = np.random.randn(num_envs).astype(np.float32)
            next_states = np.random.randn(num_envs, obs_shape).astype(np.float32)
            terminals = np.random.rand(num_envs) < 0.15
            truncations = np.random.rand(num_envs) < 0.05

            # Add to new replay
            replay_new.add_experince(states, actions, rewards, next_states.copy(), terminals, truncations)

            # Add to old replay
            for i in range(num_envs):
                new_state = next_states[i]
                if terminals[i] or truncations[i]:
                    new_state = None

                trans = Transition(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    new_state=new_state
                )
                replay_old.add_experince(trans)

        self.assert_replays_equal(replay_old, replay_new, "after buffer wraparound")

    def test_all_terminals(self, setup_replays):
        """Test edge case where all transitions are terminal"""
        replay_new, replay_old, obs_shape = setup_replays
        num_envs = 10

        states = np.random.randn(num_envs, obs_shape).astype(np.float32)
        actions = np.random.randint(0, 7, size=num_envs)
        rewards = np.ones(num_envs, dtype=np.float32)
        next_states = np.random.randn(num_envs, obs_shape).astype(np.float32)
        terminals = np.ones(num_envs, dtype=bool)
        truncations = np.zeros(num_envs, dtype=bool)

        # Add to new replay
        replay_new.add_experince(states, actions, rewards, next_states.copy(), terminals, truncations)

        # Add to old replay
        for i in range(num_envs):
            trans = Transition(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                new_state=None  # All terminal
            )
            replay_old.add_experince(trans)

        self.assert_replays_equal(replay_old, replay_new, "with all terminal transitions")

    def test_detailed_field_comparison(self, setup_replays):
        """Detailed test that checks each field individually and reports differences"""
        replay_new, replay_old, obs_shape = setup_replays
        num_envs = 5

        # Create specific test data
        states = np.array([
            np.ones(obs_shape) * i for i in range(num_envs)
        ], dtype=np.float32)
        actions = np.array([0, 1, 2, 3, 4])
        rewards = np.array([0.0, 1.0, -1.0, 0.5, 0.0], dtype=np.float32)
        next_states = np.array([
            np.ones(obs_shape) * (i + 10) for i in range(num_envs)
        ], dtype=np.float32)
        terminals = np.array([False, True, False, False, True])
        truncations = np.array([False, False, True, False, False])

        # Add to new replay
        replay_new.add_experince(states, actions, rewards, next_states.copy(), terminals, truncations)

        # Add to old replay
        for i in range(num_envs):
            new_state = next_states[i]
            if terminals[i] or truncations[i]:
                new_state = None

            trans = Transition(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                new_state=new_state
            )
            replay_old.add_experince(trans)

        # Detailed comparison with helpful error messages
        if not np.array_equal(replay_old.states, replay_new.states):
            diff_indices = np.where(replay_old.states != replay_new.states)[0]
            pytest.fail(f"States differ at indices: {diff_indices}\n"
                       f"Old: {replay_old.states[diff_indices]}\n"
                       f"New: {replay_new.states[diff_indices]}")

        if not np.array_equal(replay_old.actions, replay_new.actions):
            diff_indices = np.where(replay_old.actions != replay_new.actions)[0]
            pytest.fail(f"Actions differ at indices: {diff_indices}\n"
                       f"Old: {replay_old.actions[diff_indices]}\n"
                       f"New: {replay_new.actions[diff_indices]}")

        if not np.array_equal(replay_old.rewards, replay_new.rewards):
            diff_indices = np.where(replay_old.rewards != replay_new.rewards)[0]
            pytest.fail(f"Rewards differ at indices: {diff_indices}\n"
                       f"Old: {replay_old.rewards[diff_indices]}\n"
                       f"New: {replay_new.rewards[diff_indices]}")

        if not np.array_equal(replay_old.dones, replay_new.dones):
            diff_indices = np.where(replay_old.dones != replay_new.dones)[0]
            pytest.fail(f"Dones differ at indices: {diff_indices}\n"
                       f"Old: {replay_old.dones[diff_indices]}\n"
                       f"New: {replay_new.dones[diff_indices]}")

        if not np.array_equal(replay_old.next_states, replay_new.next_states):
            diff_mask = replay_old.next_states != replay_new.next_states
            diff_indices = np.where(diff_mask)[0]
            pytest.fail(f"Next states differ at indices: {diff_indices}\n"
                       f"Old: {replay_old.next_states[diff_indices]}\n"
                       f"New: {replay_new.next_states[diff_indices]}")


class TestReplaySampling:
    """Tests to ensure sampling returns correct shapes and identical values"""

    @pytest.fixture
    def setup_filled_replays(self):
        """Setup both replay buffers with some data already added"""
        capacity = 1000
        obs_shape = 42
        num_transitions = 500

        replay_new = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)

        np.random.seed(123)

        # Add 500 transitions to both
        for i in range(num_transitions):
            state = np.random.randn(obs_shape).astype(np.float32)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(obs_shape).astype(np.float32)
            terminal = np.random.rand() < 0.1
            truncation = np.random.rand() < 0.05

            # Add to old
            trans = Transition(
                state=state,
                action=action,
                reward=reward,
                new_state=None if (terminal or truncation) else next_state
            )
            replay_old.add_experince(trans)

            # Add to new
            replay_new.add_experince(
                states=np.array([state]),
                actions=np.array([action]),
                rewards=np.array([reward]),
                new_states=np.array([next_state]),
                terminals=np.array([terminal]),
                truncations=np.array([truncation])
            )

        return replay_new, replay_old, obs_shape

    def test_sample_size_too_small(self):
        """Test that sampling returns False when buffer is too small"""
        capacity = 100
        obs_shape = 42
        replay_new = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)

        # Try to sample 64 when buffer is empty
        result_old = replay_old.sample(64)
        result_new = replay_new.sample(64)

        # Both should return False for all values
        assert result_old == (False, False, False, False, False)
        assert result_new == (False, False, False, False, False)

    def test_sample_return_shapes(self, setup_filled_replays):
        """Test that sampling returns correct shapes"""
        replay_new, replay_old, obs_shape = setup_filled_replays
        batch_size = 64

        # Sample from both
        states_old, actions_old, rewards_old, next_states_old, dones_old = replay_old.sample(batch_size)
        states_new, actions_new, rewards_new, next_states_new, dones_new = replay_new.sample(batch_size)

        # Check shapes for old replay
        assert states_old.shape == (batch_size, obs_shape), f"Old states shape: {states_old.shape}"
        assert actions_old.shape == (batch_size,), f"Old actions shape: {actions_old.shape}"
        assert rewards_old.shape == (batch_size,), f"Old rewards shape: {rewards_old.shape}"
        assert next_states_old.shape == (batch_size, obs_shape), f"Old next_states shape: {next_states_old.shape}"
        assert dones_old.shape == (batch_size,), f"Old dones shape: {dones_old.shape}"

        # Check shapes for new replay
        assert states_new.shape == (batch_size, obs_shape), f"New states shape: {states_new.shape}"
        assert actions_new.shape == (batch_size,), f"New actions shape: {actions_new.shape}"
        assert rewards_new.shape == (batch_size,), f"New rewards shape: {rewards_new.shape}"
        assert next_states_new.shape == (batch_size, obs_shape), f"New next_states shape: {next_states_new.shape}"
        assert dones_new.shape == (batch_size,), f"New dones shape: {dones_new.shape}"

    def test_sample_dtypes(self, setup_filled_replays):
        """Test that sampling returns correct data types"""
        replay_new, replay_old, obs_shape = setup_filled_replays
        batch_size = 32

        # Sample from both
        states_old, actions_old, rewards_old, next_states_old, dones_old = replay_old.sample(batch_size)
        states_new, actions_new, rewards_new, next_states_new, dones_new = replay_new.sample(batch_size)

        # Check dtypes for old replay
        assert states_old.dtype == np.float32, f"Old states dtype: {states_old.dtype}"
        assert actions_old.dtype == np.int64, f"Old actions dtype: {actions_old.dtype}"
        assert rewards_old.dtype == np.float32, f"Old rewards dtype: {rewards_old.dtype}"
        assert next_states_old.dtype == np.float32, f"Old next_states dtype: {next_states_old.dtype}"
        assert dones_old.dtype == np.float32, f"Old dones dtype: {dones_old.dtype}"

        # Check dtypes for new replay
        assert states_new.dtype == np.float32, f"New states dtype: {states_new.dtype}"
        assert actions_new.dtype == np.int64, f"New actions dtype: {actions_new.dtype}"
        assert rewards_new.dtype == np.float32, f"New rewards dtype: {rewards_new.dtype}"
        assert next_states_new.dtype == np.float32, f"New next_states dtype: {next_states_new.dtype}"
        assert dones_new.dtype == np.float32, f"New dones dtype: {dones_new.dtype}"

    def test_sample_values_match(self, setup_filled_replays):
        """Test that sampling returns identical values when using same seed"""
        replay_new, replay_old, obs_shape = setup_filled_replays
        batch_size = 64

        # Set same random seed for both
        np.random.seed(42)
        states_old, actions_old, rewards_old, next_states_old, dones_old = replay_old.sample(batch_size)

        np.random.seed(42)
        states_new, actions_new, rewards_new, next_states_new, dones_new = replay_new.sample(batch_size)

        # Check that sampled values are identical
        assert np.array_equal(states_old, states_new), "Sampled states differ"
        assert np.array_equal(actions_old, actions_new), "Sampled actions differ"
        assert np.array_equal(rewards_old, rewards_new), "Sampled rewards differ"
        assert np.array_equal(next_states_old, next_states_new), "Sampled next_states differ"
        assert np.array_equal(dones_old, dones_new), "Sampled dones differ"

    def test_sample_different_batch_sizes(self, setup_filled_replays):
        """Test sampling with various batch sizes"""
        replay_new, replay_old, obs_shape = setup_filled_replays

        batch_sizes = [1, 16, 32, 64, 128, 256]

        for batch_size in batch_sizes:
            # Sample from old
            states_old, actions_old, rewards_old, next_states_old, dones_old = replay_old.sample(batch_size)

            # Sample from new
            states_new, actions_new, rewards_new, next_states_new, dones_new = replay_new.sample(batch_size)

            # Check shapes match
            assert states_old.shape == states_new.shape == (batch_size, obs_shape)
            assert actions_old.shape == actions_new.shape == (batch_size,)
            assert rewards_old.shape == rewards_new.shape == (batch_size,)
            assert next_states_old.shape == next_states_new.shape == (batch_size, obs_shape)
            assert dones_old.shape == dones_new.shape == (batch_size,)

    def test_sample_indices_in_range(self, setup_filled_replays):
        """Test that sampled actions are valid indices"""
        replay_new, replay_old, obs_shape = setup_filled_replays
        batch_size = 100

        # Sample multiple times
        for _ in range(10):
            _, actions_old, _, _, _ = replay_old.sample(batch_size)
            _, actions_new, _, _, _ = replay_new.sample(batch_size)

            # Actions should be in valid range (0-6 for Connect 4)
            assert np.all(actions_old >= 0), "Old replay has negative actions"
            assert np.all(actions_old < 7), "Old replay has actions >= 7"
            assert np.all(actions_new >= 0), "New replay has negative actions"
            assert np.all(actions_new < 7), "New replay has actions >= 7"

    def test_sample_dones_are_binary(self, setup_filled_replays):
        """Test that dones are either 0 or 1"""
        replay_new, replay_old, obs_shape = setup_filled_replays
        batch_size = 100

        # Sample multiple times
        for _ in range(10):
            _, _, _, _, dones_old = replay_old.sample(batch_size)
            _, _, _, _, dones_new = replay_new.sample(batch_size)

            # Dones should be 0 or 1
            assert np.all((dones_old == 0) | (dones_old == 1)), "Old replay has non-binary dones"
            assert np.all((dones_new == 0) | (dones_new == 1)), "New replay has non-binary dones"

    def test_sample_after_wraparound(self):
        """Test sampling after buffer wraps around"""
        capacity = 100
        obs_shape = 42
        replay_new = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)

        np.random.seed(456)

        # Add more than capacity to trigger wraparound
        for i in range(150):
            state = np.random.randn(obs_shape).astype(np.float32)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(obs_shape).astype(np.float32)
            terminal = np.random.rand() < 0.1
            truncation = np.random.rand() < 0.05

            # Add to old
            trans = Transition(
                state=state,
                action=action,
                reward=reward,
                new_state=None if (terminal or truncation) else next_state
            )
            replay_old.add_experince(trans)

            # Add to new
            replay_new.add_experince(
                states=np.array([state]),
                actions=np.array([action]),
                rewards=np.array([reward]),
                new_states=np.array([next_state]),
                terminals=np.array([terminal]),
                truncations=np.array([truncation])
            )

        # Now sample and check
        batch_size = 50
        np.random.seed(789)
        states_old, actions_old, rewards_old, next_states_old, dones_old = replay_old.sample(batch_size)

        np.random.seed(789)
        states_new, actions_new, rewards_new, next_states_new, dones_new = replay_new.sample(batch_size)

        # Check they match
        assert np.array_equal(states_old, states_new), "States differ after wraparound"
        assert np.array_equal(actions_old, actions_new), "Actions differ after wraparound"
        assert np.array_equal(rewards_old, rewards_new), "Rewards differ after wraparound"
        assert np.array_equal(dones_old, dones_new), "Dones differ after wraparound"

    def test_sample_size_equals_buffer_size(self, setup_filled_replays):
        """Test sampling when batch_size equals buffer size"""
        replay_new, replay_old, obs_shape = setup_filled_replays

        # Both have size 500 from fixture
        batch_size = 500

        states_old, actions_old, rewards_old, next_states_old, dones_old = replay_old.sample(batch_size)
        states_new, actions_new, rewards_new, next_states_new, dones_new = replay_new.sample(batch_size)

        # Should still work and return correct shapes
        assert states_old.shape == (batch_size, obs_shape)
        assert states_new.shape == (batch_size, obs_shape)


    def test_exact_training_loop_reproduction(self):
        """Test that reproduces the exact code from the training loop"""
        capacity = 10000
        obs_shape = 25
        NUM_ENVS = 256

        replay = ReplayMemory(capacity, obs_shape)
        replay_old = ReplayMemoryPrior(capacity, obs_shape)

        np.random.seed(42)

        # Simulate a few steps of the training loop
        # trange does nothing :()
        for step in trange(1000):
            # Simulate environment observations
            obs = np.random.randn(NUM_ENVS, obs_shape).astype(np.float32)
            actions = np.random.randint(0, 5, size=NUM_ENVS)

            # Line 308: old_obs = obs.copy()
            old_obs = obs.copy()

            # Simulate environment step (line 319)
            obs_new = np.random.randn(NUM_ENVS, obs_shape).astype(np.float32)
            rewards = np.random.randn(NUM_ENVS).astype(np.float32)
            terminals = np.random.rand(NUM_ENVS) < 0.1
            truncations = np.random.rand(NUM_ENVS) < 0.05

            # Lines 325-338: Add to old replay one by one
            for n in range(NUM_ENVS):
                new_state = obs_new[n]

                if terminals[n] or truncations[n]:
                    new_state = None

                trans = Transition(state=old_obs[n], action=actions[n], reward=rewards[n], new_state=new_state)
                replay_old.add_experince(trans)

            # Line 339: Add to new replay in batch
            replay.add_experince(old_obs.copy(), actions.copy(), rewards.copy(), obs_new.copy(), terminals.copy(), truncations.copy())

            # Lines 344-365: Check equality (this is where it fails)
            if not np.array_equal(replay_old.states, replay.states):
                diff_mask = replay_old.states != replay.states
                diff_indices = np.where(diff_mask)[0]
                print(f"\nStep {step}: State Arrays differ at indices: {diff_indices[:10]}")
                print(f"Replay old: position={replay_old.position}, size={replay_old.size}")
                print(f"Replay new: position={replay.position}, size={replay.size}")
                print(f"Old values at first diff: {replay_old.states[diff_indices[0]][:5]}")
                print(f"New values at first diff: {replay.states[diff_indices[0]][:5]}")
                pytest.fail(f"States differ at step {step}")

            assert np.array_equal(replay_old.states, replay.states)
            assert np.array_equal(replay_old.rewards, replay.rewards)
            assert np.array_equal(replay_old.dones, replay.dones)

            if not np.array_equal(replay_old.next_states, replay.next_states):
                diff_mask = replay_old.next_states != replay.next_states
                diff_indices = np.where(diff_mask)[0]
                print(f"\nStep {step}: Next State Arrays differ at indices: {diff_indices[:10]}")
                print(f"Old values at first diff: {replay_old.next_states[diff_indices[0]][:5]}")
                print(f"New values at first diff: {replay.next_states[diff_indices[0]][:5]}")
                pytest.fail(f"Next states differ at step {step}")


    def test_no_input_mutation(self):
        """Test that add_experince does not mutate input arrays"""
        capacity = 1000
        obs_shape = 25
        num_envs = 10

        replay = ReplayMemory(capacity, obs_shape)

        # Create test data
        states = np.random.randn(num_envs, obs_shape).astype(np.float32)
        actions = np.random.randint(0, 5, size=num_envs)
        rewards = np.random.randn(num_envs).astype(np.float32)
        next_states = np.random.randn(num_envs, obs_shape).astype(np.float32)
        terminals = np.zeros(num_envs, dtype=bool)
        terminals[0] = True  # First env terminates
        truncations = np.zeros(num_envs, dtype=bool)
        truncations[1] = True  # Second env truncates

        # Save original values
        original_states = states.copy()
        original_actions = actions.copy()
        original_rewards = rewards.copy()
        original_next_states = next_states.copy()
        original_terminals = terminals.copy()
        original_truncations = truncations.copy()

        # Add to replay
        replay.add_experince(states, actions, rewards, next_states, terminals, truncations)

        # Check that inputs were not mutated
        assert np.array_equal(states, original_states), "states was mutated!"
        assert np.array_equal(actions, original_actions), "actions was mutated!"
        assert np.array_equal(rewards, original_rewards), "rewards was mutated!"
        assert np.array_equal(next_states, original_next_states), "next_states was mutated!"
        assert np.array_equal(terminals, original_terminals), "terminals was mutated!"
        assert np.array_equal(truncations, original_truncations), "truncations was mutated!"


if __name__ == "__main__":
    # Can run tests directly with pytest
    pytest.main([__file__, "-v"])
