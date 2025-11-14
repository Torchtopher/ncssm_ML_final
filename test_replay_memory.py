import numpy as np
import pytest
from collections import namedtuple
from connect4_train import ReplayMemory, ReplayMemoryPrior, Transition


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


if __name__ == "__main__":
    # Can run tests directly with pytest
    pytest.main([__file__, "-v"])
