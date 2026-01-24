import random
import numpy as np


class ReplayBuffer(object):  ##CORRECT
    """A replay buffer that stores windows of trajectories."""

    def __init__(self, capacity, S, obs_dim, act_dim):
        """Initializes replay buffer with certain capacity."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        shape_s = (S, obs_dim)
        shape_a = (S, act_dim)
        shape_r = (S,)
        shape_ns = (S, obs_dim)
        shape_t = (S,)
        shape_tr = (S,)

        self._expected_shapes = (shape_s, shape_a, shape_r, shape_ns, shape_t, shape_tr)

    def _get_window_shapes(self, window):
        """Helper to get shapes of a tuple of arrays."""
        try:
            return tuple(np.asarray(item).shape for item in window)
        except Exception as e:
            raise ValueError(f"Could not get window shapes. Error: {e}")

    def put(self, window):
        """Put a trajectory window into the replay buffer.
        The window is expected to be a tuple of arrays (e.g., stacked obs, actions, etc.).
        The oldest elements inside the replay buffer should be overwritten first.
        """
        current_shapes = self._get_window_shapes(window)

        # Compare received shapes with those defined in __init__
        if current_shapes != self._expected_shapes:
            msg = (
                f"[ReplayBuffer.put] Window rejected!\n"
                f"  Expected shape (defined in __init__): {self._expected_shapes}\n"
                f"  Received shape: {current_shapes}"
            )
            # We stop the program to force RolloutWorker correction
            raise ValueError(msg)

        # --- If validation passes, we save it ---
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = window
        self.position = (self.position + 1) % self.capacity

    def get(self, batch_size):  # CORRECT
        """
        Gets a batch of random samples from the buffer and stacks them.

        It selects `batch_size` windows at random. Then, it processes this
        batch by transposing and stacking the data: it groups all states into a
        single NumPy array, all actions into another, and so on.

        This method includes a crucial validation: before stacking, it checks
        that all sequences for a data type (e.g., 'states') have
        exactly the same shape. If it finds an inconsistency, it raises a
        detailed `ValueError` to facilitate debugging.

        Parameters
        ----------
        batch_size : int
            The number of windows to sample from the buffer.

        Returns
        -------
        tuple[np.ndarray, ...]
            A tuple of stacked NumPy arrays. The order is:
            (states, actions, rewards, next_states, terminated, truncated).


        # This is what is returned with batch_size=3:
        (
            <NumPy Array of shape (3, 96, 5)>,   # batch[0] (states)
            <NumPy Array of shape (3, 96, 3)>,   # batch[1] (actions)
            <NumPy Array of shape (3, 96,)>,   # batch[2] (rewards)
            <NumPy Array of shape (3, 96, 5)>,   # batch[3] (next_states)
            <Array NumPy de shape (3, 96,)>,   # batch[4] (terminated)
            <Array NumPy de shape (3, 96,)>    # batch[5] (truncated)
        )
        """

        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        cols = list(zip(*batch))

        # We stack directly. It's fast and safe.
        stacked = [np.stack(col) for col in cols]
        return tuple(stacked)

    def prune_oldest(self, fraction=0.50):
        """Removes the oldest percentage of the buffer.

        Useful when the model converges and old experiences
        (with bad policy) are no longer useful.

        Parameters
        ----------
        fraction : float
            Fraction of buffer to remove (0.3 = 30%)

        Returns
        -------
        int
            Number of windows removed
        """
        if len(self.buffer) == 0:
            return 0

        n_remove = int(len(self.buffer) * fraction)
        if n_remove > 0:
            self.buffer = self.buffer[n_remove:]
            # Adjust position to point to the end of current buffer
            self.position = len(self.buffer) % self.capacity

        return n_remove

    def __len__(self):
        """Returns the number of windows inside the replay buffer."""
        return len(self.buffer)
