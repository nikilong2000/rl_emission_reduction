import random
import numpy as np


class ReplayBuffer(object):
    """A replay buffer that stores windows of trajectories."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def put(self, window):  # <--- SINGLE CHANGE
        """Put a trajectory window into the replay buffer.
        The window is expected to be a tuple of arrays (e.g., stacked obs, actions, etc.).
        The oldest elements inside the replay buffer should be overwritten first.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = window  # The complete window is saved
        self.position = (self.position + 1) % self.capacity

    #     def get(self, batch_size):
    #         """Gives batch_size random window samples from the replay buffer."""
    #         batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
    #         # This line now unpacks and stacks the complete windows
    #         state, action, reward, next_state, terminated, truncated = map(np.stack, zip(*batch))
    #         return state, action, reward, next_state, terminated, truncated

    def get(self, batch_size):
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

        Raises
        ------
        ValueError
            If the shapes of the sequences within the batch are not consistent
            and cannot be stacked correctly.
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))

        names = ["state", "action", "reward", "next_state", "terminated", "truncated"]
        cols = list(zip(*batch))  # list of columns (one per field)
        stacked = []

        for name, col in zip(names, cols):
            # Normalize to np.array to measure shapes coherently
            arrays = [np.asarray(x) for x in col]
            shapes = [a.shape for a in arrays]

            # All equal?
            if len(set(shapes)) != 1:
                ref = shapes[0]
                bad_pos = [i for i, s in enumerate(shapes) if s != ref]

                # (optional) locate global index in the buffer of each problematic window
                try:
                    bad_global = [self.buffer.index(batch[i]) for i in bad_pos]
                except Exception:
                    bad_global = ["?"] * len(bad_pos)

                msg = (
                    f"[ReplayBuffer.get] Failed to stack '{name}'.\n"
                    f"  Expected shape (ref): {ref}\n"
                    f"  Shapes in batch: {shapes}\n"
                    f"  Problematic windows -> "
                    f"(batch_idx, shape, buffer_idx): "
                    f"{[(i, shapes[i], bad_global[k]) for k, i in enumerate(bad_pos)]}"
                )
                raise ValueError(msg)

            stacked.append(np.stack(arrays))

        return tuple(stacked)

    def __len__(self):
        """Returns the number of windows inside the replay buffer."""
        return len(self.buffer)
