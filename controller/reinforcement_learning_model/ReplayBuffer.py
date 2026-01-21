import random
import numpy as np

class ReplayBuffer(object):
    """A replay buffer that stores windows of trajectories."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def put(self, window): # <--- ÚNICO CAMBIO
        """Put a trajectory window into the replay buffer.
        The window is expected to be a tuple of arrays (e.g., stacked obs, actions, etc.).
        The oldest elements inside the replay buffer should be overwritten first.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = window # Se guarda la ventana completa
        self.position = (self.position + 1) % self.capacity

#     def get(self, batch_size):
#         """Gives batch_size random window samples from the replay buffer."""
#         batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
#         # Esta línea ahora desempaqueta y apila las ventanas completas
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
        cols  = list(zip(*batch))  # lista de columnas (una por campo)
        stacked = []

        for name, col in zip(names, cols):
            # Normaliza a np.array para medir shapes de forma coherente
            arrays = [np.asarray(x) for x in col]
            shapes = [a.shape for a in arrays]

            # ¿Todas iguales?
            if len(set(shapes)) != 1:
                ref = shapes[0]
                bad_pos = [i for i, s in enumerate(shapes) if s != ref]

                # (opcional) localizar índice global en el buffer de cada ventana problemática
                try:
                    bad_global = [self.buffer.index(batch[i]) for i in bad_pos]
                except Exception:
                    bad_global = ["?"] * len(bad_pos)

                msg = (
                    f"[ReplayBuffer.get] Falló al apilar '{name}'.\n"
                    f"  Forma esperada (ref): {ref}\n"
                    f"  Formas en el batch: {shapes}\n"
                    f"  Ventanas problemáticas -> "
                    f"(batch_idx, shape, buffer_idx): "
                    f"{[(i, shapes[i], bad_global[k]) for k, i in enumerate(bad_pos)]}"
                )
                raise ValueError(msg)

            stacked.append(np.stack(arrays))

        return tuple(stacked)

    def __len__(self):
        """Returns the number of windows inside the replay buffer."""
        return len(self.buffer)