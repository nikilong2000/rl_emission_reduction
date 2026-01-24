import numpy as np

class NormalActionNoise():
    """
    A Gaussian action noise

    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """

    def __init__(self, mean=0, sigma=0.2):
        super().__init__()
        self._mu = mean
        self._sigma = sigma

    def sample(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)

