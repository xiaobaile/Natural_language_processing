import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def draw_likelihood(observations, mu, sigma):
    plt.ylim(-0.02, 1)
    x_locs = np.linspace(-10, 10, 500)
