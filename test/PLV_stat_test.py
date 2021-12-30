# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy

# Sample size
N = 1500

# Number of samples
M = 3000

phi = 2 * np.pi * np.random.rand(M, N)
z = np.exp(1j * phi)

x = np.mean(z, axis=1)

xre = np.real(x)
xim = np.imag(x)

p_xre, bins = np.histogram(xre, bins=50, density=True)
#p_xre = p_xre / (bins[1] - bins[0])

sigma = 1 / np.sqrt(2 * N)
#sigma = np.std(xre)

t = np.linspace(-0.2, 0.2, 1000)
#p_xre_hat = sigma * np.random.normal(t)
p_xre_hat = scipy.stats.norm.pdf(t, scale=sigma)

#plt.figure()
#plt.plot(bins[1:], p_xre)
#plt.plot(t, p_xre_hat)


a = np.abs(x)

p_a, bins = np.histogram(a, bins=50, density=True)

sigma_a = sigma * np.sqrt(2 - np.pi / 2)
mu_a = sigma * np.sqrt(np.pi / 2)
p_a_hat = scipy.stats.norm.pdf(t - mu_a, scale=sigma_a)

plt.figure()
plt.plot(bins[1:], p_a)
plt.plot(t, p_a_hat)




