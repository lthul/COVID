import numpy as np
import matplotlib.pyplot as plt
import Sim_class
import data_loader as dl

T = 10
beta = 0.9
gamma = 0.3
alpha = 0.8
n_seeds = 100
p_inf = 0.01
p_rec = 0.001

sim = Sim_class.Simulator(T=T, beta=beta, gamma=gamma, alpha=alpha, n_seeds=n_seeds, p_inf=p_inf, p_rec=p_rec)

c = np.argmax(sim.I[:,0])
resp = []
for t in range(T-1):

	if t == 5:
		x = np.zeros(sim.N.shape)
		FLOWn = np.random.binomial(np.int32(0.2 * sim.Nvals.T), sim.FLOW_mat)
		for xx in np.arange(0, sim.N[c], np.int32(sim.N[c] / 7)):
			x[c] = xx
			sim.reset_t(t)
			S,I,R,V = sim.forward_one_step(x, t, FLOWn=FLOWn)
			print('x = ' + str(xx) + ' ... ')
			rt.append(I[c])
	else:
		x = np.zeros(sim.N.shape)


sim.plot_totals()
plt.figure()
xxx = np.arange(0, sim.N[c], np.int32(sim.N[c] / 7))
plt.plot(xxx, np.array(resp)[0, :])
plt.plot(xxx, np.array(resp)[1, :])
plt.plot(xxx, np.array(resp)[2, :])
plt.plot(xxx, np.array(resp)[3, :])
plt.plot(xxx, np.array(resp)[5, :])
plt.plot(xxx, np.array(resp)[7, :])


plt.show()