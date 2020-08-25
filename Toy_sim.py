import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ToySim_class as TSim_class
import model as mod
import policy
T = 30
beta = 0.6
gamma = 0.25
alpha = 0.8
nc = 4
bw = 0.15
p_inf = 0.01
p_rec = 0.001
c=1
np.random.seed(c)
locs = np.random.rand(nc, 2)

def NVac_const(t):
	return 50

def NVac_t(t):
	return 3*t + 80

vacfun = NVac_const

ii = 1
rewlist = []
sim = TSim_class.TSimulator(locs=locs, ii=ii, nc=nc, bw=bw, T=T, beta=beta, gamma=gamma, alpha=alpha, p_inf=p_inf,
                            p_rec=p_rec, c=c)

for m in range(4):
	sim.reset()
	mS = np.zeros(sim.S.shape)
	mI = np.zeros(sim.I.shape)
	mR = np.zeros(sim.R.shape)
	mV = np.zeros(sim.V.shape)

	Model = mod.model(sim, NVac = vacfun(0))
	if m == 0:
		null_policy = policy.null_policy(Model)
	elif m == 1:
		even_alloc = policy.even_allocate(Model)
	elif m==2:
		sim_model = mod.det_model_trainer(sim, NVac=vacfun(0))

		VFA_linear_policy = policy.VFA_linear_policy(Model, theta=VFA_trainer.theta)
	elif m==3:
		sim_modelt = mod.det_model_trainer(sim, NVac=vacfun(0))

		VFA_lineart_policy = policy.VFA_linear_t_policy(Model, theta=VFA_trainert.theta)
	cumrew = 0
	cumlist = []
	for t in range(T-1):
		if m==0:
			null_policy.update(Model)
			x = null_policy.decision()
		elif m==1:
			even_alloc.update(Model)
			x = even_alloc.decision()
		elif m==2:
			VFA_linear_policy.update(Model)
			x = VFA_linear_policy.decision(t)
		elif m==3:
			VFA_lineart_policy.update(Model)
			x = VFA_lineart_policy.decision(t)
		S,I,R,V = sim.forward_one_step(x, t)
		mS[:, t] = Model.state["Sbar"]
		mI[:, t] = Model.state["Ibar"]
		mR[:, t] = Model.state["Rbar"]
		mV[:, t] = x
		mean_pbar = Model.forward_one_step(x,sim,vacfun(t+1),t)

		cumrew+=mean_pbar
		cumlist.append(cumrew)

	rewlist.append(cumlist)

sim.plot_totals(vacfun(np.arange(0,T)))
I = 2
J = 2
sim.plot_county_grid(I,J)
# Model.plotcounties(mS,mI,mR,mV,I,J)
plt.figure()
sim.plot_net(ii)
plt.figure()
plt.plot(rewlist[0])
plt.plot(rewlist[1])
plt.plot(rewlist[2])
plt.plot(rewlist[3])
plt.legend(['Null Policy', 'Even Allocation', 'Linear VFA', 'time-dependent Linear VFA'])

plt.show()
# anim = animation.ArtistAnimation(FIRE.fig, FIRE.ims, interval=1000)