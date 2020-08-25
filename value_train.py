import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ToySim_class as TSim_class
import model as mod
import policy
from mpl_toolkits import mplot3d

T = 30
beta = 0.9
gamma = 0.25
alpha = 0.8
nc = 3
bw = 0.15
p_inf = 0.01
p_rec = 0.001
np.random.seed(1)
locs = np.random.rand(nc, 2)
NVac = 50
ii=1
t = 5

sim = TSim_class.TSimulator(locs=locs, ii=ii, nc=nc, bw=bw, T=T, beta=beta, gamma=gamma, alpha=alpha, p_inf=p_inf,
                            p_rec=p_rec)
Model = mod.model(sim, NVac=NVac)
VFA_linear_policy = policy.VFA_linear_trainer(Model, N=10, M=6)
VFA_linear_policy.train()


# print(VFA_linear_policy.theta[t,:])

# disc = 5
# nc2 = 2*sim.nc
# points = [np.arange(0, sim.N[c], sim.N[c] / disc) for c in range(sim.nc)]
# points2 = [np.arange(0, sim.N[c], sim.N[c] / disc) for c in range(sim.nc)]
# points.extend(points2)
# Dspace = np.vstack(np.meshgrid(*points))
# Dspace = Dspace.reshape(nc2,np.int32(np.product(Dspace.shape)/nc2))
# bools = np.sum(Dspace,axis=0) <= np.sum(sim.N)
# fDspace = Dspace[:,bools]
# Sspace = [Dspace[:, i] for i in range(Dspace.shape[1])]
# pairs = []
# for ss in Sspace:
#
# 	v = np.dot(VFA_linear_policy.theta[t,:], ss)
# 	pairs.append(np.hstack([ss,v]))
#
# A = np.array(pairs)
#
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(A[:,0], A[:,3], A[:,nc2], cmap='Greens')
# ax.set_xlabel('1I')
# ax.set_ylabel('1S')
# ax.set_zlabel('V')
#
# plt.show()


