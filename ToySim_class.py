import data_loader as dl
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
class TSimulator:
	def __init__(self, locs, ii=3, T=25, bw=0.1, nc=10, beta = 0.9, gamma = 0.3, alpha = 0.8, p_inf = 0.01, p_rec = 0.00, c=1):
		# T = time horizon
		# nc = number of counties
		# beta = the transmission rate of the virus (expected number of interactions between infected-susc persons)
		# gamma = the recovery rate of the virus (1 / time of recovery)
		# alpha = the mobility rate of people traveling
		# n_seeds is the number of infected counties to start with
		# p_inf = percent of infected people to initialize per county
		# p_rec = percent of recovered people to initialize per county

		maxpop = 1000
		mixing_bandwidth = bw
		self.N = np.random.randint(maxpop, size=nc)
		self.ii = ii
		self.locs = locs
		self.bw = bw
		self.nc = nc
		self.T = T
		self.beta = beta
		self.gamma = gamma
		self.alpha = alpha
		self.p_inf = p_inf
		self.p_rec = p_rec
		self.c = c
		#
		self.FLOW = np.exp(-0.5 * pdist(self.locs / mixing_bandwidth, 'sqeuclidean'))
		self.FLOW = squareform(self.FLOW)
		self.FLOW += 0.001

		# county populations

		np.random.seed(self.c)

		# initialize SIR population to total populations for S and 0 IR
		self.S = np.zeros([nc, T])
		self.I = np.zeros([nc, T])
		self.R = np.zeros([nc, T])
		self.V = np.zeros([nc, T])

		self.pct = np.max([0.01, p_inf * np.random.rand()])
		self.I[ii, 0] = np.int32(self.pct * self.N[ii])
		self.R[:, 0] = np.int32(p_rec * self.N)
		self.S[:, 0] = self.N - self.I[:, 0]

		# initialize model parameters

		# all beta and gamma values are constant throughout time and space
		# self.beta_ = beta * np.ones(self.S.shape)
		# self.gamma_ = gamma * np.ones(self.S.shape)
		self.beta_ = beta * np.random.normal(self.beta, scale=.001, size=self.S.shape)
		self.gamma_ = gamma * np.random.normal(self.gamma, scale=.001, size=self.S.shape)
		self.alpha_ = alpha

		self.Nvals = self.N.repeat(self.nc).reshape(self.nc, self.nc)
		self.FLOWn = None
		# initialize the epidemic to have starter infections

	def reset(self):
		self.S = np.zeros([self.nc, self.T])
		self.I = np.zeros([self.nc, self.T])
		self.R = np.zeros([self.nc, self.T])
		self.V = np.zeros([self.nc, self.T])

		self.I[self.ii, 0] = np.int32(self.pct * self.N[self.ii])
		self.R[:, 0] = np.int32(self.p_rec * self.N)
		self.S[:, 0] = self.N - self.I[:, 0]

	def forward_one_step(self, x, t, FLOWn=None):
		# x = N length vector of [x_vac]
		if FLOWn is None:
			self.FLOWn = np.random.binomial(np.int32(0.5 * self.Nvals.T), self.FLOW)
		else:
			self.FLOWn = FLOWn

		# lam = 0.1*self.N
		# L = np.random.poisson(lam, self.nc)
		L=0

		self.S[:, t + 1] = np.max([np.zeros(self.nc), self.S[:, t] - (self.beta_[:,t] * self.S[:, t] * self.I[:, t]) / self.N - (
					self.alpha_ * self.S[:, t] * np.dot(self.FLOWn, self.beta_[:,t] * self.I[:, t] / self.N)) / (self.N + np.sum(self.FLOWn, axis=1)) - x - L], axis=0)
		self.S[:, t + 1] = np.min([self.S[:, t + 1], self.N], axis=0)
		self.I[:, t + 1] = np.max([np.zeros(self.nc), (1 - self.gamma_[:,t]) * self.I[:, t] + (self.beta_[:,t] * self.S[:, t] * self.I[:, t]) / self.N + (
					self.alpha_ * self.S[:, t] * np.dot(self.FLOWn, self.beta_[:,t] * self.I[:, t] / self.N)) / (self.N + np.sum(self.FLOWn, axis=1)) + L], axis=0)
		self.I[:, t + 1] = np.min([self.I[:, t + 1], self.N], axis=0)
		self.R[:, t + 1] = np.max([np.zeros(self.nc), self.R[:, t] + self.gamma_[:,t] * self.I[:, t] + x], axis=0)
		self.R[:, t + 1] = np.min([self.R[:, t + 1], self.N], axis=0)
		self.V[:, t] = np.min([x, self.N], axis=0)
		# Np = np.sum(self.S[:, t + 1]) + np.sum(self.I[:, t + 1]) + np.sum(self.R[:, t + 1])
		return self.S[:,t+1], self.I[:,t+1], self.R[:,t+1], self.V[:,t+1]

	def getFLOWn(self):
		if self.FLOWn is not None:
			return self.FLOWn
		else:
			return None

	def plot_totals(self, NVac):
		plt.plot(np.sum(self.S, axis=0), 'b')
		plt.plot(np.sum(self.I, axis=0), 'r')
		plt.plot(np.sum(self.R, axis=0), 'g')
		plt.plot(np.sum(self.V, axis=0), 'k')
		plt.plot(NVac * np.ones(self.T), 'c--')
		plt.legend(["Susc", "Inf", "Rec", "Vac", "Vaccine Load"])


	def plot_county(self, idx):
		plt.plot(self.S[idx, :], 'b')
		plt.plot(self.I[idx, :], 'r')
		plt.plot(self.R[idx, :], 'g')
		plt.plot(self.V[idx, :], 'k')
		plt.legend(["Susc", "Inf", "Rec", "Vac"])

	def plot_county_grid(self, I, J):
		fig, ax = plt.subplots(I,J)
		if I>1:
			for i in np.arange(0,I):
				for j in np.arange(0,J):
					if ((J*i+j)<self.S.shape[0]):
						ax[i][j].plot(self.S[J*i + j, :], 'b')
						ax[i][j].plot(self.I[J*i + j, :], 'r')
						ax[i][j].plot(self.R[J*i + j, :], 'g')
						ax[i][j].plot(self.V[J*i + j, :], 'k')




		# plt.legend(["Susc", "Inf", "Rec", "Vac"])



	def plot_net(self, ii=None):
		plt.plot(self.locs[:, 0], self.locs[:, 1], 'c*')
		if ii is not None:
			plt.plot(self.locs[ii, 0], self.locs[ii, 1], 'r*')
		for i in range(self.nc):
			for j in np.arange(i, self.nc):
				alpha = (self.FLOW[i, j] - np.min(self.FLOW)) / (np.max(self.FLOW) - np.min(self.FLOW))
				alpha = np.min([1, alpha + 0.01])
				plt.plot([self.locs[i, 0], self.locs[j, 0]], [self.locs[i, 1], self.locs[j, 1]], 'b', alpha=alpha)




	def getS(self,t):
		return self.S[:,t]

	def getI(self,t):
		return self.I[:,t]

	def getR(self,t):
		return self.R[:,t]

	def getV(self,t):
		return self.V[:,t]