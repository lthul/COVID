import numpy as np
import model as mod
import matplotlib.pyplot as plt
import ToySim_class as Tsim
import gurobi as gp
from gurobipy import GRB

class null_policy:
	def __init__(self, model):
		self.model = model

	def update(self, model_new):
		self.model = model_new

	def decision(self):
		x = np.zeros(self.model.sim.nc)
		return x

class even_allocate:
	def __init__(self, model):
		self.model = model

	def update(self, model_new):
		self.model = model_new

	def decision(self):
		Nvac = self.model.NVac
		nc = self.model.sim.nc
		xi = np.floor(Nvac/nc)
		rem = Nvac % nc
		x = xi * np.ones(nc)
		x[:rem] +=1

		return x

class VFA_linear_t_trainer:
	def __init__(self, model_trainer, N=10, M =10):
		self.model_trainer = model_trainer
		self.N = N
		self.M = M
		self.gg = 1.0
		# Tsim.TSimulator(locs = model.locs, ii = model.ii, nc = model.nc, bw = model.bw, T = model.T, beta = model.beta, gamma = model.gamma, alpha = model.alpha, p_inf = model.p_inf, p_rec = model.p_rec)

		self.Stnm = [[[{} for _ in range(M)] for _ in range(N)] for _ in range(self.model_trainer.sim.T+1)]
		for m in range(self.M):
			for n in range(self.N):
				self.Stnm[0][n][m] = self.model_trainer.state

		#recursive LS parameters
		self.theta = np.random.rand(model_trainer.sim.T, 2 * model_trainer.sim.nc)
		self.ll = 0.5
		self.B = [self.ll*np.eye(self.theta.shape[1]) + 1e-6 for _ in range(self.model_trainer.sim.T)]

	def Vbar(self, theta, state, t):
		if t >= self.model_trainer.sim.T:
			V = 0
		else:
			V = np.dot(theta[t,:self.model_trainer.sim.nc], state["Ibar"]) + np.dot(theta[t,self.model_trainer.sim.nc:], state["Sbar"])
		return V

	def recursiveLS(self, error, n, m):
		for t in range(self.model_trainer.sim.T):
			phi = np.hstack([self.Stnm[t][n][m]["Ibar"], self.Stnm[t][n][m]["Sbar"]])[:,None]
			gn = self.ll + np.dot(np.dot(phi.T, self.B[t]), phi)
			aa = (1 / gn) * np.dot(self.B[t], phi) * error[t]
			self.theta[t,:] = self.theta[t,:] - aa.reshape(2*self.model_trainer.sim.nc,)
			self.B[t] = self.B[t] - (1/gn) * np.dot(np.dot(self.B[t], np.dot(phi, phi.T)), self.B[t])

	def train(self):
		c = self.model_trainer.sim.c
		for n in range(self.N):
			c+=1
			ii = np.random.randint(self.model_trainer.sim.nc)
			self.model_trainer.sim = Tsim.TSimulator(locs=self.model_trainer.sim.locs, ii=ii, nc=self.model_trainer.sim.nc, bw=self.model_trainer.sim.bw, T=self.model_trainer.sim.T, beta=self.model_trainer.sim.beta,
			                gamma=self.model_trainer.sim.gamma, alpha=self.model_trainer.sim.alpha, p_inf=self.model_trainer.sim.p_inf, p_rec=self.model_trainer.sim.p_rec, c = c)

			for m in range(self.M):
				for t in range(self.model_trainer.sim.T):
					# Vset = []
					# xset = []
					# for xx in self.model_trainer.decision_space:
					# 	state0 = self.Stnm[t][n][m].copy()
					# 	state1 = self.model_trainer.sim_trans(xx, state0)
					# 	Vhat = self.model_trainer.sim_objective(state0, xx) + self.gg* self.Vbar(self.theta, state1, t+1)
					# 	xset.append(xx)
					# 	Vset.append(Vhat)
					# xstar = xset[np.argmin(np.array(Vset))]
					state0 = self.Stnm[t][n][m].copy()
					xstar = self.bellman_solver(state0, t)
					self.Stnm[t+1][n][m] = self.model_trainer.sim_trans(xstar, self.Stnm[t][n][m])
				vhat = np.zeros(self.model_trainer.sim.T+1)
				error = np.zeros(self.model_trainer.sim.T)
				for t in np.arange(self.model_trainer.sim.T, 0, -1):
					vhat[t-1] = self.model_trainer.sim_objective(self.Stnm[t][n][m]) + self.gg * vhat[t]
					error[t-1] = self.Vbar(self.theta, self.Stnm[t-1][n][m],t-1) - vhat[t-1]

				print(np.sum(np.square(error)))
				self.recursiveLS(error, n, m)

				# print(self.theta[5,:])

	def bellman_solver(self, state1, t):
		obx = state1["Sbar"] - (self.model_trainer.sim.beta * state1["Sbar"] * state1["Ibar"]/self.model_trainer.sim.N)
		# w = 1/self.model_trainer.sim.N + self.theta[t, self.model_trainer.sim.nc:]
		w = 1 + self.theta[t, self.model_trainer.sim.nc:]
		m = gp.Model("bellman solver")
		x = m.addMVar(shape=self.model_trainer.sim.nc, vtype=GRB.INTEGER, name="x")

		m.setObjective(w @ x, GRB.MAXIMIZE)

		A = np.ones(self.model_trainer.sim.nc)

		# Build rhs vector
		rhs = np.array([self.model_trainer.NVac])

		# Add constraints
		m.addConstr(A @ x <= rhs, name="c")
		m.addConstr(0 <= x)
		m.addConstr(x <= obx)
		m.optimize()
		return x.X

class VFA_linear_t_policy(VFA_linear_t_trainer):
	def __init__(self, model, theta=None, N=10, M=10):
		self.model = model
		self.gg = 1.0
		if theta is None:
			self.theta = np.random.rand(model.sim.T, 2 * model.sim.nc)
		else:
			self.theta = theta
		self.VFA_trainert = VFA_linear_t_trainer(model_trainer=model.__copy__(), N=N, M=M)
		self.VFA_trainert.train()

	def update(self, model_new):
		self.model = model_new

	def decision(self,t):
		# Vset = []
		# xset = []
		# for xx in self.model.decision_space:
		# 	state1 = self.model.sim_trans(xx, self.model.state)
		# 	Vhat = self.model.sim_objective(state1) + self.gg * self.Vbar(self.theta, state1, t)
		# 	xset.append(xx)
		# 	Vset.append(Vhat)
		# xstar = xset[np.argmin(np.array(Vset))]
		xstar = self.bellman_solver(self.model.state, t)
		return xstar

	def Vbar(self, theta, state, t):
		if t >= self.model.sim.T:
			V = 0
		else:
			V = np.dot(theta[t, :self.model.sim.nc], state["Ibar"]) + np.dot(theta[t, self.model.sim.nc:],
			                                                                 state["Sbar"])
		return V


class VFA_linear_trainer:
	def __init__(self, model_trainer, N=10, M =10):
		self.model_trainer = model_trainer
		self.N = N
		self.M = M
		self.gg = 1.0
		# Tsim.TSimulator(locs = model.locs, ii = model.ii, nc = model.nc, bw = model.bw, T = model.T, beta = model.beta, gamma = model.gamma, alpha = model.alpha, p_inf = model.p_inf, p_rec = model.p_rec)

		self.Stnm = [[[{} for _ in range(M)] for _ in range(N)] for _ in range(self.model_trainer.sim.T+1)]
		for m in range(self.M):
			for n in range(self.N):
				self.Stnm[0][n][m] = self.model_trainer.state

		#recursive LS parameters
		self.theta = np.random.rand(2 * model_trainer.sim.nc)
		self.ll = 0.5
		self.B = self.ll*np.eye(self.theta.shape[0]) + 1e-6

	def Vbar(self, theta, state):
		V = np.dot(theta[:self.model_trainer.sim.nc], state["Ibar"]) + np.dot(theta[self.model_trainer.sim.nc:], state["Sbar"])
		return V

	def recursiveLS(self, error, n, m,t):
		phi = np.hstack([self.Stnm[t][n][m]["Ibar"], self.Stnm[t][n][m]["Sbar"]])[:,None]
		gn = self.ll + np.dot(np.dot(phi.T, self.B), phi)
		aa = (1 / gn) * np.dot(self.B, phi) * error[t]
		self.theta = self.theta - aa.reshape(2*self.model_trainer.sim.nc,)
		self.B = self.B - (1/gn) * np.dot(np.dot(self.B, np.dot(phi, phi.T)), self.B)

	def train(self):
		for n in range(self.N):
			ii = np.random.randint(self.model_trainer.sim.nc)
			self.model_trainer.sim = Tsim.TSimulator(locs=self.model_trainer.sim.locs, ii=ii, nc=self.model_trainer.sim.nc, bw=self.model_trainer.sim.bw, T=self.model_trainer.sim.T, beta=self.model_trainer.sim.beta,
			                gamma=self.model_trainer.sim.gamma, alpha=self.model_trainer.sim.alpha, p_inf=self.model_trainer.sim.p_inf, p_rec=self.model_trainer.sim.p_rec, c = self.model_trainer.sim.c+1)

			for m in range(self.M):
				print(m)
				for t in range(self.model_trainer.sim.T):
					# Vset = []
					# xset = []
					# for xx in self.model_trainer.decision_space:
					# 	state0 = self.Stnm[t][n][m].copy()
					# 	state1 = self.model_trainer.sim_trans(xx, state0)
					# 	Vhat = self.model_trainer.sim_objective(state0, xx) + self.gg* self.Vbar(self.theta, state1)
					# 	xset.append(xx)
					# 	Vset.append(Vhat)
					state0 = self.Stnm[t][n][m].copy()
					xstar = self.bellman_solver(state0)
					self.Stnm[t+1][n][m] = self.model_trainer.sim_trans(xstar, self.Stnm[t][n][m])
				vhat = np.zeros(self.model_trainer.sim.T+1)
				error = np.zeros(self.model_trainer.sim.T)
				for t in np.arange(self.model_trainer.sim.T, 0, -1):
					vhat[t-1] = self.model_trainer.sim_objective(self.Stnm[t][n][m]) + self.gg * vhat[t]
					error[t-1] = self.Vbar(self.theta, self.Stnm[t-1][n][m]) - vhat[t-1]
					# print(np.sum(np.square(error)))
					self.recursiveLS(error, n, m, t-1)

				# print(self.theta[5,:])

	def bellman_solver(self, state1):
		obx = state1["Sbar"] - (self.model_trainer.sim.beta * state1["Sbar"] * state1["Ibar"]/self.model_trainer.sim.N)
		w = 1/self.model_trainer.sim.N + self.theta[self.model_trainer.sim.nc:]
		m = gp.Model("bellman solver")
		x = m.addMVar(shape=self.model_trainer.sim.nc, vtype=GRB.INTEGER, name="x")

		m.setObjective(w @ x, GRB.MAXIMIZE)

		A = np.ones(self.model_trainer.sim.nc)

		# Build rhs vector
		rhs = np.array([self.model_trainer.NVac])

		# Add constraints
		m.addConstr(A @ x <= rhs, name="c")
		m.addConstr(0 <= x)
		m.addConstr(x <= obx)

		m.optimize()

		return x.X


class VFA_linear_policy(VFA_linear_trainer):
	def __init__(self, model, theta=None, N=10, M=10):
		self.model = model
		self.gg = 1.0
		if theta is None:
			self.theta = np.random.rand(2 * model.sim.nc)
		else:
			self.theta = theta
		self.VFA_trainer = VFA_linear_trainer(model_trainer=model.__copy__(), N=N, M=10)
		VFA_trainer.train()

	def update(self, model_new):
		self.model = model_new

	def decision(self,t):
		xstar = self.bellman_solver(self.model.state)
		return xstar

	def Vbar(self, theta, state):
		V = np.dot(theta[:self.model.sim.nc], state["Ibar"]) + np.dot(theta[self.model.sim.nc:],
			                                                                 state["Sbar"])
		return V

	def bellman_solver(self, state1):
		obx = state1["Sbar"] - (self.model.sim.beta * state1["Sbar"] * state1["Ibar"]/self.model.sim.N)
		w = 1  + self.theta[self.model.sim.nc:]
		m = gp.Model("bellman solver")
		x = m.addMVar(shape=self.model.sim.nc, vtype=GRB.INTEGER, name="x")

		m.setObjective(w @ x, GRB.MAXIMIZE)

		A = np.ones(self.model.sim.nc)

		# Build rhs vector
		rhs = np.array([self.model.NVac])

		# Add constraints
		m.addConstr(A @ x <= rhs, name="c")
		m.addConstr(0 <= x)
		m.addConstr(x <= obx)
		m.optimize()
		return x.X



