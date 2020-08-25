import numpy as np
import matplotlib.pyplot as plt

class model:
	def __init__(self, sim, NVac = 50):
		# sim = the simulator object
		self.sim = sim
		self.NVac = NVac
		self.state = {}
		self.state["Ibar"] = self.sim.getI(0)
		self.state["Rbar"] = np.zeros(self.state["Ibar"].shape)
		self.state["Sbar"] = self.sim.N - self.state["Ibar"] - self.state["Rbar"]
		# self.decision_space = self.get_feasibleDspace()


	def __copy__(self):
		return model(self.sim)

	def exog(self):
		return self.NVac

	def transition(self,x_t,t):
		Rbar1 = np.int32(self.state["Rbar"] + self.sim.gamma_[0,0] * self.state["Ibar"] + x_t)
		Rbar1 = np.max([np.zeros(self.sim.nc), np.min([Rbar1, self.sim.N], axis=0)], axis=0)
		Ibar1 = np.int32(self.sim.getI(t+1))
		Ibar1 = np.max([np.zeros(self.sim.nc), np.min([Ibar1, self.sim.N], axis=0)], axis=0)

		Sbar1 = np.int32(self.sim.N - Rbar1 - Ibar1)
		Sbar1 = np.max([np.zeros(self.sim.nc), np.min([Sbar1, self.sim.N], axis=0)], axis=0)

		self.state["Ibar"] = Ibar1
		self.state["Rbar"] = Rbar1
		self.state["Sbar"] = Sbar1

	def objective(self):
		return (1/self.sim.nc) * np.sum(self.state["Ibar"] / self.sim.N)

	def forward_one_step(self, x_t, sim, t):
		Nvac = self.exog()
		self.sim = sim
		self.transition(x_t, t)
		return self.objective()

	def get_feasibleDspace(self,disc=10):
		points = [np.arange(0, self.NVac, self.NVac / disc)] * self.sim.nc
		Dspace = np.vstack(np.meshgrid(*points))
		Dspace = Dspace.reshape(self.sim.nc,np.int32(np.product(Dspace.shape)/self.sim.nc))
		bools = np.sum(Dspace,axis=0) <= self.NVac
		fDspace = Dspace[:,bools]
		fspace = [fDspace[:, i] for i in range(fDspace.shape[1])]
		return fspace

	def sim_trans(self, x_t, state1):
		state = state1.copy()
		Sbar1 = state["Sbar"] - (self.sim.beta * state["Sbar"] * state["Ibar"]) / self.sim.N - x_t
		Rbar1 = state["Rbar"] + self.sim.gamma * state["Ibar"] + x_t
		Ibar1 = (1 - self.sim.gamma) * state["Ibar"] + (self.sim.beta * state["Sbar"] * state["Ibar"]) / self.sim.N

		# Sbar1 = self.sim.N - Rbar1 - Ibar1

		Rbar1 = np.max([np.zeros(self.sim.nc), np.min([Rbar1, self.sim.N], axis=0)], axis=0)
		Ibar1 = np.max([np.zeros(self.sim.nc), np.min([Ibar1, self.sim.N], axis=0)], axis=0)
		Sbar1 = np.max([np.zeros(self.sim.nc), np.min([Sbar1, self.sim.N], axis=0)], axis=0)

		state["Ibar"] = Ibar1
		state["Sbar"] = Sbar1
		state["Rbar"] = Rbar1
		return state

	def sim_objective(self, state,x=None):
		if x is not None:
			s1 = self.sim_trans(x,state)
		else:
			s1 = state
		return (1/self.sim.nc) * np.sum(s1["Sbar"] / self.sim.N)

	def plotcounties(self,mS,mI,mR,mV,I,J):
		fig, ax = plt.subplots(I, J)
		if I > 1:
			for i in np.arange(0, I):
				for j in np.arange(0, J):
					if ((J * i + j) < mS.shape[0]):
						ax[i][j].plot(mS[J * i + j, :], 'b')
						ax[i][j].plot(mI[J * i + j, :], 'r')
						ax[i][j].plot(mR[J * i + j, :], 'g')
						ax[i][j].plot(mV[J * i + j, :], 'k')


class model:
	def __init__(self, sim, NVac):
		# sim = the simulator object
		self.sim = sim
		self.NVac = NVac
		self.state = {}
		self.state["Ibar"] = self.sim.getI(0)
		self.state["Rbar"] = np.zeros(self.state["Ibar"].shape)
		self.state["Sbar"] = self.sim.N - self.state["Ibar"] - self.state["Rbar"]
		self.state["nvac"] = self.NVac
		# self.decision_space = self.get_feasibleDspace()


	def __copy__(self):
		return model(self.sim)

	def exog(self):
		return self.NVac

	def transition(self,x_t,t):
		Ibar1 = np.int32(self.sim.getI(t + 1))
		Ibar1 = np.max([np.zeros(self.sim.nc), np.min([Ibar1, self.sim.N], axis=0)], axis=0)

		Rbar1 = np.int32(self.state["Rbar"] + self.sim.gamma_[0,0] * Ibar1 + x_t)
		Rbar1 = np.max([np.zeros(self.sim.nc), np.min([Rbar1, self.sim.N], axis=0)], axis=0)

		Sbar1 = np.int32(self.sim.N - Rbar1 - Ibar1)
		Sbar1 = np.max([np.zeros(self.sim.nc), np.min([Sbar1, self.sim.N], axis=0)], axis=0)

		self.state["Ibar"] = Ibar1
		self.state["Rbar"] = Rbar1
		self.state["Sbar"] = Sbar1
		self.state["nvac"] = self.NVac

	def objective(self):
		return (1/self.sim.nc) * np.sum(self.state["Ibar"] / self.sim.N)

	def forward_one_step(self, x_t, sim, NVac, t):
		self.NVac = NVac
		self.sim = sim
		self.transition(x_t, t)
		return self.objective()

	def get_feasibleDspace(self,disc=10):
		points = [np.arange(0, self.NVac, self.NVac / disc)] * self.sim.nc
		Dspace = np.vstack(np.meshgrid(*points))
		Dspace = Dspace.reshape(self.sim.nc,np.int32(np.product(Dspace.shape)/self.sim.nc))
		bools = np.sum(Dspace,axis=0) <= self.NVac
		fDspace = Dspace[:,bools]
		fspace = [fDspace[:, i] for i in range(fDspace.shape[1])]
		return fspace

	def sim_trans(self, x_t, state1):
		state = state1.copy()
		Sbar1 = state["Sbar"] - (self.sim.beta * state["Sbar"] * state["Ibar"]) / self.sim.N - x_t
		Rbar1 = state["Rbar"] + self.sim.gamma * state["Ibar"] + x_t
		Ibar1 = (1 - self.sim.gamma) * state["Ibar"] + (self.sim.beta * state["Sbar"] * state["Ibar"]) / self.sim.N

		# Sbar1 = self.sim.N - Rbar1 - Ibar1

		Rbar1 = np.max([np.zeros(self.sim.nc), np.min([Rbar1, self.sim.N], axis=0)], axis=0)
		Ibar1 = np.max([np.zeros(self.sim.nc), np.min([Ibar1, self.sim.N], axis=0)], axis=0)
		Sbar1 = np.max([np.zeros(self.sim.nc), np.min([Sbar1, self.sim.N], axis=0)], axis=0)

		state["Ibar"] = Ibar1
		state["Sbar"] = Sbar1
		state["Rbar"] = Rbar1
		return state

	def sim_objective(self, state,x=None):
		if x is not None:
			s1 = self.sim_trans(x,state)
		else:
			s1 = state
		# return (1/self.sim.nc) * np.sum(s1["Sbar"] / self.sim.N)
		return (1 / self.sim.nc) * np.sum(s1["Sbar"])

	def plotcounties(self,mS,mI,mR,mV,I,J):
		fig, ax = plt.subplots(I, J)
		if I > 1:
			for i in np.arange(0, I):
				for j in np.arange(0, J):
					if ((J * i + j) < mS.shape[0]):
						ax[i][j].plot(mS[J * i + j, :], 'b')
						ax[i][j].plot(mI[J * i + j, :], 'r')
						ax[i][j].plot(mR[J * i + j, :], 'g')
						ax[i][j].plot(mV[J * i + j, :], 'k')



class det_model_trainer:
	def __init__(self, sim, NVac):
		# sim = the simulator object
		self.sim = sim
		self.NVac = NVac
		self.state = {}
		self.state["Ibar"] = self.sim.getI(0)
		self.state["Rbar"] = np.zeros(self.state["Ibar"].shape)
		self.state["Sbar"] = self.sim.N - self.state["Ibar"] - self.state["Rbar"]
		# self.decision_space = self.get_feasibleDspace()

	def __copy__(self):
		return det_model_trainer(self.sim, self.NVac)

	def get_feasibleDspace(self,disc=10):
		points = [np.arange(0, self.NVac, self.NVac / disc)] * self.sim.nc
		Dspace = np.vstack(np.meshgrid(*points))
		Dspace = Dspace.reshape(self.sim.nc,np.int32(np.product(Dspace.shape)/self.sim.nc))
		bools = np.sum(Dspace,axis=0) <= self.NVac
		fDspace = Dspace[:,bools]
		fspace = [fDspace[:, i] for i in range(fDspace.shape[1])]
		return fspace


	def exog(self):
		return self.NVac

	def sim_trans(self, x_t, state1):
		state = state1.copy()
		Rbar1 = state["Rbar"] + self.sim.gamma_[0, 0] * state["Ibar"] + x_t
		Ibar1 = (1 - self.sim.gamma_[0, 0]) * state["Ibar"] + (
					self.sim.beta_[0, 0] * state["Sbar"] * state["Ibar"]) / self.sim.N


		Sbar1 = self.sim.N - Rbar1 - Ibar1

		Rbar1 = np.max([np.zeros(self.sim.nc), np.min([Rbar1, self.sim.N], axis=0)], axis=0)
		Ibar1 = np.max([np.zeros(self.sim.nc), np.min([Ibar1, self.sim.N], axis=0)], axis=0)
		Sbar1 = np.max([np.zeros(self.sim.nc), np.min([Sbar1, self.sim.N], axis=0)], axis=0)

		state["Ibar"] = Ibar1
		state["Sbar"] = Sbar1
		state["Rbar"] = Rbar1
		return state



	def sim_objective(self, state,x=None):
		if x is not None:
			s1 = self.sim_trans(x,state)
		else:
			s1 = state
		# return (1/self.sim.nc) * np.sum(s1["Sbar"] / self.sim.N)
		return (1 / self.sim.nc) * np.sum(s1["Sbar"])

