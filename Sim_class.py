from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]

import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import *
import chart_studio.plotly as py
import plotly.offline as offline
import data_loader as dl
import numpy as np
import matplotlib.pyplot as plt

class Simulator:
	def __init__(self, T=25, beta = 0.9, gamma = 0.3, alpha = 0.8, n_seeds = 100, p_inf = 0.01, p_rec = 0.001):
		# T = time horizon
		# beta = the transmission rate of the virus (expected number of interactions between infected-susc persons)
		# gamma = the recovery rate of the virus (1 / time of recovery)
		# alpha = the mobility rate of people traveling
		# n_seeds is the number of infected counties to start with
		# p_inf = percent of infected people to initialize per county
		# p_rec = percent of recovered people to initialize per county

		# load data
		POP = dl.load_data('fips_populations.obj')
		FLOW = dl.load_data('FLOW2.obj')
		self.countyf = dl.load_data('fips_county.obj')
		idx_dict = dl.load_data('idx_dict2.obj')

		# county populations
		self.N = np.array(list(POP.values()))
		self.nc = len(self.N)
		self.T = T
		# initialize SIR population to total populations for S and 0 IR
		self.S = np.zeros([self.N.shape[0], T])
		self.S[:, 0] = self.N.__copy__()
		self.I = np.zeros(self.S.shape)
		self.R = np.zeros(self.S.shape)
		self.V = np.zeros(self.S.shape)

		# initialize model parameters

		# all beta and gamma values are constant throughout time and space
		self.beta_ = beta * np.ones(self.S.shape)
		self.gamma_ = gamma * np.ones(self.S.shape)
		self.alpha_ = alpha

		# set FLOW probability matrices
		FLOW += 1
		np.fill_diagonal(FLOW, 0)
		self.Nvals = self.N.repeat(self.nc).reshape(self.nc, self.nc)
		FLOW_mat = np.divide(FLOW, self.Nvals)
		self.FLOW_mat = np.min([FLOW_mat, 0.01 * np.ones(FLOW_mat.shape)], axis=0)

		# initialize the epidemic to have starter infections
		ii = np.arange(0, len(self.S))
		np.random.shuffle(ii)

		tot = np.sum(self.N)
		cdf = np.cumsum(self.N / tot)
		U = np.random.rand(n_seeds)
		ii = [np.argmin(np.abs(U[i] - cdf)) for i in range(len(U))]
		pct = p_inf * np.random.rand(n_seeds)
		infected = np.int32(self.N[ii] * pct)
		self.I[ii, 0] = infected
		self.R[:, 0] = np.int32(p_rec * self.N)
		self.S[:, 0] = self.N - self.I[:, 0] - self.R[:, 0]

	def forward_one_step(self, x, t, FLOWn=None):
		# x = N length vector of [x_vac]
		if FLOWn is None:
			FLOWn = np.random.binomial(np.int32(0.2 * self.Nvals.T), self.FLOW_mat)

		self.S[:, t + 1] = np.max([np.zeros(self.nc), self.S[:, t] - (self.beta_[:,t] * self.S[:, t] * self.I[:, t]) / self.N - (
					self.alpha_ * self.S[:, t] * np.dot(FLOWn, self.beta_[:,t] * self.I[:, t] / self.N)) / (self.N + np.sum(FLOWn, axis=1)) -x], axis=0)
		self.S[:, t + 1] = np.min([self.S[:, t + 1], self.N], axis=0)
		self.I[:, t + 1] = np.max([np.zeros(self.nc), (1 - self.gamma_[:,t]) * self.I[:, t] + (self.beta_[:,t] * self.S[:, t] * self.I[:, t]) / self.N + (
					self.alpha_ * self.S[:, t] * np.dot(FLOWn, self.beta_[:,t] * self.I[:, t] / self.N)) / (self.N + np.sum(FLOWn, axis=1))], axis=0)
		self.I[:, t + 1] = np.min([self.I[:, t + 1], self.N], axis=0)
		self.R[:, t + 1] = np.max([np.zeros(self.nc), self.R[:, t] + self.gamma_[:,t] * self.I[:, t]], axis=0)
		self.R[:, t + 1] = np.min([self.R[:, t + 1], self.N], axis=0)
		self.V[:, t + 1] = np.min([self.V[:,t] + x, self.N], axis=0)
		# Np = np.sum(self.S[:, t + 1]) + np.sum(self.I[:, t + 1]) + np.sum(self.R[:, t + 1])
		return self.S[:,t+1], self.I[:,t+1], self.R[:,t+1], self.V[:,t+1]



	def plot_USA_sim(self, code = "I"):
		# codes
		# "I" = infected, "S" = susceptible, "R" = recovered, "V" = vaccinated
		if code == "I":
			data = self.I
		elif code == "S":
			data = self.S
		elif code == "R":
			data = self.R
		elif code == "V":
			data = self.V
		data_slider = []

		colorbar = dict(tickvals=[-1, 0, 1, 2, 3, 4, 5],
		                ticktext=['0', '1', '10', '100', '1000', '10k', '100k'])
		for t in range(self.T):
			data_each_yr = dict(
				type='choropleth',
				geojson=counties,
				locations=list(self.countyf.keys()),
				z=np.max([np.log10(data[:, t] + 1e-1), -1 * np.ones(data[:, t].shape)], axis=0),
				colorbar=colorbar,
				colorscale='Jet',
				locationmode='geojson-id'
			)

			data_slider.append(data_each_yr)

		steps = []
		for i in range(len(data_slider)):
			step = dict(method='restyle',
			            args=['visible', [False] * len(data_slider)],
			            label='Day {}'.format(i))
			step['args'][1][i] = True
			steps.append(step)

		sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

		layout = dict(title='Virus cases', geo=dict(scope='usa',
		                                            projection={'type': 'albers usa'}),
		              sliders=sliders)

		fig = dict(data=data_slider, layout=layout)
		# fig = dict(data=data_slider)
		offline.plot(fig)

	def plot_totals(self):
		plt.plot(np.sum(self.S, axis=0), 'b')
		plt.plot(np.sum(self.I, axis=0), 'r')
		plt.plot(np.sum(self.R, axis=0), 'g')
		plt.plot(np.sum(self.V, axis=0), 'k')
		plt.legend(["Susc", "Inf", "Rec", "Vac"])


	def plot_county(self, idx):
		plt.plot(np.sum(self.S[idx, :], axis=0), 'b')
		plt.plot(np.sum(self.I[idx, :], axis=0), 'r')
		plt.plot(np.sum(self.R[idx, :], axis=0), 'g')
		plt.plot(np.sum(self.V[idx, :], axis=0), 'k')

	def reset_counts(self):
		self.S[:, 1:] = 0
		self.I[:, 1:] = 0
		self.R[:, 1:] = 0
		self.V[:, 1:] = 0

	def reset_t(self, t):
		self.S[:, t+1] = 0
		self.I[:, t+1] = 0
		self.R[:, t+1] = 0
		self.V[:, t+1] = 0