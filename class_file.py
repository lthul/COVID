import numpy as np
import pandas as pd
import data_loader as dl
import matplotlib.pyplot as plt

def DATA_LOAD(date="2020-06-19"):
	fips_pop = dl.load_data("fips_populations.obj")
	fips_county = dl.load_data("fips_county.obj")
	df = pd.read_csv("us-counties.csv")
	datebools = df.date == date
	FIPS = np.int32(df.fips[datebools])
	CASES = df.cases[datebools]
	FIPS_LIST = list(np.unique(np.array(df.fips)[~np.isnan(df.fips)]))
	counties = [County(df, fips_pop, fips_county, fips) for fips in FIPS_LIST]
	return counties

def getFIPSList(counties, key='pop'):
	if key == 'pop':
		return {counties[i].fips: counties[i].population for i in range(len(counties))}
	elif key == 'tot_cases':
		return {counties[i].fips: counties[i].cdata.cases.tolist()[len(counties[i].cdata.cases.tolist())-1] for i in range(len(counties))}

def convFIPS2Name(fips):
	fips_county = dl.load_data("fips_county.obj")
	return [fips_county[fips[i]] for i in range(len(fips))]


class County:
	def __init__(self, df, pop, county, fips):
		self.cdata = df[df.fips == fips]
		self.fips = fips
		self.county = county[fips]
		self.population = pop[fips]

	def getDate(self, day, month, year="20"):
		# day = "XX" XX is the day of the month
		# month = "YY" where YY is the number of the month
		# year = "ZZ" where 20ZZ would represent the year (default 20)

		date = year + "-" + month + "-" + day



	def plot_covid(self):
		dates = list(self.cdata.date.tolist())
		cases = list(self.cdata.cases.tolist())
		plt.plot(dates, cases, 'r*')
		plt.title(self.county+", "+self.cdata.state.tolist()[0])
		return dict(zip(dates, cases))








