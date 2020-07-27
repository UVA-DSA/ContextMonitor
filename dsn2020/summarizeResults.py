import os, sys, glob
import numpy as np
import pandas as pd

class summarize:
	def __init__(self, path):
		"""
		Just takes the path
		"""
		self.data_path = path

	def getResults(self):
		"""
		gets the result path
		"""
#/media/yq/0d57f6b9-4507-4edb-9800-537ac5258c2b/home/hdd_files/samin/dsn2019/prelimstudy/lstm_models/needle_passinglogs/Balanced/OneTrialOut/BalancedNeedle_PassingOneTrialOut1_Out.csv
		result_path = os.path.join(self.data_path, "lstm_models/needle_passinglogs/Balanced/OneTrialOut/*.csv")
		self.iterateFiles(result_path)

	def iterateFiles(self, result_path):
		"""
		reads the result file
		"""
		val_mean = 0
		count = 0
		for name in sorted(glob.glob(result_path)):
			print (name)
			csv_df = pd.read_csv(name, delimiter=";")
			val_mean += csv_df[["val_acc"]].iloc[[-1]].mean()
			#print ("subject name {} mean {}".format(name.split("/")[-1], csv_df[["val_acc"]].mean()))
			print ("subject name {} mean {}".format(name.split("/")[-1], csv_df[["val_acc"]].iloc[[-1]]))
			count +=1
		print (val_mean)
		val_mean = val_mean/count
		print ("file mean {}".format(val_mean))


path_ = os.path.abspath(os.path.dirname(sys.argv[0]))
smr = summarize(path_)
smr.getResults()
