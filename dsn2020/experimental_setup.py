import os, sys, time, glob, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lstm_sequence_nonpadded import needlePassing
from lstm_experimentalsetup import interpretModel
from datetime import datetime
"""

Add noise for regularization
"""
class experimentalSetup:
	def __init__(self, path, task ="Suturing", run=0):
		self.data_path = path
		self.files_path = os.path.join(self.data_path, "Experimental_setup")
		self.task = task#"Needle_Passing"
		print ("Current task {}".format(self.task))
		self.to_scale = 1
		now = datetime.now()
		month_ = datetime.now().month
		day_ = now.day
		hour_ = now.hour
		minute_ = now.minute
		self.currentTimestamp = "{}{}{}{}".format(month_, day_, hour_, minute_)
		if run==0:
			"""
			Running this as the main script
			"""
			print ("Running this as the main script")
			self.to_scale = 0
			self.prepareSetup()
			self.iterateSetup()

	def loadOffsets(self):
		"""
		Loads kinematics related offsets
		"""
		kinOffset = dict()
		kinSpan = dict()
		kinOffset['cartesian'] = 0
		kinOffset['rotation'] = 3
		kinOffset['linearVelocity'] = 12
		kinOffset['angularVelocity'] = 15
		kinOffset['grasperAngle'] = 18
		kinSpan['cartesian'] = 3
		kinSpan['rotation'] = 9
		kinSpan['linearVelocity'] = 3
		kinSpan['angularVelocity'] = 3
		kinSpan['grasperAngle'] = 1
		return kinOffset, kinSpan

	def selectTrial(self, mode=0):
		"""
		This function provides the mode of trial: OneTrialOut, SuperTrialOut or UserOut
		"""

		if mode == 0:
			return "OneTrialOut"
		elif mode == 1:
			return "SuperTrialOut"
		return "UserOut"

	def prepareSetup(self, setup="Balanced", super_task="GestureClassification", trial=0):
		"""
		This function will set the path variables, starting from the base path which contains the script, to the experimentalSetup
		to the task (suturing/knottyng/needlepassing) to the  task setup (GestureClassification)
		"""

		self.task_path = os.path.join(self.files_path, self.task)
		self.task_setup = os.path.join(self.task_path, setup)
		self.super_tasksetup = os.path.join(self.task_setup, super_task)

	def iterateSetup(self, setup="Balanced", super_task="GestureClassification", trial=0):
		"""
		This function will get the trial type of the experiments by calling the selectTrial function
		and then provide the experiment file which requires another layer of globbing
		"""

		self.loadDemonstrations()

		for mode in range(2,3):
			self.iter_labels = {}
			trial = self.selectTrial(mode)
			print ("current mode {}".format(trial))
			trial_path = os.path.join(self.super_tasksetup, trial)
			glob_path = os.path.join(trial_path, "*")
			#print ("glob path {}".format(glob_path))
			super_outlength = 0
			for name in sorted(glob.glob(glob_path)):
				#print (name.split("/")[-1])
				self.readIteration(name, trial, setup, feature_selection = "specific")



	def readIteration(self, itr_path, trial, setup, train=1, feature_selection=None):
		"""
		This function will get the iteration folder and read the train and
		test files along with the gesture for each experiment

		"""
		user_out = itr_path.split("/")[-1]
		itr_path = os.path.join(itr_path,"*")
		itr_num = 0
		model_class = None
		lstm_classifier = None
		data = []
		for name in sorted(glob.glob(itr_path)):
			#print ("will extract sequence for itr_path ", name)
			data.append(self.readtrainTest(name))
			itr_num += 1
		data = np.asarray(data)
		if feature_selection == "specific":
			data, kinvars = self.getSpecificFeatures(data)
		else:
			kinvars = "All"

		#data = self.getSpecificFeatures(data)
		print ("data shape {}".format(data.shape))
		if train == 1:
			model_class = interpretModel(self.data_path, self.task)
			model_class, lstm_classifier = self.trainModel(data, itr_num, model_class, lstm_classifier, trial, user_out, setup, kinvars)
		return data, kinvars

	def getSpecificFeatures(self, data):
		"""
		Gets only certain features from the data vector, say only cartesian, linear velocity, etc.
		"""
		kinOffset, kinSpan = self.loadOffsets()
		kin_var1 = 'cartesian'; kin_var2 = 'rotation'; kin_var3 = 'grasperAngle'#feature_shape = 14
		kin_vars = "{}_{}_{}".format(kin_var1, kin_var2, kin_var3)
		feature_shape = 2*(kinSpan[kin_var1] + kinSpan[kin_var2] + kinSpan[kin_var3])#14
		temp_offset = int(feature_shape/2)
		for i in range(data.shape[0]):
			x_train, y_train, x_test, y_test = data[i]
			offset = int(x_train.shape[-1]/2)

			temp_xtrain = np.zeros((x_train.shape[0], x_train.shape[1], feature_shape))
			temp_xtest = np.zeros((x_test.shape[0], x_test.shape[1], feature_shape))
			print ("x train shape {}".format(temp_xtrain.shape))
			temp_offset = int(temp_xtrain.shape[-1]/2)
			cartesian_offset = 0
			#linearVelocity_offset = kinSpan['cartesian'] + cartesian_offset
			angularVelocity_offset = kinSpan[kin_var1] + cartesian_offset
			#grasperAngle_offset = kinSpan['linearVelocity'] + linearVelocity_offset
			grasperAngle_offset = kinSpan[kin_var2] + angularVelocity_offset


			temp_xtrain[:,:,cartesian_offset:cartesian_offset+kinSpan[kin_var1]] = x_train[:,:,kinOffset[kin_var1]:kinOffset[kin_var1]+kinSpan[kin_var1]]
			temp_xtrain[:,:,temp_offset+cartesian_offset+kinOffset[kin_var1]:(temp_offset+cartesian_offset+kinSpan[kin_var1])] = x_train[:,:,offset+kinOffset[kin_var1]:(offset+kinOffset[kin_var1]+kinSpan[kin_var1])]

			temp_xtrain[:,:,angularVelocity_offset:angularVelocity_offset+kinSpan[kin_var2]] = x_train[:,:,kinOffset[kin_var2]:kinOffset[kin_var2]+kinSpan[kin_var2]]
			temp_xtrain[:,:,(temp_offset+angularVelocity_offset):(temp_offset+angularVelocity_offset+kinSpan[kin_var2])] = x_train[:,:,(offset+kinOffset[kin_var2]):(offset+kinOffset[kin_var2]+kinSpan[kin_var2])]

			temp_xtrain[:,:,grasperAngle_offset:grasperAngle_offset+kinSpan[kin_var3]] = x_train[:,:,kinOffset[kin_var3]:kinOffset[kin_var3]+kinSpan[kin_var3]]
			temp_xtrain[:,:,(temp_offset+grasperAngle_offset):(temp_offset+grasperAngle_offset+kinSpan[kin_var3])] = x_train[:,:,offset+kinOffset[kin_var3]:(offset+kinOffset[kin_var3]+kinSpan[kin_var3])]

			temp_xtest[:,:,cartesian_offset:(cartesian_offset+kinSpan[kin_var1])] = x_test[:,:,kinOffset[kin_var1]:kinOffset[kin_var1]+kinSpan[kin_var1]]
			temp_xtest[:,:,(temp_offset+cartesian_offset):(temp_offset+cartesian_offset+kinSpan[kin_var1])] = x_test[:,:,offset+kinOffset[kin_var1]:(offset+kinOffset[kin_var1]+kinSpan[kin_var1])]

			temp_xtest[:,:,angularVelocity_offset:(angularVelocity_offset+kinSpan[kin_var2])] = x_test[:,:,kinOffset[kin_var2]:kinOffset[kin_var2]+kinSpan[kin_var2]]
			temp_xtest[:,:,(temp_offset+angularVelocity_offset):(temp_offset+angularVelocity_offset+kinSpan[kin_var2])] = x_test[:,:,(offset+kinOffset[kin_var2]):(offset+kinOffset[kin_var2]+kinSpan[kin_var2])]

			temp_xtest[:,:,grasperAngle_offset:(grasperAngle_offset+kinSpan[kin_var3])] = x_test[:,:,kinOffset[kin_var3]:kinOffset[kin_var3]+kinSpan[kin_var3]]
			temp_xtest[:,:,(temp_offset+grasperAngle_offset):(temp_offset+grasperAngle_offset+kinSpan[kin_var3])] = x_test[:,:,offset+kinOffset[kin_var3]:(offset+kinOffset[kin_var3]+kinSpan[kin_var3])]

			#print ("x train {} x test {}".format(x_train.shape, x_test.shape))
			if x_train.shape[-1]>38:
			    temp_xtrain = np.concatenate((temp_xtrain, x_train[:,:,38:]), axis=2)
			    temp_xtest = np.concatenate((temp_xtest, x_test[:,:,38:]), axis=2)
			data[i] = [temp_xtrain, y_train, temp_xtest, y_test]

		return data, kin_vars

	def readtrainTest(self, dir_name):
		"""
		This function will get the iteration subfolders and read the train and
		test files along with the gesture for each experiment
		"""
		#print ("dir_name {}".format(dir_name))
		train_path = os.path.join(dir_name, "Train.txt")
		train_sequences = self.readTranscripts(train_path)
		x_train, y_train = self.extractSequences(train_sequences)
		test_path = os.path.join(dir_name, "Test.txt")
		test_sequences = self.readTranscripts(test_path)
		x_test, y_test = self.extractSequences(test_sequences)
		X_train, y_train, X_test, y_test = self.prepareData(x_train, y_train, x_test, y_test)

		data = [X_train, y_train, X_test, y_test]

		return data


	def readTranscripts(self, itr_path):
		"""
		This function reads the train and test file for each iteration
		"""

		try:

			df = np.array(pd.read_csv(itr_path, delimiter='           ', header=None, engine="python"))
			return df
		except:
			print ("File not found for directory ",itr_path)
			pass
		return []


	def extractSequences(self, sequences):
		"""
		Input: df contraining the sequences to extract
		Decription: This function gets to know the sequences for train and test that needs extracting and calls the dictionary to extract it
		Output: train/test sequence along with labels
		"""

		x_temp = []
		y_temp = []
		sequence_length = 0 #checking for sanity, needs removing
		for sequence in (sequences):
			tokenized_sequence = sequence[0].split("_")
			name = "{}_{}".format(self.task, tokenized_sequence[-3])
			start = int(tokenized_sequence[-2])
			end = int(tokenized_sequence[-1].replace(".txt",""))

			x_temp.extend(self.all_demonstrationdict[name][start:end])

			y_temp.extend(self.encodeLabel(start, end, sequence[1]))

			sequence_length += end-start
			#print ("subject name {} start {} end {}".format(name, start, end))
		x_temp = np.asarray(x_temp)
		#print ("sequence length {} xtrain shape {}".format(sequence_length, x_train.shape[0]))
		assert sequence_length == x_temp.shape[0]

		return x_temp, y_temp


	def encodeLabel(self, start, end, label):
		one_hotlabel = np.zeros((end-start,15))
		one_hotlabel[:,int(label.replace("G",""))-1] = 1
		#print ("label {} onehot {}".format(label, one_hotlabel))
		return one_hotlabel

	def prepareData(self, x_train, y_train, x_test, y_test):
		"""
		Thus function calls the scale function to standarize the data and then temporalize it using a certain window size
		"""
		x_trainscaled= x_train
		x_testscaled =  x_test
		if self.to_scale == 0:

			print ("scaling data, which means running this as the main script")

			x_trainscaled, x_testscaled = self.scaleData(x_train, x_test)

		x_train_temporalized, y_train_temporalized, x_test_temporalized, y_test_temporalized = self.temporalizeData(x_trainscaled, y_train, x_testscaled, y_test, 1)
		#print ("shapes {} {} {} {}".format(np.asarray(x_train_temporalized).shape, np.asarray(y_train_temporalized).shape,
		#np.asarray(x_test_temporalized).shape, np.asarray(y_test_temporalized).shape))

		return np.asarray(x_train_temporalized), np.asarray(y_train_temporalized), np.asarray(x_test_temporalized), np.asarray(y_test_temporalized)


	def scaleData(self, x_train, x_test):
		"""
		This function scales the training and testing data
		"""

		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)
		return x_train, x_test

	def temporalizeData(self, x_train, y_train, x_test, y_test, timesteps=1):
		"""
		This function temporalizes data
		"""

		x_train_temporalize = []
		y_train_temporalize = []
		x_test_temporalize = []
		y_test_temporalize = []

		for i in range(x_train.shape[0]-timesteps):
			x_train_temporalize.append(x_train[i:i+timesteps])
			y_train_temporalize.append(y_train[i+timesteps-1])

		for i in range(x_test.shape[0]-timesteps):
			x_test_temporalize.append(x_test[i:i+timesteps])
			y_test_temporalize.append(y_test[i+timesteps-1])

		return np.asarray(x_train_temporalize), np.asarray(y_train_temporalize), np.asarray(x_test_temporalize), np.asarray(y_test_temporalize)

	def loadDemonstrations(self):
		"""
		This function calls another class to load the demonstrations into a dict
		"""
		print ("entered load demonstrations.loadDemonstrations()")
		seq = needlePassing(self.data_path, "/%s/"%(self.task))
		seq.loadDemonstrations("/kinematics")

		self.all_demonstrationdict = seq.all_demonstrationdict


	def trainModel(self, data, itr, model_class, lstm_classifier, trial, user_out, setup, kinvars):
		"""
		This function calls the trainModel function in another class and gets accuracies
		"""
		_, timesteps, n_features = data[0][0].shape
		lr = 0.001
		print ("timesteps {} n_features {}".format(timesteps, n_features))
		lstm_classifier  = model_class.buildModelv5(timesteps, n_features) #switching to GRU model
		model_class.setData(data)
		model_class.trainModel(lstm_classifier, trial, user_out, setup, lr, "convlstm_ae", self.currentTimestamp, kinvars)
		return model_class, lstm_classifier

	def setVariables(self, demonstrations):
		self.all_demonstrationdict = demonstrations

path = os.path.abspath(os.path.dirname(sys.argv[0]))
try:
    script,mode = sys.argv
except:
    print ("Error: missing parameters")
    print (usage)
    sys.exit(0)
if mode == "0":
	print ("Gesture classification")
	exp = experimentalSetup(path, run=int(mode))

else:
    print ("Running anomaly detection")
    exp = experimentalSetup(path, run=mode)
