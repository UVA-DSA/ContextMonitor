import os, sys, time , glob
from sys import argv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lstm_sequence_nonpadded import needlePassing
from experimental_setup import experimentalSetup
from lstm_vaesuturing import lstmVAE
from vae_keras import VAE
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
class vae_setup:
    def __init__(self, path, detection):
        self.data_path = path
        self.files_path = os.path.join(self.data_path, "Experimental_setup")
        self.task = "Suturing"#"Needle_Passing"
        self.detection = detection
        self.exp = experimentalSetup(path, task=self.task, run=1)
        self.lstmae = lstmVAE(path, self.task)
        self.vae = VAE(path, self.task)
        print ("testing complete")
        self.prepareSetup()
        #self.iterateSetup()

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
        This function will load the errorcsvs and map them to the demonstrations
        """
        categorical = 1 #flag for doing segment wise or non-segmentwise
        if self.detection =="0" or self.detection=="1":
            timesteps = 1
        else:
            timesteps = 1#5

        self.task_path = os.path.join(self.files_path, self.task)
        self.task_setup = os.path.join(self.task_path, setup)
        self.super_tasksetup = os.path.join(self.task_setup, super_task)
        merged_dict = self.iterateSetup()
        self.exp.setVariables(merged_dict)
        for mode in range(0,1):
            self.iter_labels = {}
            trial = self.selectTrial(mode)
            print ("current mode {}".format(trial))
            trial_path = os.path.join(self.super_tasksetup, trial)
            glob_path = os.path.join(trial_path, "*")

            super_outlength = 0
            for name in sorted(glob.glob(glob_path)):
                data = self.exp.readIteration(name, trial, setup, train=0)
                print ("trial name {} data shape {}".format(name.split("/")[-1], data.shape))
                if trial == "OneTrialOut":
                    data = data[0].reshape(1,-1)
                    print ("reducing iterations for 50 to 1 for One Trial Out {}".format(data.shape))

                data = self.sortData(data)
                for i in range(len(data)):
                    x_train, y_train, x_test, y_test = data[i]
                    data[i] = self.exp.temporalizeData(x_train, y_train, x_test, y_test, timesteps=timesteps) #timestep 0  for only vae

                if self.detection == "0":
                    print ("Running anomaly detection using VAE on entire trajectory ")
                    self.vae.getInput(data)

                if self.detection == "1":
                    print ("Running anomaly detection using VAE and based on gestures")
                    self.vae.getCategorizedData(self.exp.currentTimestamp, data, name.split("/")[-1]) #gsture specific

                elif self.detection =="2":
                    print ("Running anomaly detection using LSTMAE on entire trajectory")
                    self.lstmae.getInput(self.exp.currentTimestamp, data, name.split("/")[-1]) #all gestures

                elif self.detection == "3":
                    print ("Running anomaly detection using LSTMAE and based on gestures")
                    self.lstmae.getCategorizedData(self.exp.currentTimestamp, data, name.split("/")[-1]) #gsture specific

                elif self.detection == "4":
                    print ("Running anomaly detection using LSTMAE concatenating feature vectors with gestures ")
                    self.lstmae.getInput(self.exp.currentTimestamp, data, name.split("/")[-1], "concatenate") #all gestures

                elif self.detection == "5":
                    print ("Running anomaly detection using LSTMVAE concatenating feature vectors with gestures ")
                    self.vae.getInput(data, "concatenate") #all gestures

    def readIteration(self, itr_path, trial, setup):
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
            print ("will extract sequence for itr_path {}".format(name))
            data.append(self.readtrainTest(name))
            itr_num += 1
        data = np.asarray(data)

    def iterateSetup(self, setup="Balanced", super_task="GestureClassification", trial=0):
        """
        This function will firstly load demonstrations, iterate over the experimental setup,
        load the sequences and map them to optimal and suboptimal sequences
        """
        all_demonstrationdict = self.loadDemonstrations()

        error_dict = self.loadErrorAnnotations()
        merged_dict = self.mergeDicts(all_demonstrationdict, error_dict)

        return merged_dict

    def getAnnotations(self, name, subject, error_dict):
        """
        This function gets the annotations for sub-optimal/safety-critical/optimals, given the subject
        """
        error_index = -1
        if self.task == "Suturing":
            error_index = -2
        annotations = np.asarray(pd.read_csv(name, delimiter=",", engine="python"))[:,error_index]
        error_dict[subject] = annotations
        return error_dict


    def loadDemonstrations(self):
        """
        Calls the meedlepassing class to load demonstrations, given the task
        """
        seq = needlePassing(self.data_path, "/%s/"%(self.task))
        seq.loadDemonstrations("/kinematics")

        return seq.all_demonstrationdict

    def loadErrorAnnotations(self):
        """
        This function will load the csvs with the error annotations
        """
        error_path = os.path.join(self.data_path, self.task, "suboptimals/*")
        error_dict = dict()
        for name in sorted(glob.glob(error_path)):
            subject = name.split("/")[-1].replace(".csv","").replace("error","")
            error_dict = self.getAnnotations(name, subject, error_dict)
        return error_dict

    def mergeDicts(self, all_demonstrationdict, error_dict):
        """
        This function will merge the demonstration dict to the error dict
        """
        merged_dict = dict()
        for key  in error_dict.keys():
            error_annotations = error_dict[key]
            all_demonstrations = all_demonstrationdict[key]
            merged_dict[key] = np.concatenate((all_demonstrations, error_annotations.reshape(-1,1)), axis=1)
        return merged_dict

    def sortData(self, data):
        """
        data received from exp class has y for predicting label, but this is a error classification problem
        """
        for i in range(len(data)):
            X_train, y_train, X_test, y_test = data[i]
            X_train = X_train.reshape(-1,X_train.shape[-1])
            X_test = X_test.reshape(-1,X_test.shape[-1])
            y_traintemp = X_train[:,-1].reshape(-1,1)
            y_testtemp = X_test[:,-1].reshape(-1,1)
            X_train = X_train[:,0:-1]
            X_test = X_test[:,0:-1]

            X_train, X_test = self.exp.scaleData(X_train, X_test)

            X_train = np.concatenate((X_train, y_train), axis=1)
            y_train = y_traintemp

            X_test = np.concatenate((X_test, y_test), axis=1)
            y_test = y_testtemp
            data[i] = [X_train, y_train, X_test, y_test]

        return data



path = os.path.abspath(os.path.dirname(sys.argv[0]))
usage = "0: "
try:
    script,mode = argv
except:
    print ("Error: missing parameters")
    print (usage)
    sys.exit(0)

v_setup = vae_setup(path, mode)
