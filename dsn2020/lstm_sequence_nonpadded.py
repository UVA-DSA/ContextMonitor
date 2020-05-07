import os, glob, sys, _pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class needlePassing:
    def __init__(self, dataPath, task):
        """
        Usage: if making change to the data, use the following sequence:
        loadDemonstrations -> dumpDemonstrations -> loadPickle -> temporalizeData

        """
        self.dataPath = dataPath
        self.max_gestures = 15
        self.task = task


    def priorSetup(self):
        self.task = "/Needle Passing/"
        self.loadDemonstrations("/kinematics")
        self.dumpDemonstrations("/pickles")
        self.loadPickle("/pickles")
        self.temporalizeData()

    def loadDemonstrations(self, key):
        """
        This function globs over all the demonstrations for the given key (kinematics, transcriptions, video) and calls the plotgraph function after completing globbing
        """
        print ("ented load demonstraitons")
        self.all_demonstrations = []
        self.all_labels = []
        self.all_demonstrationdict = {}
        #self.task = "/Suturing/"
        demonstrationsPath = self.dataPath + self.task + key
        scores = self.loadMetaFile()
        globPath = demonstrationsPath + "/**/*"
        color = ""
        for name in glob.glob(globPath):
            if key == "/kinematics":
                cartesians = self.readCartesians(name)
                transcriptFile = name.replace(key, "/transcriptions").replace("AllGestures", "")
                transcript = self.readTranscripts(transcriptFile)
                if len(transcript)>0:
                    scaled_kinematics, labels = self.makeLabels(transcript, cartesians)
                    self.all_demonstrations.append(scaled_kinematics)
                    self.all_labels.append(labels)
                    self.all_demonstrationdict[name.split("/")[-1].replace(".txt","")] = cartesians

    def getSequenceLengths(self, sequences, labels):
        flattened_sequence = list()
        flattened_labels = list()
        len_sequences = np.zeros(len(sequences))
        for i in range(len(sequences)):
            len_sequences[i] = sequences[i].shape[0]
            print (labels[i].shape)
            flattened_sequence.extend(sequences[i])
            flattened_labels.extend(labels[i])
        flattened_sequence = np.asarray(flattened_sequence)#.reshape(-1, sequences[0].shape[-1])
        flattened_labels = np.asarray(flattened_labels)#.reshape(-1, sequences[0].shape[-1])
        print ("flattened array length {} len_sequences {} len_labels {}".format(flattened_sequence.shape, sum(len_sequences), len(flattened_labels)))
        assert flattened_sequence.shape[0] == len(flattened_labels) == sum(len_sequences)

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.splitDataset(flattened_sequence, flattened_labels)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def splitDataset(self, flattened_sequence, flattened_labels):
        """
        this method is used for standarizing the whole dataset, regardless of the sequence
        """
        X_train, X_test, y_train, y_test = train_test_split(flattened_sequence, flattened_labels, test_size=0.2, random_state=42, shuffle=True) #remove shuffle=False to get previous accuracy
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42 , shuffle=True) #remove shuffle=False to get previous accuracy
        scaler = StandardScaler().fit(X_train)
        #scaler = MinMaxScaler(feature_range=(0,1)).fit(kinematics)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)


        print ("train {} valid {} test {}".format(X_train.shape, X_valid.shape, X_test.shape))
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def restorePadSequence(sequences, len_sequences):
        prev_index = 0
        restored_sequences = list()
        for i in range(len(len_sequences)):
            restored_sequences.append(sequences[prev_index:prev_index+int(len_sequences[i])])
            prev_index = int(len_sequences[i])
        padded_sequences = np.asarray(sequence.pad_sequences(restored_sequences, value=-1.0, dtype='float32'))
        print (padded_sequences.shape)
        padded_sequences = padded_sequences.reshape(-1, padded_sequences.shape[-2], padded_sequences.shape[-1])
        return padded_sequences


    def dumpDemonstrations(self, key):
        """
        input: path to dump
        This function dumps the pre-processed kinematics and labels to the specified pickle path
        """
        dump_path = self.dataPath + key
        kinematics_pickle = dump_path + "/kinematics_nonpaded{}.p".format(self.task.replace("/",""))
        labels_pickle = dump_path + "/labels_nonpaded{}.p".format(self.task.replace("/",""))
        if not  os.path.exists(dump_path):
            os.makedirs(dump_path)
        with open(kinematics_pickle, 'wb') as fp:
            _pickle.dump(self.all_demonstrations, fp)
        with open(labels_pickle, 'wb') as fp:
            _pickle.dump(self.all_labels, fp)

    def loadPickle(self, key):
        """
        input: path to dump
        This function dumps the pre-processed kinematics and labels to the specified pickle path
        """
        dump_path = self.dataPath + key
        kinematics_pickle = dump_path + "/kinematics_nonpaded{}.p".format(self.task.replace("/",""))
        labels_pickle = dump_path + "/labels_nonpaded{}.p".format(self.task.replace("/",""))
        with open(kinematics_pickle, 'rb') as fp:
            self.all_demonstrations = _pickle.load(fp)
        with open(labels_pickle, 'rb') as fp:
            self.all_labels = _pickle.load(fp)

    def dumpGenerators(self, key, generator, gen_type):
        """
        input: generator
        This function dumps the generator
        """
        dump_path = self.dataPath + key
        generator_pickle = dump_path + "/%s_generator%s.p"%(gen_type, (self.task.replace("/","")))
        if not  os.path.exists(dump_path):
            os.makedirs(dump_path)
        with open(generator_pickle, 'wb') as fp:
            _pickle.dump(generator, fp)

    def dumptrainData(self, key, data, label, gen_type):
        """
        input: data
        This function dumps the x_train:x_test
        """
        dump_path = self.dataPath + key
        x_pickle = dump_path + "/x_%s_%s.p"%(gen_type, (self.task.replace("/","")))
        y_pickle = dump_path + "/y_%s_%s.p"%(gen_type, (self.task.replace("/","")))
        print (x_pickle)
        if not  os.path.exists(dump_path):
            os.makedirs(dump_path)
        with open(x_pickle, 'wb') as fp:
            _pickle.dump(data, fp)

        with open(y_pickle, 'wb') as fp:
            _pickle.dump(label, fp)

    def readTranscripts(self, transcript):
    	"""
    	This function reads the transcript file for each demonstration
    	"""
    	try:
    		df = np.array(pd.read_csv(transcript, delimiter=' ', header=None))
    		return df
    	except IOError:
    		pass
    	return []

    def makeLabels(self, transcript, kinematics):
        """
        This function reads the transcripts and makes labels and resizes kinematics
        """
        labels = np.zeros((kinematics.shape[0], 1))
        for i in range(transcript.shape[0]):
            labels[transcript[i][0]:transcript[i][1]] = transcript[i][-2].replace("G", "")

        labels = self.resizeKinematics(labels, transcript)
        kinematics = self.resizeKinematics(kinematics, transcript)
        scaled_kinematics = self.scaleKinematics(kinematics)
        encoded_labels = self.oneHotLabel(labels)

        return scaled_kinematics, encoded_labels

    def oneHotLabel(self, labels):
        max_labels = self.max_gestures
        encoded_labels = np.zeros((len(labels), max_labels))
        for i in range(labels.shape[0]):
          #print (labels[i])
          encoded_labels[i][int(labels[i][0])-1] = 1

        return encoded_labels

    def resizeKinematics(self, kinematics, transcript):
        """
        Uses the beginning and end of the label array to resize kinematics
        """
        return kinematics[transcript[0][0]:transcript[-1][1]]

    def scaleandreturn(self, kinematics, scaler):
        """
        Scale the kinematics and return the scalar
        """



    def scaleKinematics(self, kinematics, scaler=None):
        """
        Uses standard scaler function of sklearn to scale the kinematics
        """
        scaler = StandardScaler().fit(kinematics)
        #scaler = MinMaxScaler(feature_range=(0,1)).fit(kinematics)
        scaled_kinematics = scaler.transform(kinematics)
        return kinematics

    def readCartesians(self, demonstration):
        """
        This function reads the cartesian values from the kinematics file for each demonstration
        """
        #print (demonstration)
        df = np.array(pd.read_csv(demonstration, delimiter = '    ', header = None, engine='python'))
        psm_offset = df.shape[1]//2

        cartesians = df[:,psm_offset:]
        return cartesians

    def loadMetaFile(self):
        """
        This function loads the meta_file for needle_passing which gives the category-wise score for each demonstration along with the total score
        """
        scores = {}
        metaFilePath = self.dataPath + "/meta_file_Needle_Passing.txt"
        metaFilePath = self.dataPath + "/meta_file_Suturing.txt"
        for name in glob.glob(metaFilePath):
            df = np.array(pd.read_csv(name, delimiter='\t', engine='python', header=None))
            for i in range(df.shape[0]):
                scores[df[i][0]] = []
                scores[df[i][0]].append(df[i,1:])

            return scores


    def temporalizeData(self):
        """
        This function will be called after loading the pickle data, it will temporalize the data using the build-in timeseriesgenerator
        """
        batch_size = 32 #prev batch_size=128
        timesteps = 60#150

        data =  self.all_demonstrations
        target =  self.all_labels

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.getSequenceLengths(data, target)
        print ("X_train shape {}".format(X_train.shape))
        X_train_temporalize = []
        y_train_temporalize = []
        X_valid_temporalize = []
        y_valid_temporalize = []
        X_test_temporalize = []
        y_test_temporalize = []
        for i in range(X_train.shape[0]-timesteps):
            X_train_temporalize.append(X_train[i:i+timesteps])
            y_train_temporalize.append(y_train[i+timesteps-1])
        for i in range(X_valid.shape[0]-timesteps):
            X_valid_temporalize.append(X_valid[i:i+timesteps])
            y_valid_temporalize.append(y_valid[i+timesteps-1])
        for i in range(X_test.shape[0]-timesteps):
            X_test_temporalize.append(X_test[i:i+timesteps])
            y_test_temporalize.append(y_test[i+timesteps-1])

        X_train_temporalize = np.array(X_train_temporalize)
        y_train_temporalize = np.array(y_train_temporalize)
        X_valid_temporalize = np.array(X_valid_temporalize)
        y_valid_temporalize = np.array(y_valid_temporalize)
        X_test_temporalize = np.array(X_test_temporalize)
        y_test_temporalize = np.array(y_test_temporalize)

        print ("x_train_temporalized shape {}".format(X_train_temporalize.shape))
        print ("y_train_temporalized shape {}".format(y_train_temporalize.shape))
        self.dumptrainData("/pickles", X_train_temporalize, y_train_temporalize, "train%s"%(timesteps))
        self.dumptrainData("/pickles", X_valid_temporalize, y_valid_temporalize, "valid%s"%(timesteps))
        self.dumptrainData("/pickles", X_test_temporalize, y_test_temporalize, "test%s"%(timesteps))

        print ("previous shapes {} {}".format(X_train.shape, X_valid.shape))
        #return train_data, val_data, test_data

    def checkDataIntergrity(self):
        for i in range(len(self.all_demonstrations)):
            print (self.all_demonstrations[i].shape)
            print (self.all_labels[i].shape)
        print (i)

    def useExperimentalSetup(self):
        """
        Will organize data using the pre-defined experimental setup
        """



path = os.path.abspath(os.path.dirname(sys.argv[0]))
#npds = needlePassing(path)
