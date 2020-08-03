import os, sys, glob, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, roc_curve, auc, jaccard_score, classification_report

class gestureSpecific:
    def __init__(self, path):
        self.path = path
        self.task = "Suturing"
        self.kinvars = "NoGestures"
        self.jaccard_path = "Jaccard_v6%s1004"%self.kinvars
        self.model_name = "914839"
        self.mode = "evaluation"
        self.total_failure = 0
        self.total_misses = 0
        self.total_gestures = 0

    def iterateEvaluation(self):
        evaluation_path = os.path.join(self.path, self.task, "%s%s"%(self.jaccard_path, self.model_name),  "%s*"%(self.mode), "all*")
        print(evaluation_path)
        gesture_latency = dict()

        for name in sorted(glob.glob(evaluation_path)):
            print (name.split("/")[-1])
            evaluation_array = self.readTranscripts(name)
            gesture_dict, gesture_latency = self.getGestureBoundaries(evaluation_array, gesture_latency)
            gesture_latency = self.getLatency(gesture_dict, evaluation_array, gesture_latency)
        for key in gesture_latency.keys():
            gesture_latency[key] = sum(gesture_latency[key])/len(gesture_latency[key]) if len(gesture_latency[key])>0 else 0
        print (gesture_latency)
        print (self.total_failure)
        print (self.total_misses)

    def getGestureBoundaries(self, evaluation_array, gesture_latency):
        gesture_column = evaluation_array[:,1]
        gestures = np.unique(gesture_column)

        gesture_dict = dict()
        for gesture in gestures:
            if gesture not in gesture_dict.keys():
                gesture_dict[gesture] = list()
            if gesture not in gesture_latency.keys():
                gesture_latency[gesture] = list()
            candidate_boundaries = np.where(gesture_column==gesture)[0]
            gesture_dict[gesture].append(candidate_boundaries[0])
            for i in range(len(candidate_boundaries)-1):
                if candidate_boundaries[i+1]-candidate_boundaries[i]>5:
                    gesture_dict[gesture].append(candidate_boundaries[i])
                    gesture_dict[gesture].append(candidate_boundaries[i+1])
            gesture_dict[gesture].append(candidate_boundaries[i+1])

        return gesture_dict, gesture_latency

    def getLatency(self, gesture_dict, evaluation_array, gesture_latency):

        for key in gesture_dict.keys():
            boundaries = gesture_dict[key]

            for i in range(0, len(boundaries)-1, 2):
                actual_positive = evaluation_array[boundaries[i]:boundaries[i+1], 3]
                predicted_positive = evaluation_array[boundaries[i]:boundaries[i+1], 5]

                actual_index = np.where(actual_positive==1)[0]
                predicted_index = np.where(predicted_positive>=0.5)[0]
                self.total_gestures += 1 if boundaries[i+1]-boundaries[i]>0 else 0

                if len(actual_index)>0 and len(predicted_index)>0:
                    print (actual_index[0]-predicted_index[0])
                    gesture_latency[key].append(actual_index[0]-predicted_index[0])
                    self.total_failure +=1
                elif len(actual_index)>0 and len(predicted_index)<=0:
                    self.total_failure +=1
                    self.total_misses +=1
                    print ("gesture {} range {} {} total misses".format(key, boundaries[i], boundaries[i+1]),self.total_misses)




        return gesture_latency

    def readTranscripts(self, itr_path):
        """
        This function reads the train and test file for each iteration
        """
        try:
            df = np.array(pd.read_csv(itr_path, delimiter=',', engine="python"))
            return df
        except:
            print ("File not found for directory ",itr_path)
            pass
        return []


path = os.path.abspath(os.path.dirname(sys.argv[0]))
gsp = gestureSpecific(path)
gsp.iterateEvaluation()
