import os, sys, glob, graphviz
import numpy as np
import pandas as pd

class cfg:
    def __init__(self, path_):
        self.data_path = path_
        self.task = "Suturing"
        gesture_matrix = self.definegestureMatrix()
        self.iterateTranscripts(gesture_matrix)

    def definegestureMatrix(self):
        """
        gesture matrix
        """
        gestures = ["G%s"%string_ for string_ in range(14)]
        dict_1 = dict()
        dict_2 = dict()
        for gesture in gestures:
            dict_1[gesture] = np.zeros((len(gestures)))
            dict_2[gesture] = np.zeros((len(gestures)))
        matrix = np.zeros((len(gestures), len(gestures)))
        return matrix

    def iterateTranscripts(self, gesture_matrix):
        """
        iterate over transcripts
        """

        transcript_path = os.path.join(self.data_path, self.task, "transcriptions", "*.txt")
        for name in glob.glob(transcript_path):
            transcript_gestures = self.readTranscripts(name)
            first_gesture = int(transcript_gestures[0].replace("G",""))
            gesture_matrix[0][first_gesture] +=1
            for i in range(len(transcript_gestures)-1):
                current_gesture = int(transcript_gestures[i].replace("G",""))
                next_gesture = int(transcript_gestures[i+1].replace("G",""))
                gesture_matrix[current_gesture][next_gesture] +=1
            last_gesture = 13
            gesture_matrix[next_gesture][last_gesture] +=1
        print (gesture_matrix)
        for i in range(gesture_matrix.shape[0]):
            if sum(gesture_matrix[i])>0:
                gesture_matrix[i] = gesture_matrix[i]/sum(gesture_matrix[i])
        print (gesture_matrix)
        self.getgraphs(gesture_matrix)
        return gesture_matrix

    def getgraphs(self, gesture_matrix):
        """
        plot graphs
        """
        dot = graphviz.Digraph()
        for i in range(gesture_matrix.shape[0]):
            if i==0 and sum(gesture_matrix[i])>0:
                dot.node("Start", fontsize="35", shape="box"); dot.node("End", fontsize="35", shape="box")
            elif i==13 and sum(gesture_matrix[i])>0:
                dot.node("End", fontsize="60")
            elif sum(gesture_matrix[i])>0:
                dot.node("{}".format(i), "G{}".format(i),fontsize="30")
        traversed = np.zeros((gesture_matrix.shape[0], gesture_matrix.shape[1]))
        #print (gesture_matrix[:,9])
        for i in range(gesture_matrix.shape[0]):
            for j in range(gesture_matrix.shape[1]):

                if (gesture_matrix[i][j]>0) and traversed[i][j]==0:# and traversed[j][i]==0:
                    if i==8:
                        ( (gesture_matrix[i]))#dot.edge("Start", "%d"%(j), label="%.2f"%(gesture_matrix[i][j]), fontsize="30")
                    if i==0:
                        dot.edge("Start", "%d"%(j), label="%.2f"%(gesture_matrix[i][j]), fontsize="30")
                    elif j==13:
                        dot.edge("%d"%(i), "End", label="%.2f"%(gesture_matrix[i][j]), fontsize="30")
                    else:
                        dot.edge("%d"%(i), "%d"%(j), label="%.2f"%(gesture_matrix[i][j]), fontsize="30")
                    traversed[i][j] = 1
                    #traversed[j][i] = 1

        dot.render("/home/student/Documents/samin/detection/test_output")

    def readTranscripts(self, itr_path):
    	"""
    	This function reads the train and test file for each iteration
    	"""
    	try:
    		df = np.array(pd.read_csv(itr_path, delimiter=' ', header=None, engine="python"))
    		return df[:,-2]
    	except:
    		print ("File not found for directory ",itr_path)
    		pass
    	return []




path_ = os.path.abspath(os.path.dirname(sys.argv[0]))
cfg_ =cfg(path_)
