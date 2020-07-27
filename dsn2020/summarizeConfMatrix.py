import numpy as np
import pandas as pd
import glob, os, sys


class summarizeConf:
    def __init__(self, path):
        self.data_path = path

    def iteratePaths(self):
        iterate_path = os.path.join(self.data_path, "Suturing", "experiments", "10_Outnogestures", "csvs")
        csv_path = os.path.join(iterate_path, "conf*")
        tpr_rate = list()
        tnr_rate = list()
        for name in glob.glob(csv_path):
            conf_matrix = pd.read_csv(name, index_col=0)

            tpr_rate.append(conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0]))

            tnr_rate.append(conf_matrix.iloc[1][1]/(conf_matrix.iloc[0][1]+conf_matrix.iloc[1][1]))
        print (sum(tpr_rate)/len(tpr_rate))
        print (sum(tnr_rate)/len(tnr_rate))

    def iterategesturePaths(self):
        iterate_path = os.path.join(self.data_path, "Suturing", "experiments", "10_Out", "csvs")
        csv_path = os.path.join(iterate_path, "conf*")
        tpr_rate = dict()
        tpr_sum = 0.0
        tnr_sum = 0.0
        tnr_rate = dict()
        for name in glob.glob(csv_path):

            gesture = name.split("/")[-1].replace(".csv", "")
            if gesture not in tpr_rate.keys():
                tpr_rate[gesture] = []
                tnr_rate[gesture] = []

            conf_matrix = pd.read_csv(name, index_col=0)
            tnr_value = conf_matrix.iloc[1][1]/(conf_matrix.iloc[0][1]+conf_matrix.iloc[1][1])
            tpr_rate[gesture].append(conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0]))
            tnr_sum += tnr_value
            tpr_sum += conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])

            if tnr_value != 0:
                tnr_rate[gesture].append(tnr_value)

        for key in tpr_rate.keys():
            print (key)
            #print (sum(tpr_rate[key])/len(tpr_rate[key]))
            #print (sum(tnr_rate[key])/len(tnr_rate[key])if len(tnr_rate[key])>0 else "0")

        print (tnr_sum/len(tnr_rate.values()))
        print (tpr_sum/len(tpr_rate.values()))
        #print (sum(tnr_rate[key])/len(tnr_rate[key])if len(tnr_rate[key])>0 else "0")



path = os.path.abspath(os.path.dirname(sys.argv[0]))
scg = summarizeConf(path)
scg.iteratePaths()
#scg.iterategesturePaths()
