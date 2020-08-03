import os, sys, glob, pickle, math
import numpy as np
import pandas as pd
import keras as K
import scipy.stats as ss
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experimental_setup import experimentalSetup
from lstm_vaesuturing import lstmVAE
from vae_keras import VAE

class newSuboptimals:
    def __init__(self, data_path):
        self.data_path = data_path

    def iterateTranscripts(self, transcript_path):
        """
        Iterates over all transcript files to collect transcripts
        """
        self.unsequenced_transcript = dict()
        self.transcript_dict = dict()
        transcript_path = os.path.join(transcript_path, "*.txt")
        #print (transcript_path)
        for name in glob.glob(transcript_path):
            #print ("directory name {}".format(name))
            subject_name = name.split("/")[-1].replace(".txt","")
            self.unsequenced_transcript[subject_name] = self.readTranscripts(name)


    def readTranscripts(self, itr_path, delim=' '):
        """
        This function reads the train and test file for each iteration
        """
        try:
            df = np.array(pd.read_csv(itr_path, delimiter=delim, header=None, engine="python"))
            return df
        except:
            print ("File not found for directory ",itr_path)
            pass
        return []

    def getLabelledGestures(self, gesture_):
        gesture_path = os.path.join(self.data_path, gesture_, "*.csv")
        failure_df = []
        for name in glob.glob(gesture_path):
            failure_df = pd.read_csv(name)
        failure_dict = dict()
        normal_dict = dict()
        if len(failure_df)>0:
            failure_array = np.asarray(failure_df)
            for i in range(1, failure_array.shape[0]):

                start_time = int(failure_array[i][0].split("_")[-2])
                subject = "{}_{}".format(failure_array[i][0].split("_")[0], failure_array[i][0].split("_")[1])   #replace(str(start_time), "").split("__")[0]
                end_time = int(failure_array[i][0].split("_")[-1])

                if failure_array[i][3] != "0":
                    if not subject in failure_dict.keys():
                        failure_dict[subject] = []
                    failure_dict[subject].append([start_time, end_time])
                else:
                    if not subject in normal_dict.keys():
                        normal_dict[subject] = []
                    normal_dict[subject].append([start_time, end_time])


        return normal_dict, failure_dict

    def temporalizeGestures(self, gesture_array, timesteps, stride):
        x_train_temporalize = []

        for i in range(0, len(gesture_array)-timesteps, stride):
            x_train_temporalize.append(gesture_array[i:i+timesteps])
            #i +=stride
        return (np.asarray(x_train_temporalize))


    def distinguishSequence(self, all_kinematics, all_labels):

        anomalous_trajectory = np.asarray(all_kinematics)[np.where(all_labels>0)[0]]
        nonanomalous_trajectory = np.asarray(all_kinematics)[np.where(all_labels==0)[0]]
        anomalous_label = np.asarray(all_labels)[np.where(all_labels>0)[0]]
        nonanomalous_label = np.asarray(all_labels)[np.where(all_labels==0)[0]]

        return nonanomalous_trajectory, anomalous_trajectory, nonanomalous_label, anomalous_label

    def splitdata(self, x_train, x_label):
        """
        splits the data to train set and validation set
        """
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, x_label, test_size=0.2, random_state=42, shuffle=True)
        return x_train, x_valid, y_train, y_valid

    def scaleData(self, x_train, x_test, gesture):
        """
        This function scales the training and testing data
        """
        scaler = StandardScaler().fit(x_train)
        #pickle.dump(scaler, open("scaler{}.p".format(gesture), "wb"))
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test

    def temporalizeData(self, x_train, y_train, x_test, y_test, timesteps=1):

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

    def runExperiments(self):
        gesture_based = "1"
        encoder1_list = [512]#, 256, 128] #[1024, 512]#, 256, 128]
        encoder4 = 16
        lr_list = [0.0001] #0.1, 0.01,
        for encoder1 in encoder1_list:
            encoder2_list = [128]#, 64, 32]# [encoder1//2, encoder1//4, encoder1//8]
            for encoder2 in encoder2_list:
                encoder3_list = [64]#, 64]#, 16] #[encoder2//2, encoder2//4, encoder2//8]
                for encoder3 in encoder3_list:
                    for lr in lr_list:
                        nso.prepareLOSO(encoder1, encoder2, encoder3, encoder4, lr, gesture_based)

    def prepareLOSO(self, encoder1, encoder2, encoder3, encoder4, lr, gesture_based="1"):
        """
        Will get user-wise distribution of optimals/sub-optimals
        """
        glob_path = os.path.join(self.data_path, "*")
        task = "Suturing"
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.exp = experimentalSetup(path, task=task, run=1)
        self.exp.loadDemonstrations()
        self.lstmae = lstmVAE(path, task)
        self.currentTimestamp = self.exp.currentTimestamp
        all_demonstrationdict = self.exp.all_demonstrationdict
        distribution_dict = dict(); all_distribution_dict = dict()
        test_users = ["1", "2", "3", "4", "5"]
        for test_user in test_users:
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            if gesture_based == "0":
                all_xtrain = list(); all_xtest = list(); all_ytrain = list(); all_ytest = list()
            for gesture in glob.glob(glob_path):
                print (gesture.split("/")[-1])
                normal_dict, failure_dict = self.getLabelledGestures(gesture_=gesture)
                train_loso = list() ; ytrain_loso = list()#dict()
                test_loso = list(); ytest_loso = list()#dict()
                distribution_list = list()
                distribution_dict[gesture.split("/")[-1]] = list()
                all_distribution_dict[gesture.split("/")[-1]] = list()
                for key in normal_dict.keys():
                    if key.split("_")[-1][-1] == test_user:
                        for timestamps in normal_dict[key]:
                            test_loso.extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            ytest_loso.extend(np.zeros((timestamps[1]-timestamps[0])))
                            all_distribution_dict[gesture.split("/")[-1]].extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])

                    else:
                        for timestamps in normal_dict[key]:
                            train_loso.extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            ytrain_loso.extend(np.zeros((timestamps[1]-timestamps[0])))
                            all_distribution_dict[gesture.split("/")[-1]].extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])

                train_timestamps = 0; test_timestamps = 0;
                for key in failure_dict.keys():
                    if key.split("_")[-1][-1] == test_user:
                        #print (key)
                        for timestamps in failure_dict[key]:
                            test_loso.extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            ytest_loso.extend(np.ones((timestamps[1]-timestamps[0])))
                            test_timestamps += (timestamps[1]-timestamps[0])
                            #distribution_list.extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            distribution_dict[gesture.split("/")[-1]].extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            all_distribution_dict[gesture.split("/")[-1]].extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                    else:
                        for timestamps in failure_dict[key]:
                            train_loso.extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            ytrain_loso.extend(np.ones((timestamps[1]-timestamps[0])))
                            train_timestamps += (timestamps[1]-timestamps[0])
                            #distribution_list.extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            distribution_dict[gesture.split("/")[-1]].extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])
                            all_distribution_dict[gesture.split("/")[-1]].extend(all_demonstrationdict[key][timestamps[0]:timestamps[1]])

                print ("train samples {} test samples {}".format(train_timestamps/len(ytrain_loso), test_timestamps/len(ytest_loso)))
                kfold_distribution = np.asarray([gesture.split("/")[-1], len(ytrain_loso), len(ytest_loso), train_timestamps/len(ytrain_loso), test_timestamps/len(ytest_loso)])
                #print (kfold_distribution)
                kfold_distributiondf = pd.DataFrame(kfold_distribution, columns = [["gesture, len(ytrain_loso), len(ytest_loso), train_timestamps/len(ytrain_loso), test_timestamps/len(ytest_loso"]])
                kfold_distributiondf.to_csv("{}/{}_{}.csv".format("/home/student/Documents/samin/detection/Suturing/losodists", (gesture.split("/")[-1]), test_user))

                #self.pltdistribution(np.asarray(distribution_list), gesture, ax)
                if gesture_based == "1":
                    train_loso = np.asarray(train_loso)
                    test_loso = np.asarray(test_loso)
                    gesture_ = gesture.split("/")[-1]; kinvars = "All"
                    distribution_dict[gesture.split("/")[-1]] = np.asarray(distribution_dict[gesture.split("/")[-1]])
                    all_distribution_dict[gesture.split("/")[-1]] = np.asarray(all_distribution_dict[gesture.split("/")[-1]])
                    scaled_train_loso, scaled_test_loso = self.scaleData(train_loso, test_loso,("{}_{}".format(gesture.split("/")[-1], test_user)))
                    temporalized_scaled_train_loso = list(); temporalized_scaled_test_loso = list(); temporalized_ytrain_loso = list(); temporalized_ytest_loso = list()
                    window = 5; stride = 1

                    temporalized_scaled_train_loso = np.asarray(self.temporalizeGestures(scaled_train_loso, window, stride))
                    temporalized_scaled_test_loso = np.asarray(self.temporalizeGestures(scaled_test_loso, window, stride))
                    temporalized_ytrain_loso = np.asarray(self.temporalizeGestures(ytrain_loso, window, stride))
                    temporalized_ytest_loso = np.asarray(self.temporalizeGestures(ytest_loso, window, stride))

                    specific = "0"
                    if specific == "1":
                        data, kinvars = self.exp.getSpecificFeatures(np.asarray([temporalized_scaled_train_loso, temporalized_ytrain_loso, temporalized_scaled_test_loso, temporalized_ytest_loso]).reshape(1,4))
                        temporalized_scaled_train_loso, temporalized_ytrain_loso, temporalized_scaled_test_loso, temporalized_ytest_loso = data[0]
                        #print ("failure gesture {} xtest {}".format(temporalized_scaled_train_loso.shape, temporalized_scaled_test_loso.shape))

                    vae_data = np.asarray([temporalized_scaled_train_loso, temporalized_scaled_test_loso, temporalized_ytrain_loso, temporalized_ytest_loso]).reshape(-1,4)
                    #print ("train shape {} y_train {} test shape {} y test {}".format(temporalized_scaled_train_loso.shape, temporalized_ytrain_loso.shape, temporalized_scaled_test_loso.shape, temporalized_ytest_loso.shape))
                    #print ("train shape", temporalized_ytrain_loso)
                    self.lstmae.losotrainClassifier(self.currentTimestamp, window, stride, vae_data[0], trial=test_user, itr_num=0, gesture="{}".format(gesture_),
                    train=0, mode="GBE", kinvars=kinvars,encoder1=encoder1, encoder2=encoder2, encoder3=encoder3, encoder4=encoder4, lr = lr)
                else:
                    print ("extending non-specific gesture list")
                    all_xtrain.extend(train_loso); all_xtest.extend(test_loso);
                    all_ytrain.extend(ytrain_loso); all_ytest.extend(ytest_loso)

            if gesture_based == "0":
                print ("about to send all for training")
                all_xtrain = np.asarray(all_xtrain); all_xtest = np.asarray(all_xtest); all_ytrain = np.asarray(all_ytrain); all_ytest = np.asarray(all_ytest)
                scaled_train_loso, scaled_test_loso = self.scaleData(all_xtrain, all_xtest,("{}_{}".format("All", test_user)))
                temporalized_scaled_train_loso = list(); temporalized_scaled_test_loso = list(); temporalized_ytrain_loso = list(); temporalized_ytest_loso = list()
                window = 5; stride = 1
                kinvars = "All"
                temporalized_scaled_train_loso = np.asarray(self.temporalizeGestures(scaled_train_loso, window, stride))
                temporalized_scaled_test_loso = np.asarray(self.temporalizeGestures(scaled_test_loso, window, stride))
                temporalized_ytrain_loso = np.asarray(self.temporalizeGestures(all_ytrain, window, stride))
                temporalized_ytest_loso = np.asarray(self.temporalizeGestures(all_ytest, window, stride))
                specific = "0"
                if specific == "1":
                    data, kinvars = self.exp.getSpecificFeatures(np.asarray([temporalized_scaled_train_loso, temporalized_ytrain_loso, temporalized_scaled_test_loso, temporalized_ytest_loso]).reshape(1,4))
                    temporalized_scaled_train_loso, temporalized_ytrain_loso, temporalized_scaled_test_loso, temporalized_ytest_loso = data[0]
                allvae_data = np.asarray([temporalized_scaled_train_loso, temporalized_scaled_test_loso, temporalized_ytrain_loso, temporalized_ytest_loso]).reshape(-1,4)
                print ("train shape {} y_train {} test shape {} y test {}".format(temporalized_scaled_train_loso.shape, temporalized_ytrain_loso.shape, temporalized_scaled_test_loso.shape, temporalized_ytest_loso.shape))
                self.lstmae.losotrainClassifier(self.currentTimestamp, window, stride, allvae_data[0], trial=test_user, itr_num=0, gesture="{}".format("All"),
                train=0, mode="non-GBE", kinvars=kinvars,encoder1=encoder1, encoder2=encoder2, encoder3=encoder3, encoder4=encoder4, lr = lr)

            self.getJSdivergence(all_distribution_dict)
            plt.savefig("alljsdiff.pdf")

    def pltdistribution(self, gesture_distribution, gesture_, ax):
        """
        receives distributions and plots them
        """
        print (gesture_distribution.shape)
        gesture_cartesian = gesture_distribution[:,0:2]
        X = gesture_cartesian[:,0]#.reshape(-1,1)
        Y = gesture_cartesian[:,1]#.reshape(-1,1)
        pos = np.dstack((X,Y))
        cov = np.cov(gesture_cartesian.T)
        mean = gesture_cartesian.mean(axis=0)
        rv = multivariate_normal(mean, cov)
        Z = rv.pdf(pos)#.reshape(-1,1)
        ax.plot(X, Y, Z, markersize=2, label=gesture_.split("/")[-1])

    def getJSdivergence(self, gesture_dict):
        """
        Gets the gesture distribution, converts to kde
        """
        all_gestures = list(); kernel_dict = dict()
        for key in gesture_dict.keys():
            all_gestures.extend(gesture_dict[key])
        all_gestures = np.asarray(all_gestures)
        #print (all_gestures.shape)
        all_max = np.amax(all_gestures, axis=0)
        all_min = np.amin(all_gestures, axis=0)
        #print (all_max)
        #print (all_min)
        grid_space = 50
        grid = np.zeros((all_gestures.shape[1],grid_space))

        for i in range(all_gestures.shape[1]):
            grid[i] = np.linspace(all_min[i], all_max[i], num=grid_space)

        #print ("all_max shape {} all_min shape {} grid shape {}".format(all_max.shape, all_min.shape, grid.shape))

        for key in gesture_dict.keys():
            #if key == "G1":
            #    print (gesture_dict[key])
            #    print (gesture_dict[key].shape)
            gesture_kernel = ss.gaussian_kde(gesture_dict[key].T)
            kernel_dict[key] = gesture_kernel.pdf(grid)#evaluate(grid)
            #print (key, kernel_dict[key])
            #print (gesture_dict[key].shape)
        js_grid = np.zeros((len(gesture_dict), len(gesture_dict)))
        count1 = 0

        for key1 in sorted(gesture_dict.keys()):
            count2 = 0
            for key2 in sorted(gesture_dict.keys()):
                js_grid[count1][count2] = (distance.jensenshannon(kernel_dict[key1], kernel_dict[key2]))
                #if math.isnan(js_grid[count1][count2]):
                #    js_grid[count1][count2] = 0
                count2 +=1
            count1 +=1
        count1 = 0
        js_list = list()
        js_keylist1 = list()
        js_keylist2 = list()
        for key1 in sorted(gesture_dict.keys()):
            count2 = 0
            for key2 in sorted(gesture_dict.keys()):
                if not math.isnan(js_grid[count1][count2]):
                    js_list.append(distance.jensenshannon(kernel_dict[key1], kernel_dict[key2]))
                    js_keylist1.append(key1)
                    js_keylist2.append(key2)
                count2 +=1
            count1 +=1
        js_grid = np.asarray(js_list).reshape(7,7)
        print (np.asarray(js_list).reshape(7,7))
        print ((js_keylist1))
        labelsize = 16
        cmap=plt.cm.Blues
        fig, ax = plt.subplots()
        im = ax.imshow(js_grid, interpolation='nearest', cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(label="JS Divergence", size=labelsize)
        cbar.ax.tick_params(labelsize=labelsize)
        # We want to show all ticks...
        ax.set(xticks=np.arange(js_grid.shape[1]), yticks=np.arange(js_grid.shape[0]), xticklabels = (set(js_keylist1)), yticklabels = (set(js_keylist1)), ylabel='Gesture Index',  xlabel='Gesture Index', )

        #plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
        fmt = '.2f' #if normalize else 'd'
        thresh = js_grid.max() / 2.
        for i in range(js_grid.shape[0]):
            for j in range(js_grid.shape[1]):
                ax.text(j, i, format(js_grid[i, j], fmt),
                        ha="center", va="center",
                    color="white" if js_grid[i, j] > thresh else "black")

        plt.tick_params(labelsize=labelsize); #xlabel('Gestures', fontsize=20) ##plt.show()
        plt.xlabel("Gesture Index", fontsize=labelsize); #xlabel('Gestures', fontsize=20) ##plt.show()
        plt.ylabel("Gesture Index", fontsize=labelsize); #xlabel('Gestures', fontsize=20) ##plt.show()
        plt.title("Relative Entropy between gestures", fontsize=(labelsize+2))
        jsfiddpd = pd.DataFrame(data=js_list)
        jsfiddpd.to_csv("jsdifff.csv")
        #fig.tight_layout()
        #print (js_grid)



path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "Suturing", "gesture_videos")
nso = newSuboptimals(path)
nso.runExperiments()
#nso.runGestureEvaluation()
#normal_dict, failure_dict = nso.getLabelledGestures(gesture_="G4")
#print (normal_dict)
