import os, sys, glob, pickle, time
#from pylab import rcParams
import numpy as np
import pandas as pd
import keras as K
from keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, roc_curve, auc, jaccard_score, classification_report
from experimental_setup import experimentalSetup
from relabelledSuboptimals import newSuboptimals
import matplotlib.pyplot as plt

class runPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.start_time = 0; self.end_time = 0
        self.exp = experimentalSetup
        self.kinvars = "CRG"
        self.jaccard_path = "Jaccard_v6%s923"%self.kinvars
        self.model_name = "914044" #"9141243"

    def loadDemonstrations(self, task):
        """
        loads all the demonstrations from Suturing_B001 to Suturing_I004
        """
        task = "Suturing"
        self.exp = experimentalSetup(self.data_path, task=task, run=1)
        self.exp.loadDemonstrations()
        all_demonstrationdict = self.exp.all_demonstrationdict
        return all_demonstrationdict

    def loadGestureClassifier(self, user_out):
        """
        Loads a specific gesture classification model
        """
        model = load_model("/home/student/Documents/samin/detection/Suturing/GestureClassification/8132051/checkpoints/exp_segment_classifier_BalancedSuperTrialOut%s_Outconvlstm_ae.h5"%user_out)
        scaler_path = "/home/student/Documents/samin/detection/Suturing/GestureClassification/8132051/checkpoints/scaler%s_Out.p"%user_out
        scaler = (pickle.load(open(scaler_path, "rb" )))
        return model, scaler


    def loadAnomalyDetector(self, user_out):
        """
        Loads all anomaly detectors into a dict
        """
        gestures = ["G1", "G2", "G3","G4", "G5", "G6", "G8", "G9"]

        anomaly_models = dict()
        anomaly_scalers = dict()
        for gesture in gestures:
            #anomaly_models[(gesture)] = load_model("/home/student/Documents/samin/detection/Suturing/aggregate_aucs/All/911334/%s/checkpoints_0.0001_5_1_k0/clf_checkpoint_0gesture%sGBE.h5"%(user_out,gesture))
            print ("/home/student/Documents/samin/detection/Suturing/aggregate_aucs/All/911334/scaler%s_%s.p"%(gesture, user_out))
            anomaly_models[(gesture)] = load_model("/home/student/Documents/samin/detection/Suturing/archivedJaccards/aggregate_aucs/%s/%s/%s/checkpoints_0.0001_5_1_k0/clf_checkpoint_0gesture%sGBE.h5"%(self.kinvars, self.model_name, user_out,gesture))
            scaler_path = "/home/student/Documents/samin/detection/Suturing/archivedJaccards/aggregate_aucs/All/911334/scaler%s_%s.p"%(gesture, user_out) #same scaler always
            anomaly_scalers[(gesture)] = pickle.load(open(scaler_path, "rb" ))

        return anomaly_models, anomaly_scalers

    def runPipeline(self):
        """
        Loads a demonstrations, passes it first to gesture classifier and then to the anomaly detector
        """
        task = "Suturing"
        user_out = ["1", "2", "3", "4", "5"]
        write_path = os.path.join(self.data_path, task, "{}".format(self.jaccard_path))

        all_demonstration = self.loadDemonstrations(task)
        all_transcripts = dict()
        temporalized_inputdict = dict(); output_dfdict = dict(); subjectdict = dict(); anomaly_modelsdict = dict(); temporalized_unscaledinputdict = dict()
        window = 5; stride = 1
        for subject in all_demonstration.keys():
            #print ("subject {}".format(subject))
            transcript_path = os.path.join(self.data_path, task, "transcriptions", "%s.txt"%subject)
            transcript = self.readTranscripts(transcript_path)
            demonstration = all_demonstration[subject]
            demonstration, transcript_array = self.mapDemonstrationtoTranscript(demonstration, transcript)
            all_transcripts[subject] = transcript_array

        anomaly_dicts = self.getAnomalyLabels(all_transcripts)

        for user in user_out:
            anomaly_models, anomaly_scalers = self.loadAnomalyDetector(user)
            model, scaler = self.loadGestureClassifier(user)
            accuracy_list = list()
            for subject in all_demonstration.keys():
                if subject[-1] == user:
                    transcript_array = all_transcripts[subject]
                    demonstration = all_demonstration[subject]
                    anomaly_transcript = anomaly_dicts[subject]
                    #print (anomaly_transcript.shape)
                    scaled_demonstration = scaler.transform(demonstration)

                    temporalized_demonstration, temporalized_transcript = self.temporalize(scaled_demonstration, transcript_array)
                    temporalized_anomalydemonstration, temporalized_anomalytranscript = self.temporalize(demonstration, anomaly_transcript, window, stride)
                    temporalized_inputdict[subject], temporalized_unscaledinputdict[subject],  output_dfdict[subject], subjectdict[subject], anomaly_modelsdict = self.evaluatewholepipeline(subject, user, model, anomaly_models, anomaly_scalers, temporalized_demonstration, temporalized_transcript, demonstration, temporalized_anomalytranscript)

        return temporalized_inputdict, temporalized_unscaledinputdict, output_dfdict, subjectdict, anomaly_modelsdict

    def readTranscripts(self, itr_path):
        """
        This function reads the train and test file for each iteration
        """
        try:
            df = np.array(pd.read_csv(itr_path, delimiter=' ', header=None, engine="python"))
            return df
        except:
            print ("File not found for directory ",itr_path)
            pass
        return []

    def mapDemonstrationtoTranscript(self, demonstration, transcript):
        transcript_end = transcript[-1][1]; transcript_start = transcript[0][0]

        #clipped_demonstration = demonstration[transcript[0][0]:transcript[-1][1]]
        transcript_array = np.zeros((demonstration.shape[0], 15))
        for i in range(transcript.shape[0]):
            label = transcript[i][2]
            end = transcript[i][1]; start = transcript[i][0]
            transcript_array[start:end, int(label.replace("G",""))-1] = 1
            #print ("start {} end {}".format(start, end))
        #print ("clipped demonstration {} transcript array {}".format(clipped_demonstration.shape, transcript_array.shape))
        clipped_transcript = transcript_array[transcript_start:transcript_end]

        #assert (clipped_transcript.shape[0]) == clipped_demonstration.shape[0]
        return demonstration, transcript_array

    def temporalize(self, demonstration, transcript, timesteps=1, stride =1):
        """
        This function takes demonstration and transcript and temporalize them
        """
        x_train_temporalize = []
        y_train_temporalize = []
        x_test_temporalize = []
        y_test_temporalize = []

        for i in range(0, demonstration.shape[0]-timesteps, stride):
            x_train_temporalize.append(demonstration[i:i+timesteps])
            y_train_temporalize.append(transcript[i])#+timesteps-1])
        return np.asarray(x_train_temporalize), np.asarray(y_train_temporalize)

    def getAnomalyLabels(self, gesturetranscript_dict):
        """
        Calls the relabelledSuboptimals class to get gesture labels
        """
        transcript_dict = gesturetranscript_dict.copy()
        for subject in transcript_dict.keys():
            transcript_dict[subject] = np.zeros((transcript_dict[subject].shape[0],2))

        rso_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "Suturing", "gesture_videos")
        self.nso = newSuboptimals(rso_path)
        gestures = ["G1", "G2", "G3","G4", "G5", "G6", "G8", "G9"]
        for gesture_ in gestures:
            normal_dict, failure_dict =  self.nso.getLabelledGestures(gesture_)
            for key in normal_dict.keys():
                for timestamps in normal_dict[key]:
                    #print ("subject {} timestamps {}".format(key, timestamps))
                    start = timestamps[0]
                    end = timestamps[1]
                    transcript_dict[key][start:end,0] = 1
            for key in failure_dict.keys():
                for timestamps in failure_dict[key]:
                    start = timestamps[0]
                    end = timestamps[1]
                    transcript_dict[key][start:end,1] = 1
        return transcript_dict

    def evaluatewholepipeline(self, subject, model_index, classifier, anomaly_models, anomaly_scalers, temporalized_demonstration, temporalized_transcript, demonstration, anomalytranscript):
        """
        run pipeline
        """
        print ("subject {} model index {} ".format(subject.split("_")[-1].replace(".csv",""), model_index))
        if subject.split("_")[-1].replace(".csv","")[-1]==model_index:
            print ("entered")
            task = "Suturing"
            write_path = os.path.join(self.data_path, task, "{}{}".format(self.jaccard_path, self.model_name))

            if not os.path.exists(write_path):
                os.makedirs(write_path)

            temporalized_input = list(); temporalized_unscaledinput = list()
            y_test = anomalytranscript
            y_preds = []
            gesture_preds = []
            window = 5; stride = 1; grand_truth = "0"
            for i in range(temporalized_demonstration.shape[0]-window):
                #print (temporalized_demonstration[i].shape)
                self.start_time = time.time() #print (temporalized_demonstration[i].shape)
                gesture_sample = temporalized_demonstration[i].reshape(1,temporalized_demonstration[i].shape[0],  temporalized_demonstration[i].shape[1])#np.asarray([1,temporalized_demonstration])
                gesture = classifier.predict(gesture_sample).argmax(axis=1)
                if grand_truth == "1":
                    gesture = temporalized_transcript.argmax(axis=1)[i]
                gesture_preds.append(gesture)
                gesture_key = "G{}".format(int(gesture)+1)
                #print ("gesture key {} all keys {}".format(gesture_key, anomaly_scalers.keys()))
                if i>5 and gesture_key in anomaly_scalers.keys():
                    #print ("gesture {}".format(gesture))
                    scaled_anomalysample = anomaly_scalers[gesture_key].transform((demonstration[i-window:i])) #(demonstration[i:i+window])
                    temporalized_anomalysample = scaled_anomalysample.reshape(1, scaled_anomalysample.shape[0], scaled_anomalysample.shape[1])
                    if self.kinvars == "CRG":
                        temporalized_anomalysamplelist, kinvars = self.exp.getSpecificFeatures(np.asarray([temporalized_anomalysample, [], temporalized_anomalysample, []]).reshape(-1,4))# for specific features
                        temporalized_anomalysample = temporalized_anomalysamplelist[0][0]
                    temporalized_input.extend(scaled_anomalysample)

                    temporalized_unscaledinput.extend(demonstration[i-window:i])
                    y_preds.append(anomaly_models[gesture_key].predict(temporalized_anomalysample))#.argmax(axis=1))
                    print (time.time()-self.start_time)#print (anomaly_models[gesture_key].predict(temporalized_anomalysample))
                else:
                    y_preds.append([[1,0]])
                    unscaled_anomalysample = demonstration[i-window:i]
                    temporalized_anomalysample = unscaled_anomalysample.reshape(1, unscaled_anomalysample.shape[0], unscaled_anomalysample.shape[1])
                    temporalized_input.extend(unscaled_anomalysample)
                    temporalized_unscaledinput.extend(unscaled_anomalysample)

            y_preds = np.asarray(y_preds).reshape(-1,2)#anomalytranscript = np.asarray(anomalytranscript)[0:np.asarray(y_preds).shape[0]]
            anomalytranscript = np.asarray(anomalytranscript)[0:np.asarray(y_preds).shape[0]]
            gesture_preds = np.asarray(gesture_preds)

            gestureevaluation_data = np.concatenate((temporalized_transcript[0:gesture_preds.shape[0]].argmax(axis=1).reshape(-1,1), gesture_preds.reshape(-1,1)), axis=1)
            anomalyevaluation_data = np.concatenate((anomalytranscript, y_preds), axis=1)
            evaluation_data = np.concatenate((gestureevaluation_data, anomalyevaluation_data), axis=1)

            output_df = pd.DataFrame(data=evaluation_data, columns=["true_gesture", "predicted_gesture", "true_negativeclass", "true_positiveclass", "predicted_negativeclass", "predicted_positiveclass"])
            writecsv_path = "{}/evaluation_k{}fold".format(write_path, model_index)

            if not os.path.exists(writecsv_path):
                os.makedirs(writecsv_path)
            if grand_truth == "1":
                subject = "true{}".format(subject)
            output_df.to_csv("{}/{}_{}.csv".format(writecsv_path, subject, model_index))
            return temporalized_input, temporalized_unscaledinput, output_df, subject, anomaly_models

    def summarizeResults(self):
        """
        gives the auc/confusion matrix of the results
        """
        task = "Suturing"
        write_path = os.path.join(self.data_path, task, "{}{}".format(self.jaccard_path, self.model_name))
        result_path = os.path.join(self.data_path, "Suturing", "{}{}".format(self.jaccard_path, self.model_name), "evaluation_k*fold", "S*.csv")
        figure_path = os.path.join(self.data_path, "Suturing", "{}{}".format(self.jaccard_path, self.model_name))
        results_auc = dict()
        results_f1dict = dict(); gesture_f1dict = dict()
        results_tpr_fprdict = dict()
        threshold_dict = dict()
        gesture_f1dict = dict()
        preds_list = list()

        for anomaly_file in sorted(glob.glob(result_path)):
            subject = anomaly_file.split("/")[-1].replace(".csv","").replace("anomaly","")
            print (subject)

            anomaly_results = pd.read_csv(anomaly_file, index_col=0)

            preds_list = list(); test_list = list()

            preds = np.asarray(anomaly_results[["predicted_positiveclass"]])#.iloc[:,5];
            test = np.asarray(anomaly_results[["true_positiveclass"]])#.iloc[:,3]
            #print ("test {} preds {}".format(len(test) ,len(preds)))
            for i in range(len(preds)):
                #if "[" in preds[i]:
                preds_list.append(((preds[i])))
                test_list.append(((test[i])))

            #print ("appended_test {} appended_preds {}".format(len(test_list) ,len(preds_list)))

            false_pos_rate, true_pos_rate, thresholds = roc_curve(test_list, preds_list)
            threshold_dict[subject] = [false_pos_rate, true_pos_rate, thresholds]
            roc_auc = auc(false_pos_rate, true_pos_rate)
            results_auc[subject] = roc_auc
            plt.plot(false_pos_rate, true_pos_rate, label='AUC = %0.3f '% (roc_auc))


            if subject == "allSuturing_I002_2" or subject == "allSuturing_F004_4" or subject == "allSuturing_B002_2":
                print ("kepping")
                plt.plot(false_pos_rate, true_pos_rate, label='AUC = %0.3f '% (roc_auc))
            #else:
            #    continue #plt.plot(false_pos_rate, true_pois_rate, label='AUC = %0.3f '% (roc_auc))
            plt.plot([0,1],[0,1], linestyle='--')
            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower right')
            plt.title('Receiver operating characteristic curve (ROC) {}'.format(subject))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            auc_path = os.path.join(figure_path, "auck{}_fold".format(subject.split("_")[-1]))
            if not os.path.exists(auc_path):
                os.makedirs(auc_path)
            plt.savefig("{}/auc_{}.pdf".format(auc_path, subject))
            plt.close()

            results_tpr_fprdict[subject] = [(false_pos_rate, true_pos_rate)]
            #print (np.asarray(anomaly_results.iloc[:,4:6]).argmax(axis=1))
            y_test = np.asarray(anomaly_results.iloc[:,2:4]).argmax(axis=1)
            y_preds = np.asarray(anomaly_results.iloc[:,4:6]).argmax(axis=1)

            results_f1dict[subject] = (f1_score(y_preds, y_test, average='weighted', labels=np.unique(y_test)))# (f1_score(np.asarray(test_list).argmax(axis=1), np.asarray(preds_list).argmax(axis=1)))
            #results_f1dict[subject] = (f1_score(np.asarray(anomaly_results.iloc[:,2:4]).argmax(axis=1), np.asarray(anomaly_results.iloc[:,4:6]).argmax(axis=1)))
            #print (results_f1dict[subject])
            gesture_conf = confusion_matrix(anomaly_results[["true_gesture"]], anomaly_results[["predicted_gesture"]])
            gesture_conf = gesture_conf.astype('float') / gesture_conf.sum(axis=1)[:, np.newaxis]
            true_gesturearray = np.asarray(anomaly_results[["true_gesture"]]).reshape(-1,1)
            predicted_gesturearray = np.asarray(anomaly_results[["predicted_gesture"]]).reshape(-1,1)

            self.getjittervalues(predicted_gesturearray, true_gesturearray)

            gesture_f1 = f1_score(true_gesturearray, predicted_gesturearray, labels = np.unique(true_gesturearray), average=None)
            #print (gesture_f1)
            gesture_f1dict[subject] =  (gesture_f1)
            labelsize = 16
            cmap = plt.cm.Blues
            fig, ax = plt.subplots()
            im = ax.imshow(gesture_conf, interpolation='nearest', cmap=cmap)

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label(label="Normalized Accuracy", size=16)
            cbar.ax.tick_params(labelsize=labelsize)
            # We want to show all ticks...
            ax.set(xticks=np.arange(gesture_conf.shape[1]), yticks=np.arange(gesture_conf.shape[0]), xticklabels = ["G{}".format(int(name)+1) for name in np.unique(true_gesturearray)], yticklabels = ["G{}".format(int(name)+1) for name in np.unique(true_gesturearray)])        #plt.show()

            plt.tick_params(labelsize=labelsize); #xlabel('Gestures', fontsize=20) ##plt.show()
            plt.xlabel("Predicted Gesture Index", fontsize=labelsize); #xlabel('Gestures', fontsize=20) ##plt.show()
            plt.ylabel("True Gesture Index", fontsize=labelsize); #xlabel('Gestures', fontsize=20) ##plt.show()

            plt.savefig("{}/gestureconf{}.pdf".format(figure_path, subject)) #ax.set(xticks=np.arange(gesture_conf.shape[1]), yticks=np.arange(gesture_conf.shape[0]))        #plt.show()
            plt.close()

        results_aucdf = pd.DataFrame.from_dict(results_auc, orient='index')
        #print (results_f1dict)# = pd.DataFrame.from_dict(results_auc, orient='index')
        #results_aucdf.to_csv("{}/{}.csv".format(write_path, "results_auc"))
        results_f1df = pd.DataFrame.from_dict(results_f1dict, orient='index')
        gesture_f1df = pd.DataFrame.from_dict(gesture_f1dict, orient='index')
        gesture_f1df.to_csv("{}/{}.csv".format(write_path, "gesture_f1"))
        results_f1df.to_csv("{}/{}.csv".format(write_path, "results_f1"))
        results__tpr_fprdf = pd.DataFrame.from_dict(results_tpr_fprdict)
        results__tpr_fprdf.to_csv("{}/{}.csv".format(write_path, "results__tpr_fpr"))

        pickle.dump(threshold_dict, open( "{}/{}.p".format(figure_path, "truegesturedict"), "wb" ))
        return results_auc

    def getjittervalues(self, preds, ytest):

        """
        Finds average jitter y
        """
        classes = np.unique(ytest)
        true_indiceslist = list()
        pred_indiceslist = list()
        for gesture in classes:
            true_indices = np.where(ytest == gesture)[0];true_indiceslist = list()
            pred_indices = np.where(preds == gesture)[0];pred_indiceslist = list()
            prev = true_indices[0]
            for i in range(len(true_indices)):
                if true_indices[i]-prev>50:
                    true_indiceslist.append(true_indices[i])
                prev = true_indices[i]

            for i in range(len(pred_indices)):
                if pred_indices[i]-prev>50:
                    pred_indiceslist.append(pred_indices[i])
                prev = pred_indices[i]

            #print (true_indiceslist, pred_indiceslist)
        return true_indiceslist, pred_indiceslist

    def segmentBoundaries(self, true_gestures):
        prev_gesture = true_gestures[0]; gesture_boundaries = list()
        for i in range(len(true_gestures)):
		        if true_gestures[i]!= prev_gesture:
		  		      gesture_boundaries.append(i)
		  		      prev_gesture = true_gestures[i]
        gesture_boundaries.append(i)
        print (gesture_boundaries)
        return gesture_boundaries

    def findreactiontimebygesture(self, preds, test, jitter_gesturelist):

        trueerror_begin = np.where(test!=0)[0]; prederror_begin = np.where(preds!=0)[0];
        if not len(trueerror_begin)>0:
        		return jitter_gesturelist
        trueerror_packets = list(); trueerror_packets.append(trueerror_begin[0])
        prederror_packets = list(); prederror_packets.append(prederror_begin[0])

        prev = trueerror_begin[0]
        for i in range(1,len(trueerror_begin)-1):
            if trueerror_begin[i]-prev>5:
                trueerror_packets.append(trueerror_begin[i])
            prev = trueerror_begin[i]

        prev = prederror_begin[0]
        for i in range(1,len(prederror_begin)-1):
            if prederror_begin[i]-prev>5:
                prederror_packets.append(prederror_begin[i])
            prev = prederror_begin[i]

        for i in range(len(trueerror_packets)):
            current_packet = trueerror_packets[i]
            diff = np.asarray([abs(current_packet-prederror_packets)])
            min_index = diff.argmin(axis=1)[0]
            min_value = prederror_packets[min_index]
            jitter_gesturelist.append(min_value-current_packet)
            print ("current packet {} predicted packet {} diff {}".format(current_packet, min_value, current_packet-min_value))

        return jitter_gesturelist

    def findeventoverlap(self):
        """
        Finds the number of occasion where there is no overlap between predicted anomaly and actual anomaly
        """

        task = "Suturing"
        write_path = os.path.join(self.data_path, "Suturing", "{}{}".format(self.jaccard_path, self.model_name))
        result_path = os.path.join(self.data_path, "Suturing", "{}{}".format(self.jaccard_path, self.model_name), "evaluation_k*fold", "S*.csv")
        results_auc = self.summarizeResults()


        preds_list = list()
        jitter = dict()
        for anomaly_file in sorted(glob.glob(result_path)):
            subject = anomaly_file.split("/")[-1].replace(".csv","").replace("anomaly","")
            print (anomaly_file)
            jitter[subject] = list()
            anomaly_results = pd.read_csv(anomaly_file, index_col=0)
            preds_list = list(); test_list = list()
            diff =  (np.asarray(anomaly_results[["predicted_positiveclass"]]).reshape(-1,1) -  np.asarray(anomaly_results.iloc[:,5]).reshape(-1,1))
            boundaries = self.segmentBoundaries(anomaly_results.iloc[:,0])
            preds = np.asarray(anomaly_results.iloc[:,4:6]); test = np.asarray(anomaly_results.iloc[:,2:4])
            jitter_gesture = dict()
            for i in range(len(boundaries)-1):
                gesture = "G{}".format(anomaly_results.iloc[:,0])
                jitter_gesture[gesture] = list() if gesture not in jitter_gesture.keys() else jitter_gesture[gesture]
                jitter_gesture[gesture] = self.findreactiontimebygesture(preds[boundaries[i]:boundaries[i+1]], test[boundaries[i]:boundaries[i+1]], jitter_gesture[gesture])
            preds_list.append(jaccard_score(preds.argmax(axis=1), test.argmax(axis=1), average='binary'))
            preds = preds.argmax(axis=1)
            test = test.argmax(axis=1)
            print (jitter_gesture)

            trueerror_begin = np.where(test!=0)[0]; prederror_begin = np.where(preds!=0)[0]
            trueerror_packets = list(); trueerror_packets.append(trueerror_begin[0])
            prederror_packets = list(); prederror_packets.append(prederror_begin[0])
            #print (trueerror_begin)
            prev = trueerror_begin[0]
            for i in range(1,len(trueerror_begin)-1):
                if trueerror_begin[i]-prev>5:
                    trueerror_packets.append(trueerror_begin[i])
                prev = trueerror_begin[i]


            #print (trueerror_begin)#prev = prederror_begin[0]
            prev = prederror_begin[0]
            for i in range(1,len(prederror_begin)-1):
                if prederror_begin[i]-prev>5:
                    prederror_packets.append(prederror_begin[i])
                prev = prederror_begin[i]

            for i in range(len(trueerror_packets)):
                current_packet = trueerror_packets[i]
                diff = np.asarray([abs(current_packet-prederror_packets)])
                min_index = diff.argmin(axis=1)[0]
                min_value = prederror_packets[min_index]
                if abs(min_value-current_packet)<150:
                    jitter[subject].append(min_value-current_packet)
                #print ("current packet {} predicted packet {} diff {}".format(current_packet, min_value, current_packet-min_value))

        max_length = max([len(value) for value in jitter.values()])
        print (max_length)
        for key in jitter.keys():
            if len(jitter[key])>0:
                average = sum(jitter[key])/len(jitter[key])
                for i in range(max_length-len(jitter[key])):
                    jitter[key].append(0)
                jitter[key].append(average)
                jitter[key].append(results_auc[key])


        jitter_df = pd.DataFrame.from_dict(data=jitter, orient="index", columns= ["subject", "jitter0", "jitter1", "jitter2", "jitter3", "jitter4", "jitter5", "jitter6", "jitter7",  "average_jitter", "auc"])
        jitter_df.to_csv("{}/jitterCRG{}.csv".format(write_path, self.model_name))
        print ("{}/jitterCRG{}.csv".format(write_path, self.model_name))



path_ = os.path.abspath(os.path.dirname(sys.argv[0]))
rpl = runPipeline(path_)
#rpl.runPipeline()
#rpl.summarizeResults()
rpl.findeventoverlap()
