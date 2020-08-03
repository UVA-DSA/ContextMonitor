import os, sys, glob, pickle, time
import numpy as np
import pandas as pd
import keras as K
from keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, roc_curve, auc, jaccard_score
from experimental_setup import experimentalSetup
from relabelledSuboptimals import newSuboptimals
import matplotlib.pyplot as plt

class runPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.start_time = 0 #self.data_path = data_path
        self.jaccard_path = "Jaccard_v6NoGestures1004"
        self.model_name = "914839"

    def loadDemonstrations(self, task):
        """
        loads all the demonstrations from Suturing_B001 to Suturing_I004
        """
        task = "Suturing"
        exp = experimentalSetup(self.data_path, task=task, run=1)
        exp.loadDemonstrations()
        all_demonstrationdict = exp.all_demonstrationdict
        return all_demonstrationdict


    def loadAnomalyDetector(self, user_out):
        """
        Loads all anomaly detectors into a dict
        """
        print (user_out)
        anomaly_models = load_model("/home/student/Documents/samin/detection/Suturing/loso_experiments_clfv1/All/%s/%s/checkpoints_0.0001_5_1_k0/clf_checkpoint_0gestureAllnon-GBE.h5"%(self.model_name,user_out))
        scaler_path = "/home/student/Documents/samin/detection/Suturing/loso_experiments_clfv1/All/932226/1/checkpoints_0.0001_5_1_k0/scalerAll_%s.p"%user_out
        anomaly_scalers = pickle.load(open(scaler_path, "rb" ))
        return anomaly_models, anomaly_scalers

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
            print (np.unique(gesturetranscript_dict[subject]))
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
        return transcript_dict, gesturetranscript_dict

    def runPipeline(self):
        """
        Loads a demonstrations, passes it first to gesture classifier and then to the anomaly detector
        """
        task = "Suturing"
        user_out = ["1", "2", "3", "4", "5"]
        #write_path = os.path.join(self.data_path, task, "Jaccard_v5")
        write_path = os.path.join(self.data_path, task, "{}{}".format(self.jaccard_path, self.model_name))

        if not os.path.exists(write_path):
            os.makedirs(write_path)

        all_demonstration = self.loadDemonstrations(task)
        all_transcripts = dict()
        window = 5; stride = 1
        for subject in all_demonstration.keys():
            #print ("subject {}".format(subject))
            transcript_path = os.path.join(self.data_path, task, "transcriptions", "%s.txt"%subject)
            transcript = self.readTranscripts(transcript_path)
            demonstration = all_demonstration[subject]
            demonstration, transcript_array = self.mapDemonstrationtoTranscript(demonstration, transcript)
            all_transcripts[subject] = transcript_array

        anomaly_dicts, gesture_transcripts = self.getAnomalyLabels(all_transcripts)

        for user in user_out:
            anomaly_models, anomaly_scalers = self.loadAnomalyDetector(user)

            accuracy_list = list()
            for subject in all_demonstration.keys():
                transcript_array = gesture_transcripts[subject]
                demonstration = all_demonstration[subject]
                anomaly_transcript = anomaly_dicts[subject]
                print (anomaly_transcript.shape)
                temporalized_demonstration, temporalized_transcript = self.temporalize(demonstration, transcript_array)
                temporalized_anomalydemonstration, temporalized_anomalytranscript = self.temporalize(demonstration, anomaly_transcript, window, stride)
                self.evaluatewholepipeline(subject, user, anomaly_models, anomaly_scalers,temporalized_demonstration, transcript_array, demonstration, temporalized_anomalytranscript)


    def evaluatewholepipeline(self, subject, model_index, anomaly_models, anomaly_scalers,temporalized_demonstration, temporalized_transcript, demonstration, anomalytranscript):
        """
        run pipeline
        """
        print ("subject {} model index {} ".format(subject.split("_")[-1].replace(".csv",""), model_index))
        if subject.split("_")[-1].replace(".csv","")[-1]==model_index:
            print ("entered")
            task = "Suturing"
            write_path = os.path.join(self.data_path, task, "{}{}".format(self.jaccard_path, self.model_name))

            y_test = anomalytranscript
            y_preds = []
            gesture_preds = []
            window = 5; stride = 1
            for i in range(temporalized_demonstration.shape[0]-window):
                #print (temporalized_demonstration[i].shape)
                gesture = temporalized_transcript.argmax(axis=1)[i]
                gesture_preds.append(gesture)
                if i>5:
                    self.start_time = time.time() #print ("gesture {}".format(gesture))
                    scaled_anomalysample = anomaly_scalers.transform((demonstration[i-window:i])) #(demonstration[i:i+window])
                    temporalized_anomalysample = scaled_anomalysample.reshape(1, scaled_anomalysample.shape[0], scaled_anomalysample.shape[1])
                    y_preds.append(anomaly_models.predict(temporalized_anomalysample))#.argmax(axis=1))

                    #print (time.time() - self.start_time) # = time.time() #print ("gesture {}".format(gesture))
                else:
                    y_preds.append([[1,0]])
            gesture_preds = np.asarray(gesture_preds)
            y_preds = np.asarray(y_preds).reshape(-1,2)#anomalytranscript = np.asarray(anomalytranscript)[0:np.asarray(y_preds).shape[0]]
            anomalytranscript = np.asarray(anomalytranscript)[0:np.asarray(y_preds).shape[0]]
            anomalyevaluation_data = np.concatenate((anomalytranscript, y_preds), axis=1)
            anomalyevaluation_data = np.concatenate(( gesture_preds.reshape(-1,1),anomalyevaluation_data), axis=1)

            writecsv_path = "{}/evaluation_k{}fold".format(write_path, model_index)
            if not os.path.exists(writecsv_path):
                os.makedirs(writecsv_path)

            output_df = pd.DataFrame(data=anomalyevaluation_data, columns=["gesture","true_negativeclass", "true_positiveclass", "predicted_negativeclass", "predicted_positiveclass"])
            output_df.to_csv("{}/all{}_{}.csv".format(writecsv_path, subject, model_index))

    def summarizeResults(self):
        """
        gives the auc/confusion matrix of the results
        """
        task = "Suturing"
        write_path = os.path.join(self.data_path, task, "{}{}".format(self.jaccard_path, self.model_name))
        result_path = os.path.join(self.data_path, "Suturing", "{}{}".format(self.jaccard_path, self.model_name), "evaluation_k*fold", "*.csv")
        figure_path = os.path.join(self.data_path, "Suturing", "{}{}".format(self.jaccard_path, self.model_name))

        results_auc = dict()
        results_f1dict = dict()
        results_tpr_fprdict = dict()
        threshold_dict = dict()

        preds_list = list()

        for anomaly_file in sorted(glob.glob(result_path)):
            subject = anomaly_file.split("/")[-1].replace(".csv","").replace("anomaly","")
            #print (subject)

            anomaly_results = pd.read_csv(anomaly_file, index_col=0)

            preds_list = list(); test_list = list()

            preds = np.asarray(anomaly_results[["predicted_positiveclass"]]); test = np.asarray(anomaly_results[["true_positiveclass"]])#.iloc[:,1]
            #print ("test {} preds {}".format(len(test) ,len(preds)))
            for i in range(len(preds)):
                #if "[" in preds[i]:
                preds_list.append(((preds[i])))
                test_list.append(((test[i])))


            false_pos_rate, true_pos_rate, thresholds = roc_curve(test_list, preds_list)
            threshold_dict[subject] = [false_pos_rate, true_pos_rate, thresholds]
            roc_auc = auc(false_pos_rate, true_pos_rate)
            results_auc[subject] = roc_auc
            if subject == "allSuturing_D003_3" or subject == "allSuturing_D005_5" or subject == "allSuturing_E003_3":
                print ("Keeping")
                results_tpr_fprdict[subject] = [(false_pos_rate, true_pos_rate)]
            #else:
            #    continue #plt.plot(false_pos_rate, true_pois_rate, label='AUC = %0.3f '% (roc_auc))
            plt.plot(false_pos_rate, true_pos_rate, label='AUC = %0.3f '% (roc_auc))
        #plt.plot(false_pos_rate, true_pos_rate, label='AUC = %0.3f '% (roc_auc))
            plt.plot([0,1],[0,1], linestyle='--')
            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower right')
            plt.title('Receiver operating characteristic curve (ROC) ' )#{}'.format(subject))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            print ("{}/auck{}_fold/allauc_{}.pdf".format(figure_path, subject.split("_")[-1], subject))
            plt.savefig("{}/auck{}_fold/allauc_{}.pdf".format(figure_path, subject.split("_")[-1], subject))
            plt.close()
            #print (np.asarray(anomaly_results.iloc[:,1:3]).argmax(axis=1))
            #print (f1_score(np.asarray(anomaly_results.iloc[:,1:3]).argmax(axis=1), np.asarray(anomaly_results.iloc[:,3:5]).argmax(axis=1)))
            y_preds = np.asarray(anomaly_results.iloc[:,1:3]).argmax(axis=1)
            y_test = np.asarray(anomaly_results.iloc[:,3:5]).argmax(axis=1)
            results_f1dict[subject] = (f1_score(y_preds, y_test, average='micro'))# (f1_score(np.asarray(test_list).argmax(axis=1), np.asarray(preds_list).argmax(axis=1)))
        print (results_f1dict.keys())


        results_aucdf = pd.DataFrame.from_dict(results_auc, orient='index')
        #results_aucdf.to_csv("{}/{}.csv".format(write_path, "results_auc"))
        results_f1df = pd.DataFrame.from_dict(results_f1dict, orient='index')
        results_f1df.to_csv("{}/{}.csv".format(write_path, "results_f1"))
        results__tpr_fprdf = pd.DataFrame.from_dict(results_tpr_fprdict)
        results__tpr_fprdf.to_csv("{}/all{}.csv".format(write_path, "results__tpr_fpr"))
        pickle.dump(threshold_dict, open( "{}/{}.p".format(figure_path, "allgesturedict"), "wb" ))

        return results_auc
    def findeventoverlap(self):
        """
        Finds the number of occasion where there is no overlap between predicted anomaly and actual anomaly
        """

        task = "Suturing"
        write_path = os.path.join(self.data_path, task, "{}{}".format(self.jaccard_path, self.model_name))
        result_path = os.path.join(self.data_path,task, "{}{}".format(self.jaccard_path, self.model_name), "evaluation_k*fold", "all*.csv")
        results_auc = self.summarizeResults()


        preds_list = list()
        jitter = dict()
        for anomaly_file in sorted(glob.glob(result_path)):
            subject = anomaly_file.split("/")[-1].replace(".csv","").replace("anomaly","")
            print (anomaly_file)
            jitter[subject] = list()
            anomaly_results = pd.read_csv(anomaly_file, index_col=0)

            preds_list = list(); test_list = list()
            #diff =  (np.asarray(anomaly_results[["predicted_positiveclass"]]).reshape(-1,1) -  np.asarray(anomaly_results.iloc[:,5]).reshape(-1,1))

            preds = np.asarray(anomaly_results.iloc[:,3:5]); test = np.asarray(anomaly_results.iloc[:,1:3])
            #print (jaccard_score(preds.argmax(axis=1), test.argmax(axis=1), average='weighted'))
            preds_list.append(jaccard_score(preds.argmax(axis=1), test.argmax(axis=1), average='binary'))
            preds = preds.argmax(axis=1)
            test = test.argmax(axis=1)

            trueerror_begin = np.where(test!=0)[0]; prederror_begin = np.where(preds!=0)[0]
            trueerror_packets = list(); trueerror_packets.append(trueerror_begin[0])
            prederror_packets = list(); prederror_packets.append(prederror_begin[0])
            #print (trueerror_begin)
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
                if abs(min_value-current_packet)<150:
                    jitter[subject].append(current_packet-min_value)
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


        jitter_df = pd.DataFrame.from_dict(data=jitter, orient="index", columns= ["subject", "jitter0", "jitter1", "jitter2", "jitter3", "jitter4", "jitter5", "jitter6", "jitter7", "jitter8",  "average_jitter", "auc"])
        jitter_df.to_csv("{}/jitterCRG_new{}.csv".format(write_path, self.model_name))




path_ = os.path.abspath(os.path.dirname(sys.argv[0]))
rpl = runPipeline(path_)
#rpl.runPipeline()
rpl.summarizeResults()
rpl.findeventoverlap()
