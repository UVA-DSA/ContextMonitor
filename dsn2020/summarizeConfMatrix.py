import numpy as np
import pandas as pd
import glob, os, sys, math
from keras.models import load_model

class summarizeConf:
    def __init__(self, path):
        self.data_path = path

    def iterategesturePaths(self, clf_mode, kinvars, model_num):

        result_dict = dict()
        file_keys = list()

        count = 0
        iterate_path = os.path.join(self.data_path, "Suturing", "{}".format(clf_mode), "{}".format(kinvars), "{}".format(model_num), "*", "csvs*")
        tpr_sum = 0; tnr_sum = 0; fpr_sum = 0; fnr_sum = 0
        for path_name in sorted(glob.glob(iterate_path), reverse=True):
            file = path_name.split("/")[-3]
            csv_path = os.path.join(path_name, "confusion_matrix", "*.csv")
            result_dict[file] = []
            tpr_dict = dict(); tnr_dict = dict(); tpr_avg = dict(); tnr_avg = dict();  f_beta = list(); beta = 0.01;
            tpr_count = 0
            tnr_count = 0
            tp = 0; fp = 0; fn = 0; tn = 0
            for name in sorted(glob.glob(csv_path)):
                print (name)
                file_name = name.split("/")[-1]
                gesture_block = file_name.split("_")[-2]
                gesture = gesture_block.replace("not","").split("g")[-1]
                gesture = file_name.split("/")[-1].replace(".csv","")
                conf_matrix = pd.read_csv(name, index_col=0)
                print (conf_matrix)
                if len(conf_matrix)>1:
                    tnr_value = conf_matrix.iloc[1][1]/(conf_matrix.iloc[0][1]+conf_matrix.iloc[1][1])
                    tpr_value = conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])
                    ppv_value = conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0]+conf_matrix.iloc[0][1])
                    npv_value = conf_matrix.iloc[1][1]/(conf_matrix.iloc[1][0]+conf_matrix.iloc[1][1])
                    tpr_sum += conf_matrix.iloc[0][0]
                    fpr_sum += conf_matrix.iloc[0][1]
                    fnr_sum += conf_matrix.iloc[1][0]
                    tnr_sum += conf_matrix.iloc[1][1]
                    if (conf_matrix.iloc[1][0]+conf_matrix.iloc[1][1])>0:
                        if not math.isnan(npv_value):
                            npv.append(npv_value)
                        if not math.isnan(ppv_value):
                            ppv.append(ppv_value)
                        if not math.isnan(tnr_value):
                            tnr_sum.append(tnr_value)
                        if not math.isnan(tpr_value):
                            tpr_sum.append(tpr_value)


                        precision = conf_matrix.iloc[1][1]/(conf_matrix.iloc[1][1] + conf_matrix.iloc[0][1])
                        recall = conf_matrix.iloc[1][1]/(conf_matrix.iloc[1][1] + conf_matrix.iloc[1][0])

                        numerator = (1+beta**2)*conf_matrix.iloc[1][1]
                        denominator_1 = numerator
                        denominator_2 = conf_matrix.iloc[1][0]*(beta**2)
                        denominator_3 = conf_matrix.iloc[0][1]
                        fb_value = (precision*recall*(1+beta*beta))/((recall + precision*beta*beta))
                        f_beta.append(fb_value)
                        #if tpr_value>0:

                else:
                    tpr_value = conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0])
                    ppv_value = conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0])
                    tpr_sum += conf_matrix.iloc[0][0]
                    tpr_sum.append(tpr_value)
                    ppv.append(ppv_value)
                    tp += conf_matrix.iloc[0][0]


            print ("tnr sum {}".format(tnr_sum))
            print ("tpr sum {}".format(tpr_sum))
            #print ("tnr rate {}".format(sum(tnr_sum)/len(tnr_sum)))

            #print ("fbeta value {}".format(sum(f_beta)/len(f_beta)))
            if len(tnr_sum) >0 and len(tpr_sum) >0:
                result_dict[file].append([sum(tnr_sum)/len(tnr_sum), sum(tpr_sum)/len(tpr_sum), sum(npv)/len(npv), sum(ppv)/len(ppv)])
            count += 1
            #if count >36:
            #    return result_dict
            #    tpr_value = conf_matrix.iloc[0][0]/(conf_matrix.iloc[0][0]+conf_matrix.iloc[1][0])
        print ("micro tp {} tn {} fp {}  fn{}".format(tpr_sum, tnr_sum, fpr_sum, fnr_sum))
        #print ("micro tp {} tn {} pvp {}  npv{}".format(tp/(tp+fn), tn/(tn+fp), tp/(tp+fp), tn/(tn+fn)))
        return result_dict

    def getmicroscores(self, clf_mode, kinvars, model_num):

        result_dict = dict()
        file_keys = list()

        count = 0
        iterate_path = os.path.join(self.data_path, "Suturing", "{}".format(clf_mode), "{}".format(kinvars), "{}".format(model_num), "*", "csvs*")

        tpr_list = list(); tnr_list = list(); ppv_list = list(); npv_list = list()
        for path_name in sorted(glob.glob(iterate_path), reverse=True):
            print (path_name)
            file = path_name.split("/")[-3]
            csv_path = os.path.join(path_name, "confusion_matrix", "*.csv")
            result_dict[file] = []
            tpr_sum = 0; tnr_sum = 0; fpr_sum = 0; fnr_sum = 0
            for name in sorted(glob.glob(csv_path)):
                print (name)
                file_name = name.split("/")[-1]
                gesture_block = file_name.split("_")[-2]
                gesture = gesture_block.replace("not","").split("g")[-1]
                gesture = file_name.split("/")[-1].replace(".csv","")
                conf_matrix = pd.read_csv(name, index_col=0)
                print (conf_matrix)
                if len(conf_matrix)>1:
                    tpr_sum += conf_matrix.iloc[0][0]
                    fpr_sum += conf_matrix.iloc[0][1]
                    fnr_sum += conf_matrix.iloc[1][0]
                    tnr_sum += conf_matrix.iloc[1][1]
                else:
                    tpr_sum += conf_matrix.iloc[0][0]
            tpr_list.append(tpr_sum/(tpr_sum+fnr_sum)); tnr_list.append(tnr_sum/(tnr_sum+fpr_sum)); ppv_list.append(tpr_sum/(tpr_sum+fpr_sum)); npv_list.append(tnr_sum/(tnr_sum+fnr_sum))


        print ("micro tp {} tn {} fp {}  fn{}".format(tpr_sum, tnr_sum, fpr_sum, fnr_sum))

        #print ("micro tp {} tn {} ppv {}  npv{}".format()
        print (tpr_list)
        result_dict[file].append([sum(tnr_list)/len(tnr_list), sum(tpr_list)/len(tpr_list), sum(npv_list)/len(npv_list), sum(ppv_list)/len(ppv_list)])
        return result_dict


    def getModelParameters(self,clf_mode, kinvars, model_num):
        model_dict = dict()
        lr_dict = dict()
        iterate_path = os.path.join(self.data_path, "Suturing", "{}".format(clf_mode), "{}".format(kinvars), "{}".format(model_num), "*", "checkpoints*")
        count = 0
        print (iterate_path)
        for name in sorted(glob.glob(iterate_path), reverse=True):
            file = name.split("/")[-3]
            model_dict[file] = list()
            checkpoint = glob.glob(os.path.join(name, "clf_checkpoint*.h5"))[0]
            #print (checkpoint)
            encoder_dims = 3
            print (checkpoint)
            lstm_model = (load_model(checkpoint))
            print (float(name.split("/")[-1].split("_")[1]))
            lr_dict[file] = float(name.split("/")[-1].split("_")[1])

            for layer in lstm_model.layers:
                #print (layer)
                if ("Conv1D" in str(layer)) and encoder_dims>0:
                    model_dict[file].append(layer.output_shape[-2])
                elif ("Dense" in str(layer) or "LSTM" in str(layer)) and encoder_dims>0:
                    model_dict[file].append(layer.output_shape[-1])

            count += 1
            #if count >36:
            #    return model_dict, lr_dict


        return model_dict, lr_dict

    def combineResults(self):
        clf_mode = "loso_experiments_clfv1"
        kinvars = "cartesian_rotation_grasperAngle"
        model_num = "914044"
        model_dict, lr_dict = self.getModelParameters(clf_mode, kinvars, model_num)
        result_dict = self.getmicroscores(clf_mode, kinvars, model_num)
        write_path = os.path.join(self.data_path, "Suturing", "{}".format(clf_mode), "results", kinvars, "anomaly_detection_All{}{}.csv".format(model_num,kinvars))
        combined_dict = dict()
        print (result_dict)
        print (model_dict)
        for key in model_dict.keys():
            #print ("model params  {} {} {}".format(model_dict[key][0], model_dict[key][1], model_dict[key][2]))
            #print ("result params {} {} ".format(result_dict[key][0][0], result_dict[key][0][1]))
            if len(result_dict[key])>0:
                combined_dict[key] =([lr_dict[key], model_dict[key][0],  model_dict[key][1], model_dict[key][2],model_dict[key][3], result_dict[key][0][0], result_dict[key][0][1], result_dict[key][0][2], result_dict[key][0][3]])
        combined_df = pd.DataFrame.from_dict(data=combined_dict, orient='index')
        micro_tpr = list(); macro_tpr = list(); micro_tnr = list(); macro_tnr = list()
        combined_df.to_csv(write_path, header=["lr", "encoder1", "encoder2", "encoder3", "encoder4","TNR", "TPR", "NPV", "PPV"], sep= ",")
        print (combined_dict)

path = os.path.abspath(os.path.dirname(sys.argv[0]))
scg = summarizeConf(path)
#scg.iteratePaths()
#scg.iterategesturePaths()
#scg.getModelParameters()
scg.combineResults()
