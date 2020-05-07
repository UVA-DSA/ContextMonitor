import os, sys, glob, math, _pickle, time, gc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import rcParams
import seaborn as sns

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.layers import  LSTM, RepeatVector, TimeDistributed, Dropout, Masking, BatchNormalization, Input, Flatten, Dense, GRU, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger, LearningRateScheduler
from keras.preprocessing import sequence
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score, mean_squared_error, fbeta_score
from sklearn.utils import class_weight
from numpy.random import seed
from scipy.stats import norm
from plotauc import loadandplot
seed(1)

from tensorflow import set_random_seed
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
set_random_seed(2)

SEED = 123 #used to help randomly select the data points
#DATA_SPLIT_PCT = 0.1
#rcParams['figure.figsize'] = 8, 6
LABELS = ["Optimal","Suboptimal"]
class lstmVAE:
    def __init__(self, path, task):
        self.dataPath = path
        self.task = task

    def getModel(self, timesteps, n_features, lr, encoder1 = 256, encoder2 = 128, encoder3 = 64, encoder4 = 32):
        """
        Sequence-to-sequence lstm-ae
        """
        #encoder1 = 256
        #encoder2 = 128
        #encoder3 = 64
        #encoder4 = 32
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(Masking(mask_value = 0.0, input_shape=(timesteps, n_features)))
        lstm_autoencoder.add((LSTM(encoder1, activation='relu', return_sequences=True)))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder1)))
        lstm_autoencoder.add(LSTM(encoder2, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder2)))
        lstm_autoencoder.add(LSTM(encoder3, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder3)))
        lstm_autoencoder.add((LSTM(encoder4, activation='relu', return_sequences=False)))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add((LSTM(encoder4, activation='relu', return_sequences=True)))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder4)))
        lstm_autoencoder.add(LSTM(encoder3, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder3)))
        lstm_autoencoder.add(LSTM(encoder2, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder2)))
        lstm_autoencoder.add(LSTM(encoder1, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder1)))
        lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

        #lstm_autoencoder.summary()
        adam = optimizers.Adam(lr)
        lstm_autoencoder.compile(loss='mse', optimizer=adam)

        return lstm_autoencoder

    def getModelv2(self, timesteps, n_features, lr):
        """
        Sequence-to-sequence lstm-ae
        """
        encoder1 = 128
        encoder2 = 64
        encoder3 = 128
        encoder4 = 64
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add((GRU(encoder1, activation='relu', input_shape=(timesteps, n_features), return_sequences=True)))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder1)))
        lstm_autoencoder.add(GRU(encoder2, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder2)))
        #lstm_autoencoder.add(LSTM(encoder3, activation='relu', return_sequences=True))
        #lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder3)))
        lstm_autoencoder.add((GRU(encoder4, activation='relu', return_sequences=False)))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add((GRU(encoder4, activation='relu', return_sequences=True)))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder4)))
        #lstm_autoencoder.add(LSTM(encoder3, activation='relu', return_sequences=True))
        #lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder3)))
        lstm_autoencoder.add(GRU(encoder2, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder2)))
        lstm_autoencoder.add(GRU(encoder1, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(rate=0.5, noise_shape=(None, None, encoder1)))
        lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

        #lstm_autoencoder.summary()
        adam = optimizers.Adam(lr)
        lstm_autoencoder.compile(loss='mse', optimizer=adam)

        return lstm_autoencoder

    def getClassifier(self, timesteps, n_features, lr, encoder1 = 256, encoder2 = 128, encoder3 = 32, dense1 = 64):
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(Masking(mask_value = 0.0, input_shape=(timesteps, n_features)))
        lstm_autoencoder.add(LSTM(encoder1, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(LSTM(encoder2, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(LSTM(encoder3, activation='relu', return_sequences=False))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add(Flatten())
        lstm_autoencoder.add(Dropout(rate = 0.55))
        lstm_autoencoder.add(Dense(dense1, activation='relu'))
        lstm_autoencoder.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
        lstm_autoencoder.add(Dense(2, activation='softmax'))

        #lstm_autoencoder.summary()
        adam = optimizers.Adam(lr)
        lstm_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        version = "v1"
        return lstm_autoencoder, version

    def getClassifier_v2(self, timesteps, n_features, lr, encoder1 = 256, encoder2 = 128, encoder3 = 32, dense1 = 64):
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(Conv1D(filters= encoder1, input_shape=(timesteps, n_features), kernel_size=3, data_format='channels_first'))
        lstm_autoencoder.add(MaxPooling1D(pool_size=2))
        lstm_autoencoder.add(Conv1D(filters= encoder2, kernel_size=3, data_format='channels_first'))
        lstm_autoencoder.add(MaxPooling1D(pool_size=2))
        lstm_autoencoder.add(LSTM(encoder3, activation='relu', return_sequences=False))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(RepeatVector(timesteps))
        lstm_autoencoder.add(Flatten())
        lstm_autoencoder.add(Dense(dense1, activation='relu'))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(Dense(2, activation='softmax'))

        lstm_autoencoder.summary()

        #lstm_autoencoder.summary()
        adam = optimizers.Adam(lr)
        lstm_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        version = "v2"
        return lstm_autoencoder, version

    def getClassifier_v3(self, timesteps, n_features, lr, encoder1 = 256, encoder2 = 128, encoder3 = 32, dense1 = 64):
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(Masking(mask_value = 0.0, input_shape=(timesteps, n_features)))
        lstm_autoencoder.add(LSTM(encoder1, activation='relu', return_sequences=True))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(LSTM(encoder2, activation='relu', return_sequences=False))
        lstm_autoencoder.add(Dropout(0.25))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add(Flatten())
        lstm_autoencoder.add(Dropout(rate = 0.5))
        lstm_autoencoder.add(Dense(64, activation='relu'))
        lstm_autoencoder.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
        lstm_autoencoder.add(Dense(2, activation='softmax'))

        #lstm_autoencoder.summary()
        adam = optimizers.Adam(lr)
        lstm_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        version = "v3"
        return lstm_autoencoder, version

    def distinguishSequence(self, all_kinematics, all_labels):
        #print ("set of laels {}".format(set(all_labels.flatten())))
        #print ("set of locations {}".format(np.where(all_labels>0)[0]))
        anomalous_trajectory = np.asarray(all_kinematics)[np.where(all_labels>0)[0]]
        nonanomalous_trajectory = np.asarray(all_kinematics)[np.where(all_labels==0)[0]]
        anomalous_label = np.asarray(all_labels)[np.where(all_labels>0)[0]]
        nonanomalous_label = np.asarray(all_labels)[np.where(all_labels==0)[0]]

        return nonanomalous_trajectory, anomalous_trajectory, nonanomalous_label, anomalous_label

    def flatten(self, X):
        '''
        Flatten a 3D array.

        Input
        X            A 3D array for lstm, where the array is sample x timesteps x features.

        Output
        flattened_X  A 2D array, sample x features.
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return(flattened_X)

    def splitdata(self, x_train, x_label):
        """
        splits the data to train set and validation set
        """
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, x_label, stratify =x_label[:,0], test_size=0.2, random_state=42, shuffle=True)
        return x_train, x_valid, y_train, y_valid


    def trainClassifier(self, currentTimestamp, window, stride, vae_data, trial="-1", itr_num=0, gesture="g", train=0, mode="None", kinvars="All", encoder1 = 256, encoder2 = 128, encoder3 = 64, encoder4 = 32, lr = 0.001):

        X_train_y0_scaled, X_valid_y0_scaled, X_test_scaled, y_test = vae_data
        y_normal = np.zeros((len(X_train_y0_scaled)+len(X_valid_y0_scaled), 1))
        y = np.concatenate((y_normal, y_test), axis=0)
        x = np.concatenate((X_train_y0_scaled, X_valid_y0_scaled), axis=0)
        x = np.concatenate((x, X_test_scaled), axis=0)
        kf = StratifiedKFold(n_splits=5)
        k = 0
        for train_index, test_index in kf.split(x, y):

            x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index] #self.splitdata(x, y)
            #x_train, x_test, y_train, y_test = self.splitdata(x, y)
            x_train, x_valid, y_train, y_valid = self.splitdata(x_train, y_train)
            y_train, y_valid, y_test = self.vectorizeLabels(y_train, y_valid, y_test)


            if len(x_train)>0 and len(x_valid)>0 and len(x_test)>0:
                epochs = 200
                self.lr = lr

                sequence_length, timesteps, n_features = x_train.shape
                #X_train_y0_scaled = X_train_y0_scaled.reshape(batch, -1, timesteps,n_features)
            if train==1:
                lstm_autoencoder, version = self.getClassifier(timesteps, n_features, lr, encoder1 = encoder1, encoder2 = encoder2, encoder3 = encoder3, dense1 = encoder4)
                figure_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "figures_{}_{}_{}_k{}".format(lr, window, stride, k))
                trainlog_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "csvs_{}_{}_{}_k{}".format(lr, window, stride, k), "trainlog")
                confmatrix_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "csvs_{}_{}_{}_k{}".format(lr, window, stride, k), "confusion_matrix")
                checkpoint_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "checkpoints_{}_{}_{}_k{}".format(lr, window, stride, k))
                self.makePaths(figure_path, trainlog_path, confmatrix_path, checkpoint_path)

                cp1 = EarlyStopping(monitor='val_loss', min_delta=0.0000, patience=5, verbose=0)
                cp2 = ModelCheckpoint(filepath="{}/clf_checkpoint_{}gesture{}{}.h5".format(checkpoint_path, itr_num, gesture,  mode), save_best_only=True, verbose=0)
                cp3 = CSVLogger(filename="{}/lstm_classifier_{}gesture{}{}.csv".format(trainlog_path, itr_num, gesture,  mode), append=False, separator=';')
                cp4 = LearningRateScheduler(self.step_decayclf)
                plot_model(lstm_autoencoder, to_file="{}/clf_model_{}gesture{}{}.pdf".format(figure_path,itr_num, gesture,  mode), show_shapes=True, show_layer_names=True)

                lstm_autoencoder.reset_states()
                print (np.unique(y_train))
                class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train.flatten()), y_train.flatten())
                lstm_autoencoder_history = lstm_autoencoder.fit(x_train, y_train, class_weight = class_weights, epochs=epochs, batch_size=32,
                                                                        verbose=0, callbacks=[cp1,cp2,cp3, cp4], validation_data=(x_valid, y_valid)).history
                lstm_autoencoder = load_model("{}/clf_checkpoint_{}gesture{}{}.h5".format(checkpoint_path, itr_num, gesture,  mode))

                X_test_temp = x_test.copy()
                X_test_temp = np.asarray(X_test_temp)#.reshape(-1, X_valid_scaled[2], X_valid_scaled[3])
                y_preds = lstm_autoencoder.predict(X_test_temp)

                f1 = f1_score(y_test.argmax(axis=1), y_preds.argmax(axis=1), average = 'weighted')
                conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_preds.argmax(axis=1))
                conf_matrixdf = pd.DataFrame(data=conf_matrix)
                conf_matrixdf.to_csv("{}/adaptiveconf_{}gesture{}{}.csv".format(confmatrix_path, itr_num, gesture,  mode))
                y_preds = lstm_autoencoder.predict(X_test_temp)
                false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test.argmax(axis=1), y_preds.argmax(axis=1))
                roc_auc = auc(false_pos_rate, true_pos_rate)
                plt.plot(false_pos_rate, true_pos_rate, label='AUC = %0.3f '% (roc_auc))
                plt.plot([0,1],[0,1], linestyle='--')
                plt.xlim([-0.01, 1])
                plt.ylim([0, 1.01])
                plt.legend(loc='lower right')
                plt.title('Receiver operating characteristic curve (ROC)')
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig("{}/auc_{}gesture{}{}.pdf".format(figure_path, itr_num, gesture,  mode))
                plt.close()


                print ("************** f1_score {} ***********".format(f1))

            else:
                lap = loadandplot(self.dataPath, self.task)
                lap.setTestData(x_test, y_test, trial, "1", itr_num, gesture, train, mode, kinvars, lr, window, stride, k)


            k+=1
            tf.reset_default_graph()
            gc.collect()
            K.clear_session()

    def losotrainClassifier(self, currentTimestamp, window, stride, vae_data, trial="-1", itr_num=0, gesture="g", train=0, mode="None", kinvars="All", encoder1 = 256, encoder2 = 128, encoder3 = 64, encoder4 = 32, lr = 0.001):

        x_train, x_test, y_train, y_test = vae_data


        k = 0

        x_train, x_valid, y_train, y_valid = self.splitdata(x_train, y_train)
        y_train, y_valid, y_test = self.vectorizeLabels(y_train, y_valid, y_test)


        if len(x_train)>0 and len(x_valid)>0 and len(x_test)>0:
            epochs = 200
            self.lr = lr

            sequence_length, timesteps, n_features = x_train.shape
            #X_train_y0_scaled = X_train_y0_scaled.reshape(batch, -1, timesteps,n_features)
        if train==1:
            lstm_autoencoder, version = self.getClassifier(timesteps, n_features, lr, encoder1 = encoder1, encoder2 = encoder2, encoder3 = encoder3, dense1 = encoder4)
            figure_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "figures_{}_{}_{}_k{}".format(lr, window, stride, k))
            trainlog_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "csvs_{}_{}_{}_k{}".format(lr, window, stride, k), "trainlog")
            confmatrix_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "csvs_{}_{}_{}_k{}".format(lr, window, stride, k), "confusion_matrix")
            checkpoint_path = os.path.join(self.dataPath, self.task, "loso_experiments_clf{}".format(version), kinvars, currentTimestamp, trial, "checkpoints_{}_{}_{}_k{}".format(lr, window, stride, k))
            self.makePaths(figure_path, trainlog_path, confmatrix_path, checkpoint_path)

            cp1 = EarlyStopping(monitor='val_loss', min_delta=0.0000, patience=5, verbose=0)
            cp2 = ModelCheckpoint(filepath="{}/clf_checkpoint_{}gesture{}{}.h5".format(checkpoint_path, itr_num, gesture,  mode), save_best_only=True, verbose=0)
            cp3 = CSVLogger(filename="{}/lstm_classifier_{}gesture{}{}.csv".format(trainlog_path, itr_num, gesture,  mode), append=False, separator=';')
            cp4 = LearningRateScheduler(self.step_decayclf)
            plot_model(lstm_autoencoder, to_file="{}/clf_model_{}gesture{}{}.pdf".format(figure_path,itr_num, gesture,  mode), show_shapes=True, show_layer_names=True)

            lstm_autoencoder.reset_states()
            print (np.unique(y_train))
            class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train.flatten()), y_train.flatten())
            lstm_autoencoder_history = lstm_autoencoder.fit(x_train, y_train, class_weight = class_weights, epochs=epochs, batch_size=32,
                                                                    verbose=0, callbacks=[cp1,cp2,cp3, cp4], validation_data=(x_valid, y_valid)).history
            lstm_autoencoder = load_model("{}/clf_checkpoint_{}gesture{}{}.h5".format(checkpoint_path, itr_num, gesture,  mode))

            X_test_temp = x_test.copy()
            X_test_temp = np.asarray(X_test_temp)#.reshape(-1, X_valid_scaled[2], X_valid_scaled[3])
            y_preds = lstm_autoencoder.predict(X_test_temp)

            f1 = f1_score(y_test.argmax(axis=1), y_preds.argmax(axis=1), average = 'weighted')
            conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_preds.argmax(axis=1))
            conf_matrixdf = pd.DataFrame(data=conf_matrix)
            conf_matrixdf.to_csv("{}/adaptiveconf_{}gesture{}{}.csv".format(confmatrix_path, itr_num, gesture,  mode))
            y_preds = lstm_autoencoder.predict(X_test_temp)
            false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test.argmax(axis=1), y_preds.argmax(axis=1))
            roc_auc = auc(false_pos_rate, true_pos_rate)
            plt.plot(false_pos_rate, true_pos_rate, label='AUC = %0.3f '% (roc_auc))
            plt.plot([0,1],[0,1], linestyle='--')
            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower right')
            plt.title('Receiver operating characteristic curve (ROC)')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig("{}/auc_{}gesture{}{}.pdf".format(figure_path, itr_num, gesture,  mode))
            plt.close()


            print ("************** f1_score {} ***********".format(f1))

        else:
            lap = loadandplot(self.dataPath, self.task)
            lap.setTestData(x_test, y_test, trial, "1", itr_num, gesture, train, mode, kinvars, lr, window, stride, k)


        k+=1
        tf.reset_default_graph()
        gc.collect()
        K.clear_session()

    def trainModel(self, currentTimestamp, vae_data, trial="-1", itr_num=0, gesture="g", train=0, mode="None", kinvars="All", encoder1=512, encoder2=256, encoder3=32, encoder4=128, lr = 0.001):

        figure_path = os.path.join(self.dataPath, self.task, "experiments", kinvars, currentTimestamp, trial, "figures_{}".format(lr))
        trainlog_path = os.path.join(self.dataPath, self.task, "experiments", kinvars, currentTimestamp, trial, "csvs_{}".format(lr), "trainlog")
        confmatrix_path = os.path.join(self.dataPath, self.task, "experiments", kinvars, currentTimestamp, trial, "csvs_{}".format(lr), "confusion_matrix")
        checkpoint_path = os.path.join(self.dataPath, self.task, "experiments", kinvars, currentTimestamp, trial, "checkpoints_{}".format(lr))
        self.makePaths(figure_path, trainlog_path, confmatrix_path, checkpoint_path)

        X_train_y0_scaled, X_valid_y0_scaled, X_test_scaled, y_test = vae_data

        epochs = 20#40
        self.lr = lr #0.01#1

        sequence_length, timesteps, n_features = X_train_y0_scaled.shape

        if train==1:

            lstm_autoencoder = self.getModel(timesteps, n_features, lr, encoder1=encoder1, encoder2=encoder2, encoder3=encoder3, encoder4=encoder4)

            cp1 = EarlyStopping(monitor='val_loss', min_delta=0.0000, patience=5, verbose=0)
            cp2 = ModelCheckpoint(filepath="{}/lstmae_checkpoint_{}gesture{}{}.h5".format(checkpoint_path, itr_num, gesture,  mode), save_best_only=True, verbose=0)
            cp3 = CSVLogger(filename="{}/lstm_mode_{}gesture{}{}.csv".format(trainlog_path, itr_num, gesture,  mode), append=False, separator=';')
            cp4 = LearningRateScheduler(self.step_decay)
            plot_model(lstm_autoencoder, to_file="{}/lstmae_model_{}gesture{}{}.pdf".format(figure_path, itr_num, gesture,  mode), show_shapes=True, show_layer_names=True)

            lstm_autoencoder.reset_states()

            lstm_autoencoder_history = lstm_autoencoder.fit(X_train_y0_scaled, X_train_y0_scaled, epochs=epochs, batch_size=32,
                                                                    verbose=0, callbacks=[cp1,cp2,cp3,cp4], validation_data=(X_valid_y0_scaled, X_valid_y0_scaled)).history

        #lstm_autoencoder = load_model("{}/lstmae_checkpoint_{}gesture{}{}.h5".format(checkpoint_path, itr_num, gesture,  mode))

        X_test_temp = X_test_scaled.copy()
        X_test_temp = np.asarray(X_test_temp)#.reshape(-1, X_valid_scaled[2], X_valid_scaled[3])
        test_x_predictions = lstm_autoencoder.predict(X_test_temp)
        mse = np.mean(np.power((self.flatten(X_test_temp) - self.flatten(test_x_predictions)), 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error': list(mse),
                            'True_class': list(y_test[:,0])})

        threshold_fixed = lstm_autoencoder_history['val_loss'][-1]
        thresholds = np.linspace(0,4, 100)
        f_scores = list()
        for t in thresholds:
            threshold_fixed=t
            pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
            f_scores.append(fbeta_score(error_df.True_class, pred_y, beta=0.01, pos_label=0.0))
        #print ("max f beta score with beta=0.01 {}".format(max(f_scores)))
        threshold_fixed = thresholds[f_scores.index(max(f_scores))]
        threshold_fixed = min(lstm_autoencoder_history['val_loss'])
        pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()
        print ("set of laels {}".format(set(y_test.flatten())))

        for name, group in groups:
            ax.plot(np.asarray(group.index).reshape(-1,1), np.asarray(group.Reconstruction_error).reshape(-1,1), marker='o', ms=3.5, linestyle='', label= "Safety-critical" if name == 1 else "Normal")
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig("{}/separation_{}gesture{}{}.pdf".format(figure_path, itr_num, gesture,  mode))
        plt.close()


        conf_matrix = confusion_matrix(error_df.True_class, pred_y)
        conf_matrixdf = pd.DataFrame(data=conf_matrix)
        conf_matrixdf.to_csv("{}/conf_{}gesture{}{}.csv".format(confmatrix_path, itr_num, gesture,  mode))
        plt.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig("{}/confusion_matrix_{}gesture{}{}.pdf".format(figure_path, itr_num, gesture,  mode))
        plt.close()

        false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
        roc_auc = auc(false_pos_rate, true_pos_rate,)
        plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='ROC = %0.3f'% roc_auc)
        plt.plot([0,1],[0,1], linewidth=5)
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic curve (ROC)')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("{}/auc_{}gesture{}{}.pdf".format(figure_path, itr_num, gesture,  mode))
        plt.close()

        K.clear_session()
        tf.reset_default_graph()
        gc.collect()

        return lstm_autoencoder


    def dumpDemonstrations(self, data):
        """
        dumps
        """
        dump_path = os.path.join(self.dataPath, self.task, "pickles")

        kinematics_pickle = os.path.join(dump_path, "kinematics.p")
        if not  os.path.exists(dump_path):
            os.makedirs(dump_path)
        with open(kinematics_pickle, 'wb') as fp:
            _pickle.dump(data, fp)

    def sepratebysegments(self, x_trajectory, y_label):
        """
        segemnts all data
        """
        segments = x_trajectory[:,-15:]
        segments_dict = dict()
        labels_dict = dict()
        print ("total shape {}".format(segments.shape))
        sum_length = 0
        for i in range(segments.shape[1]):
            print ("present segment {}".format(i))
            #print (np.where(segments[:,i]==1))
            segments_dict[i] = x_trajectory[np.where(segments[:,i]==1)]
            labels_dict[i] = y_label[np.where(segments[:,i]==1)]
            sum_length += len(segments_dict[i])
            print (segments_dict[i].shape)
            print (labels_dict[i].shape)

        assert sum_length == segments.shape[0]
        return

    def getInput(self, currentTimestamp, data, trial, mode="not_concatenate"):
        """
        gets input from another class, segments according to the error and then segments according to the gesture
        """
        #print ("entered")
        flag = mode
        for i in range(data.shape[0]):
            self.dumpDemonstrations(data[i])
            x_train, y_train, x_test, y_test = data[i]
            kinematics = x_train[:,:,0:-15]
            labels = x_train[:,:,-15:]
            if mode =="not_concatenate":
                print ("separating feature vectors from gesture vectors")
                x_train, x_test = x_train[:,:,0:-15], x_test[:,:,0:-15]
            else:
                print ("runnning pipeline using concatenated setup")

            print ("xtrain {} y train {}".format(x_train.shape, y_train.shape))
            nonanomalous_trajectory, anomalous_trajectory, nonanomalous_label, anomalous_label = self.distinguishSequence(x_train, y_train)
            X_train_y0_scaled, X_valid_y0_scaled, _, _ =self.splitdata(nonanomalous_trajectory, nonanomalous_label)
            print ("X_train shape {}".format(X_train_y0_scaled.shape))
            x_test = np.concatenate((x_test, anomalous_trajectory), axis=0)
            y_test = np.concatenate((y_test, anomalous_label), axis=0)
            vae_data = [X_train_y0_scaled, X_valid_y0_scaled, x_test, y_test]
            self.trainModel(currentTimestamp, vae_data, trial, itr_num = i, train=1, mode=flag, kinvars = kinvars_)


            x_train, x_valid, y_train, y_valid = self.splitdata(x_train, y_train)
            y_train, y_valid, y_test = self.vectorizeLabels(y_train, y_valid, y_test)
            clf_data = [x_train, x_valid, y_train, y_valid, x_test, y_test]
            #self.trainClassifier(clf_data, itr_num = i, train=1)

    def getCategorizedData(self, currentTimestamp, data, trial, mode="not_concatenate", kinvars_="All"):
        """
        gets input from another class, segments according to the error and then segments according to the gesture
        """
        print ("entered")

        for i in range(int(data.shape[0]/2)):
            x_train, y_train, x_test, y_test = data[i]
            print ("xtrain shape {}".format(x_train.shape))
            train_kinematics = x_train[:,:,0:-15]
            train_labels = x_train[:,:,-15:]
            traingesture_dict, trainlabel_dict = self.cataegorizeData(train_kinematics, train_labels, y_train)

            test_kinematics = x_test[:,:,0:-15]
            test_labels = x_test[:,:,-15:]
            testgesture_dict, testlabel_dict = self.cataegorizeData(test_kinematics, test_labels, y_test)
            for key in traingesture_dict.keys():
                if key in testgesture_dict.keys():
                    gesture_xtrain, gesture_ytrain = traingesture_dict[key], trainlabel_dict[key]
                    nonanomalous_trajectory, anomalous_trajectory, nonanomalous_label, anomalous_label = self.distinguishSequence(gesture_xtrain, gesture_ytrain)
                    print ("nonanomalous trajectory shape {}".format(nonanomalous_trajectory.shape))
                    gestureX_train_y0_scaled, gestureX_valid_y0_scaled, _, _ =self.splitdata(nonanomalous_trajectory, nonanomalous_label)

                    testgesture_dict[key] = np.concatenate((testgesture_dict[key], anomalous_trajectory), axis=0)
                    testlabel_dict[key] = np.concatenate((testlabel_dict[key], anomalous_label), axis=0)
                    vae_data = [gestureX_train_y0_scaled, gestureX_valid_y0_scaled, testgesture_dict[key], testlabel_dict[key]]

                    self.trainModel(currentTimestamp, vae_data, trial, itr_num = i, gesture=key, train=1, mode=mode, kinvars = kinvars_)

                    x_train, x_valid, y_train, y_valid = self.splitdata(x_train, y_train)
                    y_train, y_valid, y_test = self.vectorizeLabels(y_train, y_valid, y_test)
                    clf_data = [x_train, x_valid, y_train, y_valid, x_test, y_test]
                #self.trainClassifier(clf_data, itr_num = i, train=1)


    def vectorizeLabels(self, y_train, y_valid, y_test):
        """
        vectorizes the labels
        """
        y_trainvectorized = np.zeros((y_train.shape[0],2))
        y_validvectorized = np.zeros((y_valid.shape[0],2))
        y_testvectorized = np.zeros((y_test.shape[0],2))

        for i in range(y_trainvectorized.shape[0]):
            y_trainvectorized[i][int(y_train[i][0])]=1

        for i in range(y_validvectorized.shape[0]):
            y_validvectorized[i][int(y_valid[i][0])]=1

        for i in range(y_testvectorized.shape[0]):
            y_testvectorized[i][int(y_test[i][0])]=1
        return y_trainvectorized, y_validvectorized, y_testvectorized

    def cataegorizeData(self, kinematics, gestures, labels):
        """
        data received will be categorized into gestures before passing to the LSTM-VAE
        """
        #print ("kinematics {} gestures {}".format(kinematics.shape, gestures.shape))
        gestures_dict = dict()
        labels_dict = dict()
        sum_length = 0
        for gesture in range(gestures.shape[2]):
            if len(np.where(gestures[:,:,gesture]==1)[0]>0):
                gestures_dict["%s"%gesture] = kinematics[np.where(gestures[:,-1,gesture]==1)[0]]
                labels_dict["%s"%gesture] = labels[np.where(gestures[:,-1,gesture]==1)[0]]
                print ("gesture {} kinematics shape {}".format(gesture, gestures_dict["%s"%gesture].shape))
                sum_length += gestures_dict["%s"%gesture].shape[0]
                sum_length += labels_dict["%s"%gesture].shape[0]

        #print ("gesture shape {} total shape {}".format(gestures.shape, sum_length))
        assert sum_length == gestures.shape[0]*2

        return gestures_dict, labels_dict

    def step_decay(self, epoch):
        """
        this function creates step decay
        """

        drop = 0.5; epochs_drop = 1.0; #1
        initial_rate = self.lr
        lrate = initial_rate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        #print (lrate)
        return lrate

    def step_decayclf(self, epoch):
        """
        this function creates step decay
        """
        drop = 0.5; epochs_drop = 5.0; initial_rate = self.lr
        lrate = initial_rate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        print ("learning rate {}".format(lrate))
        return lrate

    def makePaths(self, figure_path, trainlog_path, confmatrix_path, checkpoint_path):
        """
        This function makes paths for saving model progress, figures and checkpoins
        """
        if not os.path.exists(trainlog_path):
            os.makedirs(trainlog_path)

        if not os.path.exists(confmatrix_path):
            os.makedirs(confmatrix_path)

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "Suturing/")
#a = lstmVAE(path)
#a.iterateall()
