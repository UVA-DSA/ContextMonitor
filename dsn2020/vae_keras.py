from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, LSTM, Flatten, TimeDistributed, Dropout, RepeatVector
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger, LearningRateScheduler
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, math, gc

K.tensorflow_backend._get_available_gpus()
class VAE:
    def __init__(self, path, task):
        self.dataPath = path
        self.task = task
        self.figure_path = os.path.join(path, task, "experiments", "vae")

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

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def splitdata(self, x_train, x_label):
        """
        splits the data to train set and validation set
        """
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, x_label, test_size=0.2, random_state=42, shuffle=True)
        return x_train, x_valid, y_train, y_valid


    def distinguishSequence(self, all_kinematics, all_labels):
        anomalous_trajectory = np.asarray(all_kinematics)[np.where(all_labels>0)[0]]
        nonanomalous_trajectory = np.asarray(all_kinematics)[np.where(all_labels==0)[0]]
        anomalous_label = np.asarray(all_labels)[np.where(all_labels>0)[0]]
        nonanomalous_label = np.asarray(all_labels)[np.where(all_labels==0)[0]]

        return nonanomalous_trajectory, anomalous_trajectory, nonanomalous_label, anomalous_label

    def getInput(self, data, mode="not_concatenate"):
        for i in range(data.shape[0]):
            x_train, y_train, x_test, y_test = data[i]
            #x_train = x_train.reshape(x_train.shape[0], x_train.shape[-1])
            #x_test = x_test.reshape(x_test.shape[0], x_test.shape[-1])
            kinematics = x_train[:,0:-15]
            labels = x_train[:,-15:]
            if mode =="not_concatenate":
                print ("separating feature vectors from gesture vectors")
                x_train, x_test = x_train[:,:,0:-15], x_test[:,:,0:-15]
            else:
                print ("runnning pipeline using concatenated setup")

            #x_train, x_test = x_train[:,0:-15], x_test[:,0:-15]
            nonanomalous_trajectory, anomalous_trajectory, nonanomalous_label, anomalous_label = self.distinguishSequence(x_train, y_train)
            X_train_y0_scaled, X_valid_y0_scaled, y_train_y0_scaled, y_valid_y0_scaled =self.splitdata(nonanomalous_trajectory, nonanomalous_label)
            x_test = np.concatenate((x_test, anomalous_trajectory), axis=0)
            y_test = np.concatenate((y_test, anomalous_label), axis=0)
            vae_data = [X_train_y0_scaled, X_valid_y0_scaled, x_test, y_test]

            self.trainModel(currentTimestamp, vae_data, itr_num=i)

    def getCategorizedData(self, currentTimestamp, data, trial):
        """
        gets input from another class, segments according to the error and then segments according to the gesture
        """
        print ("entered")

        for i in range(data.shape[0]):
            x_train, y_train, x_test, y_test = data[i]
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
                    print ("anomaly shape {}".format(anomalous_trajectory.shape))
                    testgesture_dict[key] = np.concatenate((testgesture_dict[key], anomalous_trajectory), axis=0)
                    testlabel_dict[key] = np.concatenate((testlabel_dict[key], anomalous_label), axis=0)
                    gestureX_train_y0_scaled, gestureX_valid_y0_scaled, _, _ =self.splitdata(nonanomalous_trajectory, nonanomalous_label)

                    vae_data = [gestureX_train_y0_scaled, gestureX_valid_y0_scaled, testgesture_dict[key], testlabel_dict[key]]

                    self.trainModel(currentTimestamp, vae_data, trial, itr_num = i, gesture=key, train=1)

    def cataegorizeData(self, kinematics, gestures, labels):
        """
        data received will be categorized into gestures before passing to the LSTM-VAE
        """
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

    def buildlstmvae(self, timesteps, n_features, figure_path, trial, itr_num, gesture):

        encoder1_dims = 1024
        encoder2_dims = 512
        encoder3_dims = 128
        encoder4_dims = 8

        inputs = Input(shape=(timesteps, n_features,), name='encoder_input')
        lstm_output = LSTM(encoder1_dims, return_sequences=True, go_backwards=True)(inputs)
        dropout_output = Dropout(rate=0.4)(lstm_output) #previously 0.4
        lstm_output2 = LSTM(encoder2_dims,  return_sequences=True)(dropout_output)
        flatten_output = Flatten()(lstm_output2)
        encoder2 = Dense(encoder3_dims, activation='relu')(flatten_output)
        z_mean = Dense(encoder4_dims, name='z_mean')(encoder2)
        z_log_var = Dense(encoder4_dims, name='z_log_var')(encoder2)

        z = Lambda(self.sampling, output_shape=(encoder4_dims,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        latent_inputs = Input(shape=(encoder4_dims,), name='z_sampling')
        repeat_vector = RepeatVector(timesteps)(latent_inputs)
        lstm_decoder1 = LSTM(encoder2_dims, activation='relu', return_sequences=True)(repeat_vector)
        lstm_decoder2 = LSTM(encoder1_dims, activation='relu', return_sequences=True)(lstm_decoder1)
        outputs = TimeDistributed(Dense(n_features))(lstm_decoder2)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        #decoder.summary()
        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        models = (encoder, decoder)
        reconstruction_loss = mse(inputs, outputs)

        reconstruction_loss *= n_features
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + 0.2*kl_loss)
        vae.add_loss(vae_loss)

        return vae

    def buildvae1(self, original_dim, figure_path, trial, itr_num, gesture):

        input_shape = (original_dim, )
        encoder1_dims = 1024
        encoder2_dims = 512
        encoder3_dims = 128
        encoder4_dims = 64

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        encoder1 = Dense(encoder1_dims, activation='relu')(inputs)
        encoder2 = Dense(encoder2_dims, activation='relu')(encoder1)
        z_mean = Dense(encoder3_dims, name='z_mean')(encoder2)
        z_log_var = Dense(encoder3_dims, name='z_log_var')(encoder2)

        z = Lambda(self.sampling, output_shape=(encoder3_dims,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')



        # build decoder model
        latent_inputs = Input(shape=(encoder3_dims,), name='z_sampling')
        decoder1 = Dense(encoder2_dims, activation='relu')(latent_inputs)
        decoder2 = Dense(encoder1_dims, activation='relu')(decoder1)
        outputs = Dense(original_dim, activation='sigmoid')(decoder2)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        #decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        models = (encoder, decoder)
        reconstruction_loss = mse(inputs, outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss)# + kl_loss)
        vae.add_loss(vae_loss)


        return vae

    def trainModel(self, currentTimestamp, vae_data, trial="-1", itr_num=0, gesture="g", train=0):
        x_train, x_valid, x_test, y_test = vae_data
        batch_size = 32
        epochs = 150
        figure_path = os.path.join(self.dataPath, self.task, "experiments", trial, "figures")
        trainlog_path = os.path.join(self.dataPath, self.task, "experiments", trial, "csvs", "trainlog")
        confmatrix_path = os.path.join(self.dataPath, self.task, "experiments", trial, "csvs", "confusion_matrix")
        checkpoint_path = os.path.join(self.dataPath, self.task, "experiments", trial, "checkpoints")
        self.makePaths(figure_path, trainlog_path, confmatrix_path, checkpoint_path)
        #self.makePaths(figure_path, csv_path, checkpoint_path)
        print ("x train shape {}".format(x_train.shape))
        _,timesteps, features = x_train.shape
        vae = self.buildlstmvae(timesteps, features, figure_path, trial, itr_num, gesture)

        adam = optimizers.Adam(lr=0.1)
        vae.compile(optimizer=adam)
        #vae.summary()
        #plot_model(vae,to_file='vae_mlp.png',show_shapes=True)
        # train the autoencoder

        cp1 = EarlyStopping(monitor='val_loss', min_delta=0.0000, patience=5, verbose=0)
        cp2 = ModelCheckpoint(filepath="{}/vae_checkpoint_ts{}_{}gesture{}.h5".format(checkpoint_path, currentTimestamp, itr_num, gesture), save_best_only=True, verbose=0)
        cp3 = CSVLogger(filename="{}/vae_model_ts{}_{}gesture{}.csv".format(trainlog_path, currentTimestamp, itr_num, gesture), append=False, separator=';')
        cp4=LearningRateScheduler(self.step_decay)
        vae_history = vae.fit(x_train, callbacks=[cp1,cp2,cp3,cp4], verbose=0,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_valid, None)).history
        vae.save_weights('vae_mlp_mnist.h5')

        test_x_predictions = vae.predict(x_test)
        test_mse = np.mean(np.power((self.flatten(x_test) - self.flatten(test_x_predictions)), 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error': list(test_mse),
                         'True_class': list(y_test[:,0])})
        mae_of_predictions = np.squeeze(np.max(np.square(test_x_predictions - x_test), axis=1))

        mae_threshold = np.mean(mae_of_predictions) + np.std(mae_of_predictions)  # can use a running mean instead.
        actual = np.where(mae_of_predictions > mae_threshold)[0]
        threshold_fixed = vae_history['val_loss'][-1]
        groups = error_df.groupby('True_class')
        pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.True_class, pred_y)
        conf_matrixdf = pd.DataFrame(data=conf_matrix)
        conf_matrixdf.to_csv("{}/vaeconf_ts{}_{}gesture{}.csv".format(confmatrix_path, currentTimestamp, itr_num, gesture))

        fig, ax = plt.subplots()
        for name, group in groups:
         ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='', label= "Safety-critical" if name == 1 else "Normal")
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig("{}/separation_ts{}{}gesture{}.pdf".format(figure_path, currentTimestamp, itr_num, gesture))
        plt.close()
        K.clear_session()
        tf.reset_default_graph()
        gc.collect()


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

    def step_decay(self, epoch):
        """
        step_decay function
        """
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 15.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
