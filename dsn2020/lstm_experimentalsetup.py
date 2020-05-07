import os, glob, sys, _pickle, time, math, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
from keras import optimizers, Sequential
from keras.utils import plot_model
from keras import layers #Dense, LSTM, RepeatVector, TimeDistributed, Dropout, Masking, BatchNormalization, Flatten, Input, Conv2D, MaxPooling1D, Conv1D, Reshape, GRU
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras import backend as K
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class interpretModel:
    """
    Note: Currently switched suffle to False for both train_generator and train_test_split

    """

    def __init__(self, path, task):
        self.dataPath = path
        self.task = task
        now = datetime.now()
        month_ = datetime.now().month
        day_ = now.day
        hour_ = now.hour
        minute_ = now.minute

    def setData(self, data):
        """
        Allows another class to set data
        """
        self.data = data
        self.train_data = [data[:,0], data[:,1]]
        self.test_data = [data[:,2], data[:,3]]
        #print ("train Data {} test Data {}".format(self.train_data[0].shape, self.test_data[0].shape))


    def buildModelv1(self, timesteps, n_features):
        """
        Simple model which yields a high accuracy of 90% on training data but sucks on the validation data
        """
        model_input = layers.Input(shape=(timesteps, n_features))
        lstm_output = layers.LSTM(256, input_shape=(timesteps, n_features), return_sequences=True)(model_input)
        dropout_output = layers.Dropout(rate=0.5)(lstm_output)
        flatten_output = layers.Flatten()(dropout_output)
        dense_output1 = layers.Dense(128, activation='relu')(flatten_output)
        dropout_output2 = layers.Dropout(rate=0.5)(dense_output1)
        dense_output2 = layers.Dense(15, activation='softmax')(dropout_output2)
        lstm_classifier = Model(model_input, dense_output2)
        #lstm_classifier.summary()

        return lstm_classifier

    def buildModelv2(self, timesteps, n_features):
        """
        An lstm-encoder model followed by dense layers, similar performance to just lstm
        """
        lstm_hidden1 = 128#*4
        lstm_hidden2 = 64
        dense_hidden1 = 64
        output_layer = 15

        model_input = layers.Input(shape=(timesteps, n_features))
        lstm_output = layers.LSTM(lstm_hidden1, input_shape=(timesteps, n_features), return_sequences=True)(model_input) #previously 96
        dropout_output = layers.Dropout(rate=0.4, noise_shape=(None, None, lstm_hidden1))(lstm_output) #previously 0.4
        batch_norm1 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(dropout_output)
        lstm_output2 = layers.LSTM(lstm_hidden2, input_shape=(timesteps, n_features), return_sequences=False)(batch_norm1)
        dropout_output = layers.Dropout(rate = 0.5, noise_shape=(None, lstm_hidden2))(lstm_output2)
        batch_norm2 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(dropout_output)
        repeat_vector = layers.RepeatVector(timesteps)(batch_norm2)

        flatten_output = layers.Flatten()(repeat_vector)
        batch_norm2 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(flatten_output)
        dropout_output = layers.Dropout(rate = 0.5)(batch_norm2)
        dense_output1 = layers.Dense(dense_hidden1, activation='relu')(dropout_output)
        batch_norm3 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(dense_output1)
        dense_output2 = layers.Dense(output_layer, activation='softmax')(batch_norm3)
        lstm_classifier = Model(model_input, dense_output2)
        lstm_classifier.summary()

        return lstm_classifier

    def buildModelv3(self, timesteps, n_features):
        """
        An lstm-encoder model followed by dense layers, similar performance to just lstm
        """
        print ("Using a conv-net approach")
        model_input = layers.Input(shape=(timesteps, n_features))
        lstm_output = layers.Conv1D(filters= 128, kernel_size=3, data_format='channels_first')(model_input) #previously 96
        dropout_output = layers.MaxPooling1D(pool_size=2)(lstm_output)
        flatten_output = layers.Flatten()(dropout_output)
        dense_output1 = layers.Dense(64, activation='relu')(flatten_output)
        batch_norm3 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(dense_output1)
        dense_output2 = layers.Dense(15, activation='softmax')(batch_norm3)
        lstm_classifier = Model(model_input, dense_output2)
        lstm_classifier.summary()

        return lstm_classifier

    def buildModelv4(self, timesteps, n_features):
        """
        An lstm-encoderdecoder model followed by dense layers
        """
        print ("Using a lstm encoder-decoder approach")

        lstm_hidden1 = 128#512
        lstm_hidden2 = 32
        dense_hidden1 = 48
        output_layer = 15

        model_input = layers.Input(shape=(timesteps, n_features))
        lstm_output = layers.LSTM(lstm_hidden1, input_shape=(timesteps, n_features), return_sequences=True)(model_input) #previously 96
        dropout_output = layers.Dropout(rate=0.4, noise_shape=(None, None, lstm_hidden1))(lstm_output) #previously 0.4
        batch_norm1 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(dropout_output)

        flatten_output = layers.Flatten()(batch_norm1)
        batch_norm2 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(flatten_output)
        dropout_output = layers.Dropout(rate = 0.5)(batch_norm2)
        dense_output1 = layers.Dense(dense_hidden1, activation='relu')(dropout_output)
        batch_norm3 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(dense_output1)
        dense_output2 = layers.Dense(output_layer, activation='softmax')(batch_norm3)
        lstm_classifier = Model(model_input, dense_output2)
        lstm_classifier.summary()

        return lstm_classifier

    def buildModelv5(self, timesteps, n_features):
        lstm_hidden1 = 1024
        dense_hidden1 = 64*2
        output_layer = 15

        model_input = layers.Input(shape=(timesteps, n_features))
        lstm_output = layers.LSTM(lstm_hidden1, input_shape=(timesteps, n_features), return_sequences=True)(model_input) #previously 96
        dropout_output = layers.Dropout(rate=0.3, noise_shape=(None, None, lstm_hidden1))(lstm_output) #previously 0.4

        flatten_output = layers.Flatten()(dropout_output)
        dense_output2 = layers.Dense(output_layer, activation='softmax')(flatten_output)
        lstm_classifier = Model(model_input, dense_output2)

        return lstm_classifier

    def trainModel(self, lstm_classifier, mode, user_out, setup, learning_rate, type, currentTimestamp, kinvars):
        """
        train the lstm model using only fit
        """

        train = 1
        epochs = 1#30
        num_epochs = 30#len(self.data)
        #print ("timesteps {} n_features {}".format(self.dataPath, self.task))
        batch_size = 32

        csv_path = os.path.join(self.dataPath, self.task, "GestureClassification", kinvars, currentTimestamp, "csvs")
        checkpoint_path = os.path.join(self.dataPath, self.task, "GestureClassification", kinvars, currentTimestamp, "checkpoints")
        self.makePaths(csv_path, checkpoint_path)

        plot_model(lstm_classifier, to_file='{}/model.png'.format(checkpoint_path),show_shapes=True)

        if train==1:
            for i in range(num_epochs):
                #train and test data
                x_train, y_train, x_test, y_test = self.data[i] #i
                # callbacks
                cp3 = CSVLogger(filename="{}/{}{}{}{}.csv".format(csv_path, setup,  mode, user_out, type), append=False if i==0 else True, separator=';')
                cp2 = ModelCheckpoint(filepath="{}/exp_segment_classifier_{}{}{}{}.h5".format(checkpoint_path, setup,  mode, user_out, type))
                cp1 = EarlyStopping(monitor='val_acc', min_delta=0.0000, patience=3, verbose=0)
                adam = optimizers.Adam(self.step_decay(i, learning_rate))
                lstm_classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

                #tensorboard = TensorBoard(log_dir="logs/{}".format(mode))
                #fitting model
                lstm_classifier.fit(x_train, y_train, batch_size=batch_size, callbacks =[cp1, cp2, cp3], validation_data=(x_test, y_test))

                """
                cf_matrix = confusion_matrix(y_test.argmax(axis=1), model_output.argmax(axis=1))
                #print (cf_matrix)
                cmap=plt.cm.Blues
                fig, ax = plt.subplots()
                im = ax.imshow(cf_matrix, interpolation='nearest', cmap=cmap)
                ax.figure.colorbar(im, ax=ax)
                # We want to show all ticks...
                ax.set(xticks=np.arange(cf_matrix.shape[1]), yticks=np.arange(cf_matrix.shape[0]))
                #plt.show()
                """
        tf.reset_default_graph()
        gc.collect()
        K.clear_session()

    def step_decay(self, epoch, lr):
        """
        step_decay function
        """
        initial_lrate = lr
        drop = 0.5
        epochs_drop = 5.0
        if epoch>10:
            drop=0.5
            epochs_drop = 3

        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        print ("learnign rate {}".format(lrate))
        return lrate

    def makePaths(self, csv_path, checkpoint_path):
        """
        This function makes paths for saving model progress, figures and checkpoins
        """
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
