import os, sys, glob
import keras as K
from keras.models import Model, load_model


class visualizeMaps:
    def __init__(self, path):
        self.data_path = path


    def get_model(self):
        """
        Loading model to check activation
        """
        model_path = "/home/student/Documents/samin/detection/Suturing/loso_experiments_clfv1/All/911334/5/checkpoints_0.0001_5_1_k0/clf_checkpoint_0gestureG6GBE.h5"
        model = load_model(model_path)

    def class_activation_map(self, time_series_original):
        new_input_layer = model.inputs
        new_output_layer = [model.get_layer("conv_final").output, model.layers[-1].output]
        new_function = keras.backend.function(new_input_layer, new_output_layer)



path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "Suturing", "gesture_videos")
