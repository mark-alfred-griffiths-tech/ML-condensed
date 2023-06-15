#!/usr/bin/env python
# coding: utf-8
import keras_tuner as kt
import tensorflow
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from add_dim_x_num_cats import DimXNumCats
from instantiate_data import InstantiateData
from create_con_vae_output_directory import CreateConVAEDirectory
from pathlib import Path
import os

class DecoderLayer(tensorflow.keras.Layer):
    def __init__(self, batch_size, epochs):
        super(DecoderLayer, self).__init__()
        data = InstantiateData(data_dir='/scratch/users/k1754828/data')
        data = DimXNumCats(data)
        self.decoder = self.obtain_decoder(data)
        self.batch_size = batch_size
        self.epochs = epochs

    def get_best_concrete_autoencoder_decoder_pretrain_hyperparameters(self, model_dir):
        tuner = kt.Hyperband(self.decoder_base, objective='val_accuracy', max_epochs=1000, factor=3, directory=model_dir,
                             project_name='con_vae_tf_tensorboard')

        best_hps = tuner.get_best_hyperparameters(1)[0]
        return best_hps

    def obtain_decoder(self, data):
        best_hps = self.get_best_concrete_autoencoder_decoder_pretrain_hyperparameters(
            model_dir='/scratch/users/k1754828/results/con_vae_tf/con_vae_tf_pretraining/')

        decoder_layer = Sequential()

        decoder_layer.add(Dense(data.dim_x, activation=best_hps.get('ni_neurons_activation'), name='ni_neurons'))
        decoder_layer.add(
            Dense(best_hps.get('nj_neurons'), activation=best_hps.get('nj_neurons_activation'),
                  name='nj_neurons_layer'))
        decoder_layer.add(
            Dense(best_hps.get('nk_neurons'), activation=best_hps.get('nk_neurons_activation'),
                  name='nk_neurons_layer'))
        decoder_layer.add(
            Dense(best_hps.get('nl_neurons'), activation=best_hps.get('nl_neurons_activation'),
                  name='nl_neurons_layer'))
        decoder_layer.add(Dense(data.dim_x, activation=best_hps.get('nm_neurons_activation'), name='nm_neurons_layer'))

        return decoder_layer

    def fit(self, input):
        return self.decoder_layer(input)


class DecoderModel(tensorflow.keras.Model):
    def __init__(self, batch_size, epochs, best_hps):
        super(DecoderModel, self).__init__()
        decoder_model = DecoderLayer(batch_size, epochs)
        decoder_model.compile(loss="categorical_crossentropy",
                              optimizer=SGD(momentum=best_hps.get('optimizer_momentum_float_value'),
                                            clipnorm=best_hps.get('optimizer_clipnorm_float_value')))

    def fit(self, x_train, y_train, x_test, y_test)
        self.decoder_model.fit(x_train, y_train, validation=(x_test, y_test), batch=self.batch_size, epochs=self.epochs)


def save_model(decoder_model, model_save_dir, model_save_filename):
    model_save_dir_filename = Path(model_save_dir, model_save_filename)
    decoder_model.save(model_save_dir_filename)


class RecoverModelFitSaveDecoder(object):
    def __init__(self, batch_size, epochs, data_dir, results_dir):
        super(RecoverModelFitSaveDecoder).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_dir = data_dir
        self.results_dir = results_dir

        self.con_vae_dir = CreateConVAEDirectory(results_dir=self.results_dir)
        self.model_save_filename = 'decoder_model.h5'
        self.decoder_model = DecoderModel(batch_size=self.batch_size, epochs=self.epochs)

        # Include the epoch in the file name (uses `str.format`)
        self.checkpoint_path = Path(self.con_vae_dir.con_vae_tf_epoch_select_model, "cp-{epoch:04d}.ckpt")
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def fit_model(self):
        # Create a callback that saves the model's weights every 5 epochs
        self.cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    verbose=1,
                    save_weights_only=True,
                    save_freq=5 * self.batch_size)

        # Save the weights using the `checkpoint_path` format
        self.decoder_model.save_weights(self.checkpoint_path.format(epoch=0))

        data = InstantiateData(data_dir=self.data_dir)

        self.decoder_model.fit(data.x_train, data.y_train, data.x_test, data.y_test, callbacks=[self.cp_callback])

    def save_model(self):
        save_model(self.decoder_model, self.con_vae_dir.con_vae_tf_final_model, self.model_save_filename)

#RUN MAIN
batch_size = 10000
epochs = 1000
data_dir = './'
results_dir ='./'
RecoverModelFitSaveDecoder(batch_size, epochs, data_dir, results_dir)