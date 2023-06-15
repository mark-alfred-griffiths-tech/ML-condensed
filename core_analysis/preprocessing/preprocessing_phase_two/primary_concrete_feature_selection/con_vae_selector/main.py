import tensorflow
from instantiate_data import InstantiateData
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from create_con_vae_output_decoder_directory import CreateConVAEDecoderDirectory
from pathlib import Path
from output_selected_features import OutputSelectedFeatures
import sys

class SelectorModelCreateFitSave(object):
    def __init__(self, data_dir, results_dir):
        super(SelectorModelCreateFitSave).__init__()
        self.selector_model = None
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.num_feats = float(sys.argv[1])
        self.batch_size = 10000
        self.epochs = 1000
        self.con_vae_dir = CreateConVAEDecoderDirectory(results_dir=self.results_dir, num_feats=self.num_feats)
        self.model_save_filename = 'selector_model_'+str(int(float(sys.argv[1]))+'.h5')
        self.model_save_dir = self.con_vae_dir.con_vae_tf_final_model
        self.model_save_dir_filename = Path(self.model_save_dir, self.model_save_filename)
        self.get_selector_model()
        self.selector_model_fit()
        self.save_model()

    def get_selector_model(self):
        decoder_model = tensorflow.keras.models.load_model(Path(self.model_save_dir,
                                                                self.model_save_filename))
        self.selector_model = ConcreteAutoencoderFeatureSelector(K=self.num_feats, output_function=decoder_model,
                                                                 num_epochs=self.epochs)
    def selector_model_fit(self):
        data = InstantiateData(data_dir=self.data_dir)

        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = Path(self.con_vae_dir.con_vae_tf_epoch_select_model, "cp-{epoch:04d}.ckpt")

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=5 * self.batch_size)

        # Save the weights using the `checkpoint_path` format
        self.selector_model.save_weights(checkpoint_path.format(epoch=0))

        self.selector_model.fit(data.x_train, data.x_train, validation_data=(data.x_test, data.x_test),
                                batch_size=self.batch_size,
                                callbacks=[cp_callback])
        return self.selector_model
    def save_model(self):
        self.selector_model.save(self.model_save_dir_filename)



data_dir = ''
results_dir = ''
num_feats = float(sys.argv[1])
instantiate_data = InstantiateData(data_dir)
SMCFS=SelectorModelCreateFitSave(data_dir, results_dir)
selector=SMCFS.selector_model_fit()
SMCFS.save_model()
OutputSelectedFeatures(results_dir, selector, instantiate_data, num_feats)7

