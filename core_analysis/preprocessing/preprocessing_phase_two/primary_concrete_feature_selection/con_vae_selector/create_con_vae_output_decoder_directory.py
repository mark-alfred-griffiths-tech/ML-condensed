import os
from pathlib import Path


class CreateConVAEDecoderDirectory:
    def __init__(self, results_dir, feat_num, *args, **kwargs):
        super(CreateConVAEDecoderDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.con_vae_decoder_tf = self.propagate_dir(results_dir, 'con_vae_' + str(feat_num) + '_decoder_tf')
        self.con_vae_decoder_tf_results = self.propagate_dir(self.con_vae_decoder_tf,
                                                             'con_vae_' + str(feat_num) + 'decoder_tf_results')
        self.con_vae_decoder_tf_final_model = self.propagate_dir(self.con_vae_decoder_tf,
                                                                 'con_vae_' + str(feat_num) + 'decoder_tf_final_model')
        self.con_vae_decoder_tf_epoch_select_model = self.propagate_dir(self.con_vae_decoder_tf, 'con_vae_' + str(
            feat_num) + 'decoder_tf_epoch_select_model')
        self.con_vae_decoder_tf_pretraining = self.propagate_dir(self.con_vae_decoder_tf,
                                                                 'con_vae' + str(feat_num) + '_tf_pretraining')
        self.con_vae_decoder_tf_tensorboard = self.propagate_dir(self.con_vae_decoder_tf_pretraining,
                                                                 'con_vae_' + str(feat_num) + 'decoder_tf_tensorboard')
        self.con_vae_tf_decoder_partial_models = self.propagate_dir(self.con_vae_decoder_tf_pretraining,
                                                                    'con_vae_' + str(feat_num) + 'tf_partial_models')

    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir
