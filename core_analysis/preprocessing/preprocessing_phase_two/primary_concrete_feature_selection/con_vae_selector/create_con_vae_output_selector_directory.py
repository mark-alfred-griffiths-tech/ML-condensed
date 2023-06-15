import os
from pathlib import Path


class CreateConVAESelectorDirectory:
    def __init__(self, results_dir, feat_num, *args, **kwargs):
        super(CreateConVAESelectorDirectory, self).__init__(*args, **kwargs)
        self.results_dir = results_dir
        self.con_vae_selector_tf = self.propagate_dir(results_dir, 'con_vae_' + str(feat_num) + '_selector_tf')
        self.con_vae_selector_tf_results = self.propagate_dir(self.con_vae_selector_tf,
                                                              'con_vae_' + str(feat_num) + 'selector_tf_results')
        self.con_vae_selector_tf_final_model = self.propagate_dir(self.con_vae_selector_tf, 'con_vae_' + str(
            feat_num) + 'selector_tf_final_model')
        self.con_vae_selector_tf_epoch_select_model = self.propagate_dir(self.con_vae_selector_tf,
                                                                         'con_vae_selector_' + str(
                                                                             feat_num) + 'tf_epoch_select_model')
        self.con_vae_selector_tf_pretraining = self.propagate_dir(self.con_vae_selector_tf, 'con_vae_selector' + str(
            feat_num) + '_tf_pretraining')

    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir
