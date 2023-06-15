#!/usr/bin/env python 
import numpy as np
import pandas as pd
import os
from pathlib import Path


class CreateConVAEControlFileDirectory:
    def __init__(self, model_dir, *args, **kwargs):
        super(CreateConVAEControlFileDirectory, self).__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.ml_stuttering_project = self.propagate_dir(model_dir, 'ml_stuttering_project')
        self.core_analysis = self.propagate_dir(self.ml_stuttering_project, 'core_analysis')
        self.preprocessing = self.propagate_dir(self.core_analysis, 'preprocessing')
        self.preprocessing_phase_two = self.propagate_dir(self.preprocessing, 'preprocessing_phase_two')
        self.primary_concrete_feature_selection = self.propagate_dir(self.preprocessing_phase_two, 'primary_concrete_feature_selection')
        self.con_vae_selector_control_file = self.propagate_dir(self.primary_concrete_feature_selection, 'con_vae_selector_control_file')



    def propagate_dir(self, old_dir, sub_dir):
        new_dir = Path.home().joinpath(old_dir, str(sub_dir))
        if new_dir.exists():
            pass
        else:
            os.makedirs(new_dir)
        new_dir = str(new_dir)
        return new_dir


class CreateControlFile(object):
    def __init__(self,  min_num_feats,  max_num_feats, increment, control_file, model_dir):
        super(CreateControlFile, self).__init__()
        self.full_list = None
        self.min_num_feats = min_num_feats
        self.max_num_feats = max_num_feats
        self.increment = increment
        self.model_dir = model_dir
        self.control_file_dir = CreateConVAEControlFileDirectory(model_dir=self.model_dir)
        self.control_file = control_file


    def create_full_list(self):
        self.full_list=np.array([])
        for i in range(self.min_num_feats, self.max_num_feats, self.increment):
            self.full_list=pd.DataFrame(np.append(self.full_list,int(i)).astype(np.int32))

    def save_to_file(self):
        self.full_list.columns = ["INTEGER_LIST"]
        self.full_list.to_csv(self.control_file, header=None, index=False)


max_num_feats = 5
min_num_feats = 35
increment = 1

control_file = 'control_file.csv'
model_dir = '/users/k1754828/'

CreateControlFile(min_num_feats, max_num_feats, increment, control_file, model_dir)
