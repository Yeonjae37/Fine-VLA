import json
import os
import random

import numpy as np
import pandas as pd

from src.data.base_dataset import TextVideoDataset

class NTU(TextVideoDataset):
    def _load_metadata(self):
        csv_fp = os.path.join(self.metadata_dir, 'annotations.csv')

        df = pd.read_csv(csv_fp)

        split_dir = os.path.join(self.metadata_dir, 'splits')
        train_list_path = "train_list.txt"
        test_list_path = "test_list.txt"
        
        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}

        if self.split == 'train':
            df = df[df['video_id'].isin(train_df['videoid'])]
        else:
            df = df[df['video_id'].isin(test_df['videoid'])]

        if self.subsample < 1:
            df = df.sample(frac=self.subsample)

        self.metadata = df.set_index('video_id')[['caption', 'video_path']]
        
        self.metadata['captions'] = self.metadata['caption'].apply(lambda x: [x])

    def _get_video_path(self, sample):
        rel_path = sample['video_path']
        actual_rel_path = os.path.join('video', rel_path)
        abs_path = os.path.join(self.data_dir, actual_rel_path)
        
        return abs_path, actual_rel_path
    
    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption