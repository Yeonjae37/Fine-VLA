import json
import os
import random

import numpy as np
import pandas as pd

from base.base_dataset import TextVideoDataset

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

        self.metadata = df.groupby(['video_id'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample.name + '.mp4'), sample.name + '.mp4'
    
    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption