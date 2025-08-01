from base import BaseDataLoaderExplicitSplit, BaseMultiDataLoader
from data_loader.ConceptualCaptions_dataset import ConceptualCaptions3M
from data_loader.LSMDC_dataset import LSMDC
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.NTU_dataset import NTU
from data_loader.WebVid_dataset import WebVid
from data_loader.VideoDirectory_dataset import VideoDirectory, CMDShotFeats
from data_loader.ImageDirectory_dataset import ImageDirectory
from data_loader.transforms import init_transform_dict


def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='decord'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader,
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "NTU":
        dataset = NTU(**kwargs)
    elif dataset_name == "WebVid":
        dataset = WebVid(**kwargs)
    elif dataset_name == "ConceptualCaptions3M":
        dataset = ConceptualCaptions3M(**kwargs)
    elif dataset_name == "LSMDC":
        dataset = LSMDC(**kwargs)
    # ---experimental--- not for public
    elif dataset_name == "VideoDirectory":
        dataset = VideoDirectory(**kwargs)
        dataset = ActivityNet(**kwargs)
    elif dataset_name == "ImageDirectory":
        dataset = ImageDirectory(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,       # 데이터셋 이름
                 text_params,        # 텍스트 처리 파라미터
                 video_params,       # 비디오 처리 파라미터 
                 data_dir,           # 비디오 파일이 있는 디렉토리
                 metadata_dir=None,  # 메타데이터 파일들이 있는 디렉토리
                 split='train',      # 'train', 'val', 'test'
                 tsfm_params=None,   # 이미지 변환 파라미터
                 tsfm_split=None,    
                 cut=None,           # 데이터셋 분할 방법
                 subsample=1,        # 데이터 샘플링 비율
                 sliding_window_stride=-1,
                 reader='decord',    # 비디오 리더 방법 ('decord', 'cv2', 'av')
                 batch_size=1,
                 num_workers=1,
                 prefetch_factor=2,
                 shuffle=True,
                 val_batch_size=None):

        # Transform 파라미터가 없으면 기본값으로 초기화
        if tsfm_params is None:
            tsfm_params = {}

        # 이미지 변환 딕셔너리 생성
        tsfm_dict = init_transform_dict(**tsfm_params)

        # split에 따른 변환 선택
        if tsfm_split is None:
            tsfm_split = split
        tsfm = tsfm_dict[tsfm_split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False
        if val_batch_size is not None and split == 'val':
            batch_size = val_batch_size

        super().__init__(dataset, batch_size, shuffle, num_workers, prefetch_factor=prefetch_factor)
        self.dataset_name = dataset_name


class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)


if __name__ == "__main__":
    kwargs = {
        "dataset_name": "CondensedMovies",
        "data_dir": "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies",
        "shuffle": True,
        "num_workers": 16,
        "batch_size": 16,
        "split": "train",
        "cut": "chall",
        "subsample": 1,
        "text_params": {
            "input": "text",
            "caption_replace_prob": 0.5
        },
        "video_params": {
            "extraction_fps": 25,
            "extraction_res": 256,
            "input_res": 224,
            "num_frames": 4,
            "shot_replace": True,
            "shot_replace_prob": 0.5
        },
        "tsfm_params": {"auto_aug": True}
       }

    dl = TextVideoDataLoader(**kwargs)
    dl.dataset.__getitem__(0)
    for x in range(1000):
        res = next(iter(dl))
        print(x)