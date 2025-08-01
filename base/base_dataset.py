import os
import random
from abc import abstractmethod

import av
import cv2
import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms


class TextVideoDataset(Dataset):
    def __init__(self,
                dataset_name,  # 데이터셋 이름
                text_params, # 텍스트 처리 파라미터
                video_params, # 비디오 처리 파라미터
                data_dir, # 데이터셋 디렉토리
                metadata_dir=None, # 메타데이터 파일들이 있는 디렉토리 (None이면 data_dir와 동일)
                split='train', # 데이터셋 분할 (train, val, test)
                tsfms=None, # 데이터 변환 함수
                cut=None, # 데이터셋 분할 방법 (None이면 전체 데이터셋 사용)
                subsample=1, # 데이터셋 서브샘플링 비율
                sliding_window_stride=-1, # 비디오 프레임 샘플링 간격
                reader='decord' # 비디오 리더 (av, cv2, decord)
                ):
        self.dataset_name = dataset_name # 인스턴스 변수에 저장
        self.text_params = text_params # 텍스트 파라미터 저장
        self.video_params = video_params # 비디오 파라미터 저장
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir # 데이터 디렉토리와 같게 설정
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption' # 라벨 타입 설정
        self._load_metadata() # 메타데이터 로딩 (자식 클래스에서 구현)
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self): # 슬라이딩 윈도우용 고정 샘플링 설정
        self.metadata['vlen'] = self._get_video_lens() # 각 비디오의 길이를 메타데이터에 추가
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self): # 메타데이터 행 수 반환
        return len(self.metadata)

    def __getitem__(self, item): # 특정 인덱스의 데이터 반환
        item = item % len(self.metadata) # 인덱스를 데이터셋 크기로 모듈로 연산
        sample = self.metadata.iloc[item] # 해당 인덱스의 메타데이터 행 가져오기
        video_fp, rel_fp = self._get_video_path(sample) # 비디오 경로 가져오기
        caption = self._get_caption(sample) # 캡션 가져오기

        video_loading = self.video_params.get('loading', 'strict') # 비디오 로딩 모드
        frame_sample = 'rand' # 기본 프레임 샘플링 : 랜덤
        fix_start = None # 고정 시작점 초기화
        if self.split == 'test': # 테스트 모드면
            frame_sample = 'uniform' # 균등 샘플링
        if self.sliding_window_stride != -1: # 슬라이딩 윈도우 사용 시
            fix_start = sample['fix_start'] # 고정 시작점 사용

        try:
            if os.path.isfile(video_fp): # 비디오 파일이 존재하면
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start) # 비디오 파일 읽기
            else:
                print(f"Warning: missing video file {video_fp}.") # 비디오 파일이 없으면 경고 메시지 출력
                assert False # 프로그램 종료
        except Exception as e:
            if video_loading == 'strict': # 비디오 로딩 모드가 strict일 때
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0)) # 비디오 로딩 실패 시 빈 이미지 생성
                imgs = transforms.ToTensor()(imgs).unsqueeze(0) # 이미지를 텐서로 변환

        if self.transforms is not None: # 데이터 변환 함수가 있으면 적용
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data


class TextImageDataset(TextVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        if self.transforms is not None:
            img = self.transforms(img)
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'meta': meta_arr}
        return data


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs


def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


decord.bridge.set_bridge("torch")


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    'av': read_frames_av,
    'cv2': read_frames_cv2,
    'decord': read_frames_decord
}
