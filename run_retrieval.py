import os
import sys
import argparse
import copy
import numpy as np
import torch
import transformers
import pathlib
import platform
import re
import faiss

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
import pandas as pd
from tqdm import tqdm
import tempfile
import pickle
import json
from pathlib import Path

import numpy as np

class SimpleIndex:
    """
    - 항상 Inner Product(IP) 기반 (L2 정규화 후 코사인과 동치).
    - add/search 시 자동 L2 정규화.
    - FAISS 있으면 IndexFlatIP, 없으면 넘파이 행렬 사용.
    """
    def __init__(self, dim, use_faiss=True):
        self.dim = dim
        self.use_faiss = (faiss is not None) and use_faiss
        self._matrix = None
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    @staticmethod
    def _l2norm(x, axis=1, eps=1e-8):
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (n + eps)

    def add(self, X):
        X = X.astype('float32')
        X = self._l2norm(X)
        if self.use_faiss:
            faiss.normalize_L2(X)
            self.index.add(X)
        # 항상 raw matrix도 유지(저장/로드/넘파이 검색)
        self._matrix = X if self._matrix is None else np.vstack([self._matrix, X])

    def search(self, Q, topk):
        Q = Q.astype('float32')
        Q = self._l2norm(Q)
        if self.use_faiss:
            faiss.normalize_L2(Q)
            scores, idxs = self.index.search(Q, topk)
            return scores, idxs
        else:
            scores = Q @ self._matrix.T  # (B, N)
            idxs = np.argsort(-scores, axis=1)[:, :topk]
            picked = np.take_along_axis(scores, idxs, axis=1)
            return picked, idxs

    def all_vectors(self):
        return self._matrix  # 저장용

sys.path.append('.')
sys.path.append('./scripts')
import src.data.data_loader as module_data
import src.model.model as module_arch
from parse_config import ConfigParser
from src.utils.util import state_dict_data_parallel_fix
from src.data.base_dataset import read_frames_decord, sample_frames
from src.data.transforms import init_transform_dict
import torchvision.transforms as T

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/uploads/*": {"origins": "*"}
}, supports_credentials=True)

class VideoRetrievalSystem:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        self.model, self.tokenizer = self._load_model(checkpoint_path)
        self.index = None            # ← 심플 인덱스 핸들
        self.segment_metadata = []   # 세그먼트 메타
        self._emb_dim = None         # 임베딩 차원
        self._emb_matrix = None      # 저장/로드용 임베딩 행렬

        self._num_frames = int(self.config['arch']['args']['video_params'].get('num_frames', 4))
        
        # transforms 초기화 (test.py와 동일)
        video_params = self.config['arch']['args']['video_params']
        self.video_params = video_params
        transform_dict = init_transform_dict(
            input_res=video_params.get('input_res', 224),
            center_crop=video_params.get('center_crop', 256)
        )
        self.transforms = transform_dict['test']  # test용 transform 사용
        
        self.load_index_and_metadata()

    def _load_config(self, config_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', default=config_path, type=str)
        parser.add_argument('-r', '--resume', default=None, type=str)
        parser.add_argument('-d', '--device', default=None, type=str)

        
        original_argv = sys.argv
        sys.argv = ['run_retrieval.py', '--config', config_path]
        
        try:
            config = ConfigParser(parser, test=True)
        finally:
            sys.argv = original_argv
            
        return config
    
    def _load_model(self, checkpoint_path):
        model = self.config.initialize('arch', module_arch)

        checkpoint_path and os.path.exists(checkpoint_path)
        if str(checkpoint_path).endswith(".pkl"):
            import pickle
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
        else:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
            
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=False)
        
        model = model.to(self.device).eval()

        text_model_name = self.config['arch']['args']['text_params']['model']
        if "openai/clip" in text_model_name:
            tokenizer_builder = transformers.CLIPTokenizer
        else:
            tokenizer_builder = transformers.AutoTokenizer
            
        tokenizer = tokenizer_builder.from_pretrained(
            text_model_name,
            model_max_length=int(self.config['arch']['args']['text_params'].get('max_length', 1e6)),
            TOKENIZERS_PARALLELISM=False,
            local_files_only=True
        )
        
        return model, tokenizer
    
    def segment_video(self, video_path, segment_duration=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        segments = []
        segment_frames = int(fps * segment_duration)
        
        for start_frame in range(0, total_frames, segment_frames):
            end_frame = min(start_frame + segment_frames, total_frames)
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            if end_time - start_time >= 1.0:
                segments.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
        
        cap.release()
        print(f"Video segmented into {len(segments)} segments of ~{segment_duration}s each")
        return segments
    
    def extract_video_segment_embedding(self, video_path, start_time, end_time):
        segment_data = {
            'video': self._load_video_segment_frames(video_path, start_time, end_time),
            'text': [''],
            'meta': {'paths': [video_path]}
        }
        temp_segment_path = video_path

        if self.tokenizer is not None:
            segment_data['text'] = self.tokenizer(
                segment_data['text'], 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            segment_data['text'] = {
                key: val.to(self.device) for key, val in segment_data['text'].items()
            }

        if isinstance(segment_data['video'], list):
            segment_data['video'] = [x.to(self.device) for x in segment_data['video']]
        else:
            segment_data['video'] = segment_data['video'].to(self.device)

        with torch.no_grad():
            text_embeds, vid_embeds = self.model(segment_data)
            
        return vid_embeds.cpu().numpy()
    
    def _create_temp_segment(self, video_path, start_time, end_time):
        temp_dir = tempfile.mkdtemp()
        safe_filename = f"segment_{start_time:.1f}_{end_time:.1f}".replace('.', '_')
        temp_path = os.path.join(temp_dir, f"{safe_filename}.mp4")

        video_path = os.path.abspath(video_path) # for Window
        temp_path = os.path.abspath(temp_path) # for Window

        if os.name == 'nt':  # Windows
            cmd = f'ffmpeg -i "{video_path}" -ss {start_time} -t {end_time - start_time} -c copy "{temp_path}" -y -loglevel quiet'
        else:  # Linux/Mac
            cmd = f'ffmpeg -i "{video_path}" -ss {start_time} -t {end_time - start_time} -c copy "{temp_path}" -y > /dev/null 2>&1'
        
        return temp_path
    
    def _load_video_segment_frames(self, video_path, start_time, end_time):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                res = self.video_params.get('input_res', 224)
                return torch.zeros(1, self._num_frames, 3, res, res)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0  # fallback

            s = int(round(start_time * fps))
            e = max(s + 1, int(round(end_time * fps)))
            total = e - s

            idxs = np.linspace(s, e - 1, self._num_frames).round().astype(int)

            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

            cap.release()

            res = self.video_params.get('input_res', 224)
            if not frames:
                return torch.zeros(1, self._num_frames, 3, res, res)

            import torchvision.transforms as T
            tfm = T.Compose([
                T.ToPILImage(),
                T.Resize(self.video_params.get('center_crop', 256)),
                T.CenterCrop(self.video_params.get('input_res', 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

            tensor_frames = [tfm(f) for f in frames]  # list of (3,H,W) tensors
            while len(tensor_frames) < self._num_frames:
                tensor_frames.append(tensor_frames[-1].clone())

            video_tensor = torch.stack(tensor_frames[:self._num_frames], dim=0)  # (T,3,H,W)
            return video_tensor.unsqueeze(0)  # (1,T,3,H,W)

        except Exception as e:
            print(f"Error loading segment frames: {e}")
            res = self.video_params.get('input_res', 224)
            return torch.zeros(1, self._num_frames, 3, res, res)


    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return torch.zeros(1, 16, 3, 224, 224)
        
        frames = []
        frame_count = 0
        max_frames = 16
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()

        if frames: # 프레임 텐서로 변환
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor_frames = torch.stack([transform(frame) for frame in frames])

            if len(tensor_frames) < 16:
                padding = torch.zeros(16 - len(tensor_frames), 3, 224, 224)
                tensor_frames = torch.cat([tensor_frames, padding], dim=0)
            
            return tensor_frames.unsqueeze(0)
            print(f"No frames extracted from {video_path}")
            return torch.zeros(1, 16, 3, 224, 224)
    
    def build_faiss_index(self, video_path, segment_duration=5):
        print(f"Building index for video: {video_path}")
        segments = self.segment_video(video_path, segment_duration)

        embeds, metadata = [], []
        print("Extracting embeddings for each segment...")
        for i, seg in enumerate(tqdm(segments)):
            try:
                e = self.extract_video_segment_embedding(video_path, seg['start_time'], seg['end_time'])
                embeds.append(e.reshape(1, -1))
                metadata.append({
                    'segment_id': i,
                    'video_path': video_path,
                    'start_time': seg['start_time'],
                    'end_time': seg['end_time'],
                    'duration': seg['duration'],
                })
            except Exception as ex:
                print(f"[WARN] segment {i} skipped: {ex}")

        if not embeds:
            raise ValueError("No valid embeddings extracted")

        X = np.vstack(embeds).astype('float32')   # (N, D)
        self._emb_dim = X.shape[1]
        self.index = SimpleIndex(self._emb_dim, use_faiss=True)
        self.index.add(X)

        self.segment_metadata = metadata
        self._emb_matrix = self.index.all_vectors()

        print(f"Index built: {len(self.segment_metadata)} segments, dim={self._emb_dim}")
        self.save_index_and_metadata()

    
    def search_query(self, query_text, top_k=8):
        if not self.segment_metadata:
            raise ValueError("Index not built. Call build_faiss_index first.")

        # 인덱스가 없다면 저장된 행렬로 재구축
        if self.index is None:
            if self._emb_matrix is None:
                raise ValueError("No index and no embedding matrix to rebuild from.")
            self.index = SimpleIndex(self._emb_dim, use_faiss=True)
            self.index.add(self._emb_matrix)

        # 쿼리 임베딩
        toks = self.tokenizer([query_text], return_tensors='pt', padding=True, truncation=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            q = self.model.compute_text(toks).cpu().numpy().astype('float32')  # (1, D)

        scores, idxs = self.index.search(q, top_k)
        scores, idxs = scores[0], idxs[0]

        results = []
        for rank, (sc, ix) in enumerate(zip(scores, idxs), start=1):
            if ix < len(self.segment_metadata):
                md = self.segment_metadata[ix]
                results.append({
                    'rank': rank,
                    'score': float(sc),
                    'segment_id': md['segment_id'],
                    'start_time': md['start_time'],
                    'end_time': md['end_time'],
                    'duration': md['duration'],
                    'video_path': md['video_path'],
                })
        return results

    
    def save_index_and_metadata(self, save_dir="saved_indices"):
        try:
            os.makedirs(save_dir, exist_ok=True)
            # 메타/행렬/설정 저장
            np.save(os.path.join(save_dir, "segment_metadata.npy"), self.segment_metadata)
            if self._emb_matrix is not None:
                np.save(os.path.join(save_dir, "embeddings.npy"), self._emb_matrix)
            with open(os.path.join(save_dir, "index_config.json"), "w") as f:
                json.dump({"embed_dim": self._emb_dim}, f)
            print(f"Saved embeddings & metadata to {save_dir}")
        except Exception as e:
            print(f"[WARN] save_index_and_metadata: {e}")

    
    def load_index_and_metadata(self, save_dir="saved_indices"):
        try:
            cfg = os.path.join(save_dir, "index_config.json")
            meta = os.path.join(save_dir, "segment_metadata.npy")
            embs = os.path.join(save_dir, "embeddings.npy")
            if not (os.path.exists(cfg) and os.path.exists(meta) and os.path.exists(embs)):
                return False

            with open(cfg, "r") as f:
                j = json.load(f)
            self._emb_dim = int(j["embed_dim"])
            self.segment_metadata = np.load(meta, allow_pickle=True).tolist()
            self._emb_matrix = np.load(embs).astype('float32')

            self.index = SimpleIndex(self._emb_dim, use_faiss=True)
            self.index.add(self._emb_matrix)

            print(f"Loaded {len(self.segment_metadata)} segments (dim={self._emb_dim}) from {save_dir}")
            return True
        except Exception as e:
            print(f"[WARN] load_index_and_metadata: {e}")
            return False


retrieval_system = None

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global retrieval_system
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uploads_dir = os.path.join(project_root, 'uploads')
    video_path = os.path.join(uploads_dir, video_file.filename)
    os.makedirs(uploads_dir, exist_ok=True)
    video_file.save(video_path)
    print(f"Video saved to: {video_path}")
    
    try:
        if retrieval_system is None:
            return jsonify({'error': 'Retrieval system not initialized'}), 500
        
        retrieval_system.build_faiss_index(video_path)
        
        return jsonify({
            'success': True,
            'message': f'Video processed successfully. {len(retrieval_system.segment_metadata)} segments indexed.',
            'video_path': video_path
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    global retrieval_system
    
    if retrieval_system is None or not hasattr(retrieval_system, 'segment_metadata') or len(retrieval_system.segment_metadata) == 0:
        return jsonify({'error': 'No video indexed. Please upload a video first.'}), 400
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    top_k = data.get('top_k', 8)
    
    try:
        results = retrieval_system.search_query(query, top_k)
        return jsonify({
            'success': True,
            'query': query,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/api/get_segment', methods=['GET'])
def get_segment():
    segment_id = request.args.get('segment_id', type=int)
    
    if retrieval_system is None or segment_id is None:
        return jsonify({'error': 'Invalid request'}), 400
    
    if segment_id >= len(retrieval_system.segment_metadata):
        return jsonify({'error': 'Segment not found'}), 404
    
    metadata = retrieval_system.segment_metadata[segment_id]

    video_filename = os.path.basename(metadata['video_path'])
    
    return jsonify({
        'success': True,
        'video_url': f'/uploads/{video_filename}',
        'start_time': metadata['start_time'],
        'end_time': metadata['end_time'],
        'duration': metadata['duration']
    })

def stream_video(file_path, range_header=None):
    file_size = os.path.getsize(file_path)
    
    if range_header:
        byte_start = 0
        byte_end = file_size - 1
        
        range_match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            byte_start = int(range_match.group(1))
            if range_match.group(2):
                byte_end = int(range_match.group(2))
        
        content_length = byte_end - byte_start + 1
        
        def generate():
            with open(file_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining:
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        response = Response(
            generate(),
            status=206,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Accept-Ranges': 'bytes',
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Content-Length': str(content_length),
                'Content-Type': 'video/mp4',
            }
        )
        return response
    else:
        def generate():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        response = Response(
            generate(),
            headers={
                'Access-Control-Allow-Origin': '*',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(file_size),
                'Content-Type': 'video/mp4',
            }
        )
        return response

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        uploads_dir = os.path.join(project_root, 'uploads')
        file_path = os.path.join(uploads_dir, filename)
        
        print(f"Looking for file: {filename} in {uploads_dir}")
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {filename}'}), 404

        range_header = request.headers.get('Range')
        print(f"Range header: {range_header}")
        
        return stream_video(file_path, range_header)
        
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'File serving error: {str(e)}'}), 500

@app.route('/')
def index():
    html_path = os.path.join(os.path.dirname(__file__), 'retrieval_interface.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
        return response

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    return response

def initialize_system(config_path, checkpoint_path):
    global retrieval_system
    retrieval_system = VideoRetrievalSystem(config_path, checkpoint_path)
    print("Video retrieval system initialized successfully!")

if __name__ == '__main__':
    config_path = 'src/exps/pretrained/config.json'
    checkpoint_path = 'src/exps/pretrained/cc-webvid2m-4f_stformer_b_16_224.pth.tar'

    initialize_system(config_path, checkpoint_path)
    
    app.run(host='0.0.0.0', port=5000, debug=True)