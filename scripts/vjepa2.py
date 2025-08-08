import torch
import numpy as np
import av
from transformers import AutoVideoProcessor, AutoModel
from PIL import Image
import io
import requests

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

def load_video_from_url(url, num_frames=64):
    response = requests.get(url)
    container = av.open(io.BytesIO(response.content))
    
    stream = container.streams.video[0]
    total_frames = stream.frames
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in frame_indices:
            img = frame.to_image()
            frames.append(img)
        if len(frames) == num_frames:
            break
    
    return frames

frames = load_video_from_url(video_url, num_frames=64)
video = processor(videos=frames, return_tensors="pt").to(model.device)
outputs = model(**video)

encoder_outputs = outputs.last_hidden_state

predictor_outputs = outputs.predictor_output.last_hidden_state

import os
os.makedirs('results/vjepa2', exist_ok=True)
torch.save({
    'encoder_outputs': encoder_outputs.detach().cpu(),
    'predictor_outputs': predictor_outputs.detach().cpu()
}, 'results/vjepa2/video_features.pt')

print(f"인코더 출력 크기: {encoder_outputs.shape}")
print(f"예측기 출력 크기: {predictor_outputs.shape}")
print("결과가 'results/vjepa2/video_features.pt'에 저장되었습니다.")