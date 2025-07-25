# Fine-VLA
A V-JEPA-based encoder for fine-grained action video retrieval.
----

### 📝 Preparation 

1. Create uv venv `uv venv`

2. Activate uv venv `source .venv/Scripts/activate`

3. Install required packages `uv pip install -r requirements.txt`

4. Create data / experiment folders `mkdir data; mkdir exps`, note this can just be a symlink to where you want to store big data.


### 🔧 Test (benchmarks: MSR-VTT)

1. Create pretrained model directory `mkdir exps/pretrained`

2. Download pretrained model:
   - `wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar -P exps/pretrained/`

3. Copy config file to pretrained model directory:
   - `cp configs/msrvtt_4f_i21k.json exps/pretrained/config.json`

4. Change `n_gpu` in the config file accordingly (default: 4)

5. Test with pretrained model: `python test.py --resume exps/pretrained/cc-webvid2m-4f_stformer_b_16_224.pth.tar --split test --batch_size 8`

6. Test with your trained model: `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`