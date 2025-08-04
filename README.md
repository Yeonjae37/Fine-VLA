# Fine-VLA
A V-JEPA-based encoder for fine-grained action video retrieval.
----

### 📝 Preparation 

1. Create uv venv `uv venv`

2. Install required packages `uv sync`

3. Activate uv venv `source .venv/Scripts/activate`

### 🔧 Test (benchmarks: MSR-VTT)

1. Download pretrained model:
   - `wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar -P src/exps/pretrained/`

2. Change `n_gpu` in the config file accordingly (default: 4)

3. Test with pretrained model: `python scripts/test.py -c src/exps/pretrained/config.json -r src/exps/pretrained/cc-webvid2m-4f_stformer_b_16_224.pth.tar`

4. Test with your trained model: `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`