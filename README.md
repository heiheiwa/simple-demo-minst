# simple-demo-minst
simple demo for minist train
Create & activate a virtualenv
python3.12 -m venv venv
source venv/bin/activate
Upgrade pip
python -m pip install --upgrade pip
3A) Option A — install from a requirements.txt (recommended if you already have it)

Create requirements.txt with: torch==2.5.0 torchvision==0.20.0 tqdm
Then install: pip install -r requirements.txt
3B) Option B — explicit CPU pip install (works if pip needs the PyTorch wheel index)

pip install --upgrade pip
pip install "torch==2.5.0+cpu" "torchvision==0.20.0+cpu" tqdm -f https://download.pytorch.org/whl/torch_stable.html
(If you need CUDA, use the selector at https://pytorch.org/get-started/locally to get the exact pip command for your CUDA version.)

Quick sanity check
python -c "import torch; print('torch', torch.version, 'cuda_available', torch.cuda.is_available())"
Run the training script
python3.12 train.py --data-dir ./data --epochs 5 --batch-size 64
Common variants
Force CPU even if GPU present: python3.12 train.py --no-cuda --data-dir ./data --epochs 5
Run fewer DataLoader workers (if you see worker spawn errors): Edit train.py DataLoader num_workers to 0, or set an env var when running: NUM_WORKERS=0 python3.12 train.py --data-dir ./data --epochs 5
