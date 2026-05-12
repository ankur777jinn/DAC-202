pip install -r requirements.txt
python preprocess.py --mode split --split-path ./splits.json
pip install kagglehub
.
torchrun --nproc_per_node=2 train_stage1.py
torchrun --nproc_per_node=1 train_stage1.py
