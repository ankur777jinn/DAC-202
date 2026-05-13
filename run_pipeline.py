#!/usr/bin/env python3
"""ONE-SHOT: Download data + Train + Evaluate + Visualize. Just run: python run_pipeline.py"""
import os, sys, time, subprocess, json

def run(cmd, desc):
    print(f'\n{"="*60}\n  {desc}\n  $ {cmd}\n{"="*60}')
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    status = 'DONE' if ret.returncode == 0 else 'FAILED'
    print(f'--- {status} ({dt:.0f}s) ---')
    if ret.returncode != 0: sys.exit(ret.returncode)
    return dt

def setup_kaggle():
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json):
        os.makedirs(kaggle_dir, exist_ok=True)
        creds = {'username': 'ankur777jinn', 'key': 'aa4ec9b24f4dc93c9ae18bb1791f2800'}
        with open(kaggle_json, 'w') as f:
            json.dump(creds, f)
        os.chmod(kaggle_json, 0o600)
        print('Kaggle credentials configured.')

def download_dataset():
    data_dir = './data/lits-png'
    check_path = os.path.join(data_dir, 'dataset_6')
    if os.path.isdir(check_path):
        print(f'Dataset already at {data_dir}')
        return
    print('Downloading LiTS dataset from Kaggle (~3.5GB)...')
    setup_kaggle()
    run('pip install -q kaggle', 'Install kaggle CLI')
    os.makedirs('./data', exist_ok=True)
    run('kaggle datasets download -d andrewmvd/lits-png -p ./data --unzip', 'Download + extract LiTS-PNG')
    # kaggle --unzip extracts to ./data/ directly; move into lits-png subfolder if needed
    if not os.path.isdir(data_dir) and os.path.isdir('./data/dataset_6'):
        os.makedirs(data_dir, exist_ok=True)
        import shutil
        for item in os.listdir('./data'):
            src = os.path.join('./data', item)
            if item != 'lits-png' and not item.endswith('.zip'):
                dst = os.path.join(data_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    print('Dataset ready.')

def main():
    try:
        import torch
        if torch.cuda.is_available():
            g = torch.cuda.get_device_properties(0)
            print(f'GPU: {torch.cuda.get_device_name(0)} ({g.total_mem/1e9:.1f}GB)')
        else:
            print('WARNING: No GPU detected!')
    except: pass

    t_total = time.time()
    times = {}

    # Step 0: Get data
    download_dataset()

    # Step 1: Stage 1 - Liver segmentation
    if not os.path.exists('./checkpoints/stage1_best.pt'):
        times['stage1'] = run('python train_stage1.py', 'Stage 1: Liver segmentation (20 epochs)')
    else:
        print('[skip] Stage 1 checkpoint exists')

    # Step 2: Generate crops manifest
    manifest = './crops_manifest_new.json'
    if os.path.exists('./checkpoints/stage1_best.pt') and not os.path.exists(manifest):
        times['crops'] = run(
            f'python preprocess.py --mode crops --stage1-ckpt ./checkpoints/stage1_best.pt --manifest-path {manifest}',
            'Generate crops manifest')
    elif not os.path.exists(manifest):
        manifest = './crops_manifest.json'

    # Step 3: Stage 2 - Tumor segmentation (Config D)
    if not os.path.exists('./checkpoints/D_full/best.pt'):
        times['stage2'] = run(
            f'python train_stage2.py --ablation D_full --crops-manifest {manifest}',
            'Stage 2: Tumor segmentation D_full (20 epochs)')
    else:
        print('[skip] Stage 2 checkpoint exists')

    # Step 4: Evaluate
    times['eval'] = run(
        f'python evaluate.py --crops-manifest {manifest} --ablations D_full',
        'Evaluate on test set')

    # Step 5: Visualize
    times['viz'] = run(
        f'python visualize.py --crops-manifest {manifest}',
        'Generate overlay figures')

    total = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'  ALL DONE in {total/60:.1f} min')
    print(f'{"="*60}')
    for k, v in times.items():
        print(f'  {k}: {v/60:.1f} min')
    print(f'\n  Results -> results/headline_table.csv')
    print(f'  Figures -> results/figures/*.png')

if __name__ == '__main__':
    main()
