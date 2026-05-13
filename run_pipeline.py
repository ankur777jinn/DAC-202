#!/usr/bin/env python3
"""ONE-SHOT: Download data + Train Stage1 + Crops + Train Stage2 + Evaluate + Visualize."""
import os, sys, time, subprocess, zipfile

def run(cmd, desc):
    print(f'\n{"="*60}\n  {desc}\n  $ {cmd}\n{"="*60}')
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    status = 'DONE' if ret.returncode == 0 else 'FAILED'
    print(f'--- {status} ({dt:.0f}s) ---')
    if ret.returncode != 0:
        sys.exit(ret.returncode)
    return dt

def download_dataset():
    data_dir = './data/lits-png'
    if os.path.isdir(os.path.join(data_dir, 'dataset_6')):
        print(f'Dataset already exists at {data_dir}')
        return
    print('Downloading LiTS dataset from Kaggle...')
    os.makedirs('./data', exist_ok=True)
    run('pip install -q kaggle', 'Install kaggle CLI')
    run('kaggle datasets download -d andrewmvd/lits-png -p ./data', 'Download LiTS-PNG from Kaggle')
    zip_path = './data/lits-png.zip'
    if os.path.exists(zip_path):
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('./data/lits-png')
        os.remove(zip_path)
        print('Extraction done.')

def main():
    import torch
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        print(f'GPU: {torch.cuda.get_device_name(0)} ({g.total_mem/1e9:.1f}GB)')
    else:
        print('WARNING: No GPU!')

    t_total = time.time()
    times = {}

    # Step 0: Get data
    download_dataset()

    # Step 1: Stage 1
    if not os.path.exists('./checkpoints/stage1_best.pt'):
        times['stage1'] = run('python train_stage1.py', 'Stage 1: Liver segmentation (20 epochs)')
    else:
        print('[skip] Stage 1 checkpoint exists')

    # Step 2: Crops
    manifest = './crops_manifest_new.json'
    if os.path.exists('./checkpoints/stage1_best.pt') and not os.path.exists(manifest):
        times['crops'] = run(
            f'python preprocess.py --mode crops --stage1-ckpt ./checkpoints/stage1_best.pt --manifest-path {manifest}',
            'Generate crops manifest')
    elif not os.path.exists(manifest):
        manifest = './crops_manifest.json'
        print(f'Using existing manifest: {manifest}')

    # Step 3: Stage 2 D_full
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
    print(f'\n{"="*60}\n  ALL DONE in {total/60:.1f} min\n{"="*60}')
    for k, v in times.items():
        print(f'  {k}: {v/60:.1f} min')
    print('\nResults:')
    print('  results/headline_table.csv')
    print('  results/stratified.csv')
    print('  results/figures/*.png')

if __name__ == '__main__':
    main()
