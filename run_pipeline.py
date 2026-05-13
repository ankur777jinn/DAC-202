#!/usr/bin/env python3
"""ONE-SHOT: Train Stage1 -> Crops -> Train Stage2 (D_full) -> Evaluate -> Visualize."""
import os, sys, time, subprocess

def run(cmd, desc):
    print(f'\n{"="*60}\n  {desc}\n  $ {cmd}\n{"="*60}')
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    print(f'--- {"DONE" if ret.returncode==0 else "FAILED"} ({dt:.0f}s) ---')
    if ret.returncode != 0: sys.exit(ret.returncode)
    return dt

def main():
    import torch
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)')
    else:
        print('WARNING: No GPU!')

    times = {}
    t0 = time.time()

    # Stage 1
    if not os.path.exists('./checkpoints/stage1_best.pt'):
        times['stage1'] = run('python train_stage1.py', 'Stage 1: Liver segmentation (20 epochs)')
    else:
        print('Stage 1 checkpoint exists, skipping.')

    # Crops manifest
    manifest = './crops_manifest_new.json'
    if os.path.exists('./checkpoints/stage1_best.pt') and not os.path.exists(manifest):
        times['crops'] = run(f'python preprocess.py --mode crops --stage1-ckpt ./checkpoints/stage1_best.pt --manifest-path {manifest}', 'Generate crops manifest')
    elif not os.path.exists(manifest):
        manifest = './crops_manifest.json'
        print(f'Using existing manifest: {manifest}')

    # Stage 2 - Config D
    if not os.path.exists('./checkpoints/D_full/best.pt'):
        times['stage2'] = run(f'python train_stage2.py --ablation D_full --crops-manifest {manifest}', 'Stage 2: Tumor segmentation D_full (20 epochs)')
    else:
        print('Stage 2 D_full checkpoint exists, skipping.')

    # Evaluate
    times['eval'] = run(f'python evaluate.py --crops-manifest {manifest} --ablations D_full', 'Evaluate on test set')

    # Visualize
    times['viz'] = run(f'python visualize.py --crops-manifest {manifest}', 'Generate overlay figures')

    total = time.time() - t0
    print(f'\n{"="*60}\n  DONE in {total/60:.1f} min\n{"="*60}')
    for k,v in times.items(): print(f'  {k}: {v/60:.1f} min')
    print(f'\nOutputs in: results/')

if __name__ == '__main__':
    main()
