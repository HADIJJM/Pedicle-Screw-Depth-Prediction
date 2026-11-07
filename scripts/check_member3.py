"""Check Member 2 DataLoader -> Member 3 UNet/loss; no optimizer training."""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import monai
from src.module1_segmentation.data.dataset import build_dataloaders, _mmap_arrays
from src.module1_segmentation.data.preprocess import CLASS_TO_LEVEL
from src.module1_segmentation.model.unet3d import VertebraUNet, LABEL_MAP
from src.module1_segmentation.model.losses import SegmentationLoss


def synthetic_manifests(root):
    """Synthetic cached volumes: same on-disk interface; NOT medical data."""
    paths = []
    for split, filename in [('training', 'train.csv'), ('validation', 'val.csv')]:
        folder = root / split
        folder.mkdir()
        mask = np.zeros((40, 48, 56), dtype=np.uint8)
        centers = []
        for k in range(8):
            z, y, x = 8+(k&1)*20, 8+((k>>1)&1)*24, 8+((k>>2)&1)*32
            mask[z:z+4, y:y+4, x:x+4] = k+1
            centers.append([k+1, z+1, y+1, x+1])
        np.save(folder/'image.npy', (mask/8).astype(np.float16))
        np.save(folder/'mask.npy', mask)
        np.save(folder/'centers.npy', np.array(centers, dtype=np.int32))
        meta = {'config_fingerprint': 'synthetic-m3-only', 'dataset_split': split,
                'processed_shape_zyx': list(mask.shape)}
        (folder/'metadata.json').write_text(json.dumps(meta))
        row = {'dataset_split': split, 'patient_id': 'synthetic-'+split,
               'scan_id': 'synthetic-'+split, 'config_fingerprint': 'synthetic-m3-only',
               'image_path': str(folder/'image.npy'), 'mask_path': str(folder/'mask.npy'),
               'centers_path': str(folder/'centers.npy'), 'metadata_path': str(folder/'metadata.json')}
        path = root/filename
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true', help='Manufactured arrays, no real CT verification')
    parser.add_argument('--manifest-dir', type=Path, default=ROOT/'outputs/m2_preprocessing')
    parser.add_argument('--patch-size', nargs=3, type=int, default=(96,96,96))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    for i in range(9):
        assert CLASS_TO_LEVEL[i] == {v:k for k,v in LABEL_MAP.items()}[i], 'Member 2/3 label disagreement'
    device = torch.device(args.device)
    model, criterion = VertebraUNet().to(device), SegmentationLoss().to(device)
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        if args.synthetic:
            train_path, val_path = synthetic_manifests(Path(tmp))
        else:
            train_path, val_path = args.manifest_dir/'train.csv', args.manifest_dir/'val.csv'
        try:
            train, val, train_ds, val_ds = build_dataloaders(
                train_path, val_path, patch_size_zyx=tuple(args.patch_size),
                batch_size=args.batch_size, num_workers=0, patches_per_case=2)
            train_ds.set_epoch(0)
            for name, loader in [('train', train), ('val', val)]:
                batch = next(iter(loader))
                images, masks = batch['image'].to(device), batch['mask'].to(device)
                assert images.dtype == torch.float32 and masks.dtype == torch.int64
                assert torch.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
                assert 0 <= masks.min() <= masks.max() <= 8
                model.train(name == 'train')
                with torch.set_grad_enabled(name == 'train'):
                    logits = model(images)
                    assert logits.shape == (images.shape[0],9,*images.shape[2:])
                    loss = criterion(logits, masks)
                    assert torch.isfinite(loss)
                    if name == 'train':
                        loss.backward()
                        grads = [p.grad for p in model.parameters() if p.requires_grad]
                        assert all(g is not None and torch.isfinite(g).all() for g in grads)
                        assert any(g.abs().sum() > 0 for g in grads)
                        model.zero_grad(set_to_none=True)
                reports.append({'partition': name, 'image_shape': list(images.shape),
                                'mask_shape': list(masks.shape), 'output_shape': list(logits.shape),
                                'classes_in_batch': masks.unique().tolist(), 'loss': loss.item(),
                                'backward_checked': name == 'train'})
        finally:
            _mmap_arrays.cache_clear()  # Release Windows mmap file handles before cleanup.
    report = {'status':'PASS', 'data_kind':'synthetic' if args.synthetic else 'real_prepared',
              'device':str(device), 'patch_size':args.patch_size, 'torch':torch.__version__,
              'monai':monai.__version__, 'batches':reports,
              'limitation':'One train and one val batch only; no optimizer updates, accuracy, or full-volume geometry validation.'}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
