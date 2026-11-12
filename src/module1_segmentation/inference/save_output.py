"""Member 5: save prediction as NIfTI in processed and original spaces."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.processing


def _build_processed_nifti(mask_zyx, metadata):
    """Construct a NIfTI image from the processed-grid mask."""
    processed_affine = np.array(metadata["processed_affine_xyz"], dtype=np.float64)
    # internal ZYX -> NIfTI XYZ
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(np.uint8)
    nii = nib.Nifti1Image(mask_xyz, affine=processed_affine)
    nii.set_data_dtype(np.uint8)
    return nii


def save_mask_two_spaces(mask_zyx, metadata, output_original, output_processed=None):
    """Write mask in original-space (required) and processed-space (optional)."""
    nii_processed = _build_processed_nifti(mask_zyx, metadata)

    # processed-space output (optional, for Member 6)
    if output_processed is not None:
        out_p = Path(output_processed)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nii_processed, str(out_p))
        print(f"[save] processed-space mask -> {out_p}")

    # resample to the raw CT grid (handles LAS/RAS orientation flip)
    raw_shape_xyz = tuple(metadata["raw_shape_xyz"])
    raw_affine = np.array(metadata["raw_affine_xyz"], dtype=np.float64)
    nii_original = nib.processing.resample_from_to(
        nii_processed,
        (raw_shape_xyz, raw_affine),
        order=0,          # nearest-neighbor for integer labels
        mode="constant",
        cval=0,           # anything outside becomes background
    )
    # resample_from_to returns float by default; cast back to uint8
    data = np.asarray(nii_original.dataobj).astype(np.uint8)
    nii_original = nib.Nifti1Image(data, affine=nii_original.affine)
    nii_original.set_data_dtype(np.uint8)

    out_o = Path(output_original)
    out_o.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_original, str(out_o))
    print(f"[save] original-space mask  -> {out_o}")

    return out_o


def verify_alignment(mask_path, reference_ct_path):
    """Confirm the saved mask truly aligns with the raw CT."""
    ct = nib.load(str(reference_ct_path))
    mk = nib.load(str(mask_path))
    if ct.shape != mk.shape:
        raise AssertionError(f"Shape mismatch: CT={ct.shape}, mask={mk.shape}")
    if not np.allclose(ct.affine, mk.affine, atol=1e-4):
        raise AssertionError("Affine mismatch — mask will not overlay correctly")
    print(f"[verify] aligned: shape={mk.shape}, spacing={mk.header.get_zooms()[:3]}")