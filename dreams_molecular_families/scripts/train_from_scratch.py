"""Train a smaller DreaMS model from scratch (self-supervised pre-training).

This is option 1 of "train": a new model learns from unlabelled spectra by
hiding some peaks and guessing them (DreaMS's masked m/z objective), exactly
as in the DreaMS paper, but smaller so that it fits one Colab GPU:

  - the published model: 7 layers, width 1024, 8 GPUs, about 24 hours, GeMS_A10
  - "small" here: 4 layers, width 512, 1 GPU, a few hours, 500,000 GeMS spectra

It uses DreaMS's own model and data classes (commit dbec3a0) and the settings
of .DreaMS/dreams/training/pre_train.sh, except: smaller model and batch, no
retention-order loss (our spectra have no retention times), one device, and
a time limit. The model is saved every --save-every steps; if a run stops (for
example when Colab disconnects), running the same command again continues
from the last saved model.

Usage:
  python scripts/train_from_scratch.py --data GeMS_A1_DreaMS_rand500k.hdf5 --out runs_scratch/small --hours 3
  python scripts/train_from_scratch.py --data GeMS_A1_DreaMS_rand50k.hdf5 --out /tmp/tiny --size tiny --n-samples 2000 --max-steps 30
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytorch_lightning as pl
import torch

SIZES = {
    # layers, heads, peak width, Fourier width (model width = peak + Fourier), Fourier FF width, batch
    "tiny": dict(n_layers=2, n_heads=4, d_peak=44, d_fourier=212, ff_fourier_d=128, batch_size=32),
    "small": dict(n_layers=4, n_heads=8, d_peak=44, d_fourier=468, ff_fourier_d=256, batch_size=128),
    "published": dict(n_layers=7, n_heads=8, d_peak=44, d_fourier=980, ff_fourier_d=512, batch_size=256),
}


def fix_dreams_to_classes():
    """Work around a bug in DreaMS commit dbec3a0: dreams.utils.spectra.to_hot()
    asks to_classes() for (classes, num_classes), but the line of to_classes()
    that returns num_classes is commented out, so pre-training stops with
    "ValueError: too many values to unpack". We restore that line here,
    without changing the DreaMS files."""
    import dreams.utils.spectra as su
    original = su.to_classes

    def to_classes(vals, max_val, bin_size, special_vals=(), return_num_classes=False):
        classes = original(vals, max_val, bin_size, special_vals)
        if return_num_classes:
            return classes, su.num_hot_classes(max_val, bin_size) + len(special_vals)
        return classes

    su.to_classes = to_classes


def dreams_args(data, out, size, lr, warmup, n_samples, seed):
    """The arguments of DreaMS's pre_train.sh, parsed by DreaMS's own parser."""
    s = SIZES[size]
    argv = [
        "train", "--project_name", str(out), "--job_key", "run", "--run_name", "run", "--no_wandb",
        "--seed", str(seed), "--dataset_pth", str(data), "--dformat", "A",
        "--batch_size", str(s["batch_size"]), "--val_frac", "0.01",
        "--train_regime", "pre-training", "--train_objective", "mask_mz_hot", "--frac_masks", "0.3",
        "--hot_mz_bin_size", "0.05", "--model", "DreaMS",
        "--ff_peak_depth", "1", "--ff_fourier_depth", "5", "--ff_fourier_d", str(s["ff_fourier_d"]),
        "--ff_out_depth", "1", "--prec_intens", "1.1",
        "--n_layers", str(s["n_layers"]), "--n_heads", str(s["n_heads"]),
        "--d_peak", str(s["d_peak"]), "--d_fourier", str(s["d_fourier"]),
        "--lr", str(lr), "--dropout", "0.1", "--att_dropout", "0.1", "--residual_dropout", "0.1",
        "--ff_dropout", "0.1", "--weight_decay", "0", "--attn_mech", "dot-product", "--train_precision", "32",
        "--mask_peaks", "--mask_intens_strategy", "intens_p", "--max_peaks_n", "60", "--ssl_probing_depth", "0",
        "--focal_loss_gamma", "5", "--no_transformer_bias", "--n_warmup_steps", str(warmup),
        "--fourier_strategy", "lin_float_int", "--mz_shift_aug_p", "0.2", "--mz_shift_aug_max", "50",
        "--pre_norm", "--graphormer_mz_diffs", "--ret_order_loss_w", "0",
    ]
    if n_samples:
        argv += ["--n_samples", str(n_samples)]
    from dreams.training.train_argparse import parse_args
    old, sys.argv = sys.argv, argv
    try:
        return parse_args()
    finally:
        sys.argv = old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="GeMS .hdf5 file (spectrum and precursor_mz)")
    parser.add_argument("--out", required=True, type=Path, help="folder for checkpoints and the loss log")
    parser.add_argument("--size", default="small", choices=list(SIZES))
    parser.add_argument("--hours", type=float, default=3.0, help="stop after this many hours (per session)")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps in total")
    parser.add_argument("--n-samples", type=int, default=None, help="use only the first N spectra (quick tests)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2000, help="learning-rate warm-up steps")
    parser.add_argument("--save-every", type=int, default=500, help="save the model every N steps")
    parser.add_argument("--seed", type=int, default=3402)
    args = parser.parse_args()

    import dreams.utils.data as du
    from dreams.models.dreams.dreams import DreaMS

    fix_dreams_to_classes()
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    a = dreams_args(args.data, args.out, args.size, args.lr, args.warmup, args.n_samples, args.seed)
    a.gains_dir = args.out / "model_gains"
    a.gains_dir.mkdir(exist_ok=True)

    spec_preproc = du.SpectrumPreprocessor(
        dformat=a.dformat, prec_intens=a.prec_intens, n_highest_peaks=a.max_peaks_n,
        spec_entropy_cleaning=a.spec_entropy_cleaning, precision=a.train_precision,
        mz_shift_aug_p=a.mz_shift_aug_p, mz_shift_aug_max=a.mz_shift_aug_max,
    )
    dataset = du.MaskedSpectraDataset(
        in_pth=a.dataset_pth, spec_preproc=spec_preproc, n_samples=a.n_samples, dformat=a.dformat,
        ssl_objective=a.train_objective, deterministic_mask=a.deterministic_mask, frac_masks=a.frac_masks,
        mask_prec=a.mask_prec, mask_peaks=a.mask_peaks, mask_intens_strategy=a.mask_intens_strategy,
        ret_order_pairs=False, acc_est_weight=a.acc_est_weight, lsh_weight=a.lsh_weight,
        mask_val=a.mask_val, bert801010_masking=a.bert801010_masking,
    )
    data_module = du.RandomSplitDataModule(dataset=dataset, batch_size=a.batch_size, val_frac=a.val_frac)
    model = DreaMS(a, spec_preproc)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{len(dataset):,} spectra, model '{args.size}' with {n_params / 1e6:.1f} M parameters, "
          f"device: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    trainer = pl.Trainer(
        accelerator="auto", devices=1, max_epochs=1000, max_steps=args.max_steps,
        max_time=timedelta(hours=args.hours), precision=a.train_precision, num_sanity_val_steps=0,
        logger=pl.loggers.CSVLogger(args.out, name="log"), log_every_n_steps=20,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.out, every_n_train_steps=args.save_every,
                                                save_top_k=0, save_last=True)],
    )
    last = args.out / "last.ckpt"
    start = time.time()
    trainer.fit(model, datamodule=data_module, ckpt_path=str(last) if last.exists() else None)
    trainer.save_checkpoint(args.out / "final.ckpt")

    info_path = args.out / "training_info.json"
    sessions = json.loads(info_path.read_text())["sessions"] if info_path.exists() else []
    sessions.append({"ended": datetime.now().isoformat(timespec="seconds"),
                     "seconds": round(time.time() - start), "global_step": trainer.global_step,
                     "epoch": trainer.current_epoch})
    info_path.write_text(json.dumps({
        "data": str(args.data), "spectra": len(dataset), "size": args.size, **SIZES[args.size],
        "parameters": n_params, "lr": args.lr, "warmup_steps": args.warmup,
        "dreams_commit": "dbec3a0b514a99e5056cfccde4559fda8cfe8129", "sessions": sessions,
    }, indent=2))
    print(f"Stopped at step {trainer.global_step} (epoch {trainer.current_epoch}). "
          f"Saved {args.out / 'final.ckpt'}")


if __name__ == "__main__":
    main()
