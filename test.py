#!/usr/bin/env python3
"""Test Script that computes R2 Score for ContextFormer
"""
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torchmetrics

from earthnet_models_pytorch.data import DATASETS
from earthnet_models_pytorch.task import SpatioTemporalTask
from earthnet_models_pytorch.utils import parse_setting
# Import your model (ContextFormer) from your local file
from my_contextformer import ContextFormer


class SpatioTemporalTaskWithR2(SpatioTemporalTask):
    """
    Subclass of SpatioTemporalTask that computes/logs R2 in test loop.
    """

    def __init__(self, model, hparams):
        super().__init__(model=model, hparams=hparams)
        # R2 metric from torchmetrics. 
        # If you have multiple outputs, adjust `num_outputs` or use multioutput='uniform_average'.
        self.r2_metric = torchmetrics.R2Score(num_outputs=1)


    def test_step(self, batch, batch_idx):
        # Instead of out = super()...
        preds = self.model(batch)
        target = batch["dynamic"][0][:, some_range, ...]  # or however you get it
        self.r2_metric.update(preds, target)
        return []



    def test_epoch_end(self, outputs):
        """
        At the end of the test epoch, compute R2 over all test steps.
        Then log or print it. 
        """
        # Call the parent's method to finalize other logs (like RMSE).
        super().test_epoch_end(outputs)

        # Now compute & log R2
        r2_value = self.r2_metric.compute()
        self.log("test_r2", r2_value, prog_bar=True)
        print(f"[test] R2 Score: {float(r2_value):.4f}")

        # Reset the metric if testing on multiple subsets
        self.r2_metric.reset()


def test_model(setting_dict: dict, checkpoint: str):
    # ------------------- 1) Data -------------------
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

    # ------------------- 2) Model -------------------
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = ContextFormer.add_model_specific_args(model_parser)
    model_params = model_parser.parse_args(model_args)
    model = ContextFormer(model_params)

    # ------------------- 3) Task (with R2) -------------------
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = SpatioTemporalTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)

    # Use SpatioTemporalTaskWithR2 instead of the default SpatioTemporalTask
    task = SpatioTemporalTaskWithR2(model=model, hparams=task_params)

    # If there's a checkpoint, load it
    if checkpoint != "None":
        task.load_from_checkpoint(
            checkpoint_path=checkpoint,
            context_length=setting_dict["Task"]["context_length"],
            target_length=setting_dict["Task"]["target_length"],
            model=model,
            hparams=task_params,
        )

    # ------------------- 4) Trainer -------------------
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["logger"] = False
    trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=10)], **trainer_dict)

    # ------------------- 5) Test -------------------
    dm.setup("test")
    trainer.test(model=task, datamodule=dm, ckpt_path=None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "setting",
        type=str,
        metavar="path/to/setting.yaml",
        help="yaml with all settings",
    )
    parser.add_argument(
        "checkpoint", type=str, metavar="path/to/checkpoint", help="checkpoint file"
    )
    parser.add_argument(
        "--track",
        type=str,
        metavar="iid|ood|ex|sea",
        default="ood-t_chopped",
        help="which track to test: either iid, ood, ex or sea",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="preds/",
        metavar="path/to/prediction/dir",
        help="Path where to save predictions",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/greenearthnet/",
        metavar="path/to/dataset",
        help="Path where dataset is located",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        metavar="n gpus",
        default=1,
        help="how many gpus to use",
    )
    args = parser.parse_args()

    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = parse_setting(args.setting, track=args.track)

    if args.pred_dir is not None:
        setting_dict["Task"]["pred_dir"] = args.pred_dir

    if args.data_dir is not None:
        setting_dict["Data"]["base_dir"] = args.data_dir

    if "gpus" in setting_dict["Trainer"]:
        setting_dict["Trainer"]["gpus"] = args.gpus

    # Run test with R2
    test_model(setting_dict, args.checkpoint)
