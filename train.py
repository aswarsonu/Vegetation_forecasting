import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
#!/usr/bin/env python3
"""Train Script
"""
import os, sys
# wherever you cloned DiT:
sys.path.insert(0, "/storage/sonu/endgame/DiT")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import yaml
from earthnet_models_pytorch.data import DATASETS
# from earthnet_models_pytorch.model import MODELS
from my_contextformer import ContextFormer
from earthnet_models_pytorch.task import SpatioTemporalTask
from earthnet_models_pytorch.utils import parse_setting
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel as DDP
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TOKENIZERS_PARALLELISM="false"
find_unused_parameters=True 





def train_model(setting_dict: dict, setting_file: str = None):
    start = time.time()

    pl.seed_everything(setting_dict["Seed"])
    # Data
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

    # Model
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    # model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
    #     model_parser
    # )

    model_parser = ContextFormer.add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    # model = MODELS[setting_dict["Architecture"]](model_params)

    model = ContextFormer(model_params)

    print("model is", ContextFormer)

    # Task
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = SpatioTemporalTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = SpatioTemporalTask(model=model, hparams=task_params)

    # Logger
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    if (
        setting_file is not None
        and type(logger.experiment).__name__ != "DummyExperiment"
    ):
        print("Copying setting yaml.")
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(os.path.join(logger.log_dir, "setting.yaml"), "w") as fp:
            yaml.dump(setting_dict, fp)

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**setting_dict["Checkpointer"])

    # Trainer
    trainer_dict = setting_dict["Trainer"]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        **trainer_dict,
    )

    trainer.fit(task, dm)

    print(
        f"Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}"
    )

    end = time.time()

    print(f"Calculation done in {end - start} seconds.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "setting",
        type=str,
        metavar="path/to/setting",
        help="yaml with all settings",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/greenearthnet/",
        metavar="path/to/dataset",
        # metavar="/storage/ij_vatsal/data_dir/greenearthnet",
        help="Path where dataset is located",
    )
    args = parser.parse_args()

    # Disabling PyTorch Lightning automatic SLURM detection
    # for k, v in os.environ.items():
    #     if k.startswith("SLURM"):
    #         del os.environ[k]

    setting_dict = parse_setting(args.setting)

    if args.data_dir is not None:
        setting_dict["Data"]["base_dir"] = args.data_dir

    train_model(setting_dict, args.setting)
