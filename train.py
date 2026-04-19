import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import yaml
import pytorch_lightning as ptl
import glob
import shutil
from Model import model
from Dataloader import dataloader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

def main(config_name):
    config_path = "./Config/"
    with open(config_path + config_name, 'r') as f:
        cfg = yaml.safe_load(f)
    config_path = config_path + config_name
    torch.set_num_threads(1)
    save_path = cfg["SavePath"]
    name = cfg["ExpName"]
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + "/" + name, exist_ok=True)
    logger = loggers.TensorBoardLogger(
        save_path, name=name, default_hp_metric=False)

    ckpt_dir = logger.log_dir + '/model_checkpoint'
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=cfg["trainer"]["monitor"],
        mode=cfg["trainer"]["monitor_mode"],
        dirpath=ckpt_dir
    )

    shutil.copyfile(config_path, save_path + "/" + name + "/config_use.yaml")

    mixed_encoder_param = cfg["Encoder"]
    noisy_encoder_param = cfg["EncoderNoisy"]
    inject_layer_param = cfg["InjectLayer"]
    decoder_param = cfg["Decoder"]
    time_encoder_param = cfg["TimeEncoder"]
    diffusion_param = cfg["Diffusion"]
    optimizer_param = cfg["Optimizer"]
    loss_param = cfg["Loss"]
    path_database = cfg["path_database"]

    modelptl = model.ModelPTL(
        mixed_encoder_param,
        noisy_encoder_param,
        inject_layer_param,
        decoder_param,
        time_encoder_param,
        diffusion_param,
        optimizer_param,
        loss_param,
        path_database,
        path_of_save_folder=cfg["SavePath"] + "/" + name
    )

    data_config = cfg["Dataloader"]

    datamodule = dataloader.DataModule(**data_config)
    trainer = ptl.Trainer(
        accelerator = cfg["trainer"]["accelerator"],
        devices = cfg["trainer"]["devices"],
        max_epochs = cfg["trainer"]["max_epochs"],
        logger = logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        modelptl,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader())

    trainer.save_checkpoint(ckpt_dir + '/last.ckpt')

    best_model = model.ModelPTL.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        encoder_param_mixed=mixed_encoder_param,
        encoder_param_noisy=noisy_encoder_param,
        inject_layer_param=inject_layer_param,
        decoder_param=decoder_param,
        time_encoder_param=time_encoder_param,
        diffusion_param=diffusion_param,
        optimizer_params=optimizer_param,
        loss_param=loss_param,
        path_database=path_database,
        path_of_save_folder=cfg["SavePath"] + "/" + name)

    modelptl.model = best_model.model

    trainer.test(dataloaders=datamodule.test_dataloader())
    trainer.save_checkpoint(ckpt_dir + '/best.ckpt')

if __name__ == "__main__":
    main("config_with_guidance.yaml")
    main("config_without_guidance.yaml")
    main("config_confidence_estimator.yaml")
