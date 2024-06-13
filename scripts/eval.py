import os
from pathlib import Path

import logging
from pyexpat import model
import pytorch_lightning as pl
import torch
import hydra
from torchmetrics import MetricCollection
from hydra.utils import instantiate


from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, remove_prefix, setup_network
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback
from cross_view_transformer.tabular_logger import TabularLogger


log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'



def load_weights(cfg, checkpoint_path: str, prefix: str = 'backbone'):
    checkpoint = torch.load(checkpoint_path)

    state_dict = remove_prefix(checkpoint['state_dict'], prefix)

    backbone = setup_network(cfg)
    backbone.load_state_dict(state_dict)

    return backbone

def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(
        f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)
    ckpt_path = cfg.experiment.ckptt

    print("Loading weights from:", ckpt_path)
    assert os.path.exists(ckpt_path), "evaluation requires ckptt"
    if ckpt_path is not None:
        model_module.backbone = load_weights(cfg, ckpt_path)

    # Loggers and callbacks

    tab_logger = TabularLogger(save_dir=cfg.experiment.save_dir)
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
    ]

    # Train
    trainer = pl.Trainer(logger=[tab_logger],
                         callbacks=callbacks,
                         enable_progress_bar=False,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    model_module.eval()
    # mcw
    # print(model_module)
    # exit()
    export=False
    ort=True
    if export:
        from onnxruntime.tools import pytorch_export_contrib_ops
        with torch.no_grad():
            for i, batch in enumerate(data_module.val_dataloader()):
                pytorch_export_contrib_ops.register()
                torch.onnx.export(
                    model_module,                     
                    (batch,batch),               
                    "/media/ava/DATA3/DATA/raj/GKT/segmentation/pretrained_models/gkt.onnx",              
                    verbose=True,
                    export_params=True,         
                    opset_version=16,
                )
                break
            import onnx
            from onnxsim import simplify
            model = onnx.load("/media/ava/DATA3/DATA/raj/GKT/segmentation/pretrained_models/gkt.onnx")
            onnx.checker.check_model(model)
            model_simplified, check = simplify(model, check_n=3)
            assert check, 'assert check failed'
            onnx.save(model_simplified, '/media/ava/DATA3/DATA/raj/GKT/segmentation/pretrained_models/simplified_gkt.onnx')
            print("Checked",check)
            print("Export Successfully...")
            exit()
    elif ort:
        with torch.no_grad():
            import onnxruntime as ort
            from tqdm import tqdm
            print("\nOnnxRuntime Inference:\n")
            sess=ort.InferenceSession("/media/ava/DATA3/DATA/raj/GKT/segmentation/pretrained_models/simplified_gkt.onnx",providers=['CUDAExecutionProvider'])
            metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()})
            
            for i, data in enumerate(tqdm(data_module.val_dataloader(), desc="Validation Iterations")):
                inputs = {
                            'onnx::Shape_1': data['image'].cpu().numpy(),
                            'com.microsoft::Inverse_2': data['intrinsics'].cpu().numpy(),
                            'com.microsoft::Inverse_3': data['extrinsics'].cpu().numpy(),
                            
                        }
                preds=sess.run(None, inputs)
                pred = {
                        'bev': torch.Tensor(preds[0]),
                        'center': torch.Tensor(preds[1])
                        }
                outputs = {'bev':[0,1], 'center':[1,2]}
                pred = {k:pred[k][:, start:stop] for k,(start,stop) in outputs.items()}
                metrics.update(pred, data)
                
                del data, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            print(metrics.compute())
    else:
        with torch.no_grad():
            # mcw
            # trainer.validate(model_module, data_module, ckpt_path=ckpt_path)
            metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()})
            from tqdm import tqdm
            for i, data in enumerate(tqdm(data_module.val_dataloader(), desc="Validation Iterations")):
                pred = model_module(data)
                outputs = {'bev':[0,1], 'center':[1,2]}

                pred = {k:pred[k][:, start:stop] for k,(start,stop) in outputs.items()}
                metrics.update(pred, data)
                
                del data, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if i>10:
                    break

        print(metrics.compute())


if __name__ == '__main__':
    main()
