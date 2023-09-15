from trainlib import PixelNeRFTrainer
from trainlib import YOLOTrainer


def make_trainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device):
    trainer_type = conf.get_string("renderer.type", "nerf")
    if trainer_type == "nerf":
        return PixelNeRFTrainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device)
    elif trainer_type == "yolo":
        return YOLOTrainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device)
    else:
        raise NotImplementedError("Unsupported trainer type")
