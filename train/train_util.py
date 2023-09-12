from trainlib import PixelNeRFTrainer


def make_trainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device):
    trainer_type = conf.get_string("renderer.type", "nerf")
    if trainer_type == "nerf":
        return PixelNeRFTrainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device)
    elif trainer_type == "yolo":
        return
    else:
        raise NotImplementedError("Unsupported trainer type")
