from .nerf import NeRFRenderer
from .yolo import YoloRenderer


def make_renderer(conf, lindisp=False):
    renderer_type = conf.get_string("type", "nerf")  # nerf | yolo
    if renderer_type == "nerf":
        return NeRFRenderer.from_conf(conf, lindisp=lindisp)
    elif renderer_type == "yolo":
        return YoloRenderer.from_conf(conf)
    else:
        raise NotImplementedError("Unsupported renderer type")
