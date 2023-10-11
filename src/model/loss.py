import torch
from util import util


class AlphaLossNV2(torch.nn.Module):
    """
    Implement Neural Volumes alpha loss 2
    """

    def __init__(self, lambda_alpha, clamp_alpha, init_epoch, force_opaque=False):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.clamp_alpha = clamp_alpha
        self.init_epoch = init_epoch
        self.force_opaque = force_opaque
        if force_opaque:
            self.bceloss = torch.nn.BCELoss()
        self.register_buffer(
            "epoch", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sched_step(self, num=1):
        self.epoch += num

    def forward(self, alpha_fine):
        if self.lambda_alpha > 0.0 and self.epoch.item() >= self.init_epoch:
            alpha_fine = torch.clamp(alpha_fine, 0.01, 0.99)
            if self.force_opaque:
                alpha_loss = self.lambda_alpha * self.bceloss(
                    alpha_fine, torch.ones_like(alpha_fine)
                )
            else:
                alpha_loss = torch.log(alpha_fine) + torch.log(1.0 - alpha_fine)
                alpha_loss = torch.clamp_min(alpha_loss, -self.clamp_alpha)
                alpha_loss = self.lambda_alpha * alpha_loss.mean()
        else:
            alpha_loss = torch.zeros(1, device=alpha_fine.device)
        return alpha_loss


def get_alpha_loss(conf):
    lambda_alpha = conf.get_float("lambda_alpha")
    clamp_alpha = conf.get_float("clamp_alpha")
    init_epoch = conf.get_int("init_epoch")
    force_opaque = conf.get_bool("force_opaque", False)

    return AlphaLossNV2(
        lambda_alpha, clamp_alpha, init_epoch, force_opaque=force_opaque
    )


class RGBWithUncertainty(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, betas):
        """computes the error per output, weights each element by the log variance
        outputs is B x 3, targets is B x 3, betas is B"""
        weighted_element_err = (
                torch.mean(self.element_loss(outputs, targets), -1) / betas
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(betas))


class RGBWithBackground(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, lambda_bg):
        """If we're using background, then the color is color_fg + lambda_bg * color_bg.
        We want to weight the background rays less, while not putting all alpha on bg"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / (
                1 + lambda_bg
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(lambda_bg))


def get_rgb_loss(conf, coarse=True, using_bg=False, reduction="mean"):
    if conf.get_bool("use_uncertainty", False) and not coarse:
        print("using loss with uncertainty")
        return RGBWithUncertainty(conf)
    #     if using_bg:
    #         print("using loss with background")
    #         return RGBWithBackground(conf)
    print("using vanilla rgb loss")
    return (
        torch.nn.L1Loss(reduction=reduction)
        if conf.get_bool("use_l1")
        else torch.nn.MSELoss(reduction=reduction)
    )


class YoloLoss(torch.nn.Module):
    def __init__(self, num_anchors_per_scale, box_loss, object_loss, no_object_loss, class_loss):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.num_anchors_per_scale = num_anchors_per_scale

        self.box_loss = box_loss
        self.object_loss = object_loss
        self.no_object_loss = no_object_loss
        self.class_loss = class_loss

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 1, self.num_anchors_per_scale, 2)
        # Box prediction confidence
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]),
                               torch.exp(pred[..., 3:5]) * anchors
                               ], dim=-1)
        # Calculating intersection over union for prediction and target
        ious = util.iou(box_preds[obj], target[..., 1:5][obj]).detach()
        # Calculating Object loss
        object_loss = self.mse(pred[..., 0:1][obj],
                               ious * target[..., 0:1][obj]) if obj.sum().item() > 0 else torch.tensor(0)

        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss
        box_loss = self.mse(pred[..., 1:5][obj],
                            target[..., 1:5][obj]) if obj.sum().item() > 0 else torch.tensor(0)

        # Calculating class loss
        class_loss = self.cross_entropy((pred[..., 5:][obj]),
                                        target[..., 5][obj].long()) if obj.sum().item() > 0 else torch.tensor(0)

        # Total loss
        return (
            box_loss * self.box_loss
            + object_loss * self.object_loss
            + no_object_loss * self.no_object_loss
            + class_loss * self.class_loss,
            box_loss, object_loss, no_object_loss, class_loss
        )

    @classmethod
    def from_conf(cls, conf, num_anchors_per_scale):
        print("using weights for yolo loss")
        print("box_loss", conf["yolo.weights.box_loss"])
        print("object_loss", conf["yolo.weights.object_loss"])
        print("no_object_loss", conf["yolo.weights.no_object_loss"])
        print("class_loss", conf["yolo.weights.class_loss"])

        return cls(
            num_anchors_per_scale,
            conf["yolo.weights.box_loss"],
            conf["yolo.weights.object_loss"],
            conf["yolo.weights.no_object_loss"],
            conf["yolo.weights.class_loss"],
        )
