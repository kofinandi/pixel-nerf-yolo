import torch

class YoloRenderer(torch.nn.Module):
    def __init__(self, n_coarse, eval_batch_size):
        super().__init__()
        self.net = None
        self.n_coarse = n_coarse
        self.eval_batch_size = eval_batch_size

    def bind_net(self, net):
        self.net = net

    def sample_coarse(self, ray_batch):
        """Sample coarse points on rays"""
        device = ray_batch.device
        near, far = ray_batch[:, -2:-1], ray_batch[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = ray_batch.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step

        return near * (1 - z_steps) + far * z_steps # this is basically just near + z_steps * (far - near)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("eval_batch_size", 1024),
        )

    def forward(self, rays):
        z_samp = self.sample_coarse(rays)

        B, K = z_samp.shape

        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]  # (B, K, 3)
        points = points.reshape(-1, 3)  # (B*K, 3)

        split_points = torch.split(points, self.eval_batch_size, dim=0)

        viewdirs = rays[:, None, 3:6].expand(-1, K, -1)  # (B, K, 3)
        viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)

        split_viewdirs = torch.split(viewdirs, self.eval_batch_size, dim=0)

        val_all = []

        for pnts, dirs in zip(split_points, split_viewdirs):
            val_all.append(self.net(pnts, coarse=True, viewdirs=dirs))

        out = torch.cat(val_all, dim=0)
        out = out.reshape(B, K, -1)  # (B, K, 7)

        # TODO: maybe this needs a different activation function?
        probabilities = torch.sigmoid(out[..., 0])  # (B, K)

        # TODO: should we divide by K?
        final_values = torch.sum(out[..., 1:] * probabilities.unsqueeze(-1), dim=1) / K  # (B, 6)
        final_probabilities = torch.sum(probabilities, dim=1) / K  # (B)

        return final_values, final_probabilities

    def bind_parallel(self, net, gpus = None):
        self.net = net
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            return torch.nn.DataParallel(self, gpus, dim=1)

        return self