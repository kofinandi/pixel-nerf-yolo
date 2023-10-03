import torch

class YoloRenderer(torch.nn.Module):
    def __init__(self, n_coarse, eval_batch_size, num_scales, num_anchors_per_scale):
        super().__init__()
        self.net = None
        self.n_coarse = n_coarse
        self.eval_batch_size = eval_batch_size
        self.num_scales = num_scales
        self.num_anchors_per_scale = num_anchors_per_scale

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
            conf.get_int("renderer.n_coarse", 128),
            conf.get_int("renderer.eval_batch_size", 1024),
            conf.get_int("model.mlp_coarse.num_scales", 1),
            conf.get_int("model.mlp_coarse.num_anchors_per_scale", 3),
        )

    def forward(self, rays):
        rays = rays.reshape(-1, 8)  # (SB * B, 8)

        # print if any of the rays are nan
        if torch.isnan(rays).any():
            print("rays contains nan")

        # print if any of the rays are inf
        if torch.isinf(rays).any():
            print("rays contains inf")

        z_samp = self.sample_coarse(rays)

        # print if any of the z_samp are nan
        if torch.isnan(z_samp).any():
            print("z_samp contains nan")

        # print if any of the z_samp are inf
        if torch.isinf(z_samp).any():
            print("z_samp contains inf")

        B, K = z_samp.shape

        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]  # (B, K, 3)
        points = points.reshape(1, -1, 3)  # (1, B*K, 3)

        split_points = torch.split(points, self.eval_batch_size, dim=1)

        viewdirs = rays[:, None, 3:6].expand(-1, K, -1)  # (B, K, 3)
        viewdirs = viewdirs.reshape(1, -1, 3)  # (1, B*K, 3)

        split_viewdirs = torch.split(viewdirs, self.eval_batch_size, dim=1)

        val_all = []

        has_nan = any(torch.isnan(p).any() for p in self.net.parameters())
        if has_nan:
            print("model parameters contain nan")

        has_inf = any(torch.isinf(p).any() for p in self.net.parameters())
        if has_inf:
            print("model parameters contain inf")

        for pnts, dirs in zip(split_points, split_viewdirs):
            val_all.append(self.net(pnts, coarse=True, viewdirs=dirs))

        out = torch.cat(val_all, dim=1)
        out = out.reshape(B, K, -1)  # (B, K, num_anchors_per_scale*7)

        # print if any of the out are nan
        if torch.isnan(out).any():
            print("out contains nan")

        # print if any of the out are inf
        if torch.isinf(out).any():
            print("out contains inf")

        # reshape the render to be (B, K, num_anchors_per_scale, 7)
        out = out.reshape(B, K, self.num_anchors_per_scale, 7)

        probabilities = torch.sigmoid(out[..., 0])  # (B, K, num_anchors_per_scale)

        # sum up the probabilities
        summed_probabilities = probabilities.sum(dim=1)  # (B, num_anchors_per_scale)

        # multiply the remaining values by the probabilities and sum them up by K
        final_values = out[..., 1:] * probabilities.unsqueeze(-1)  # (B, K, num_anchors_per_scale, 6)
        final_values = final_values.sum(dim=1)  # (B, num_anchors_per_scale, 6)

        # divide the summed values by the summed probabilities
        # add a small value to the denominator to avoid division by zero
        final_values = final_values / (summed_probabilities.unsqueeze(-1) + 1e-5)  # (B, num_anchors_per_scale, 6)

        max_probabilities = probabilities.max(dim=1)[0]  # (B, num_anchors_per_scale)

        # concatenate the probabilities and the values
        return torch.cat([max_probabilities.unsqueeze(-1), final_values], dim=-1)  # (B, num_anchors_per_scale, 7)

    def bind_parallel(self, net, gpus = None):
        self.net = net
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            return torch.nn.DataParallel(self, gpus, dim=1)

        return self