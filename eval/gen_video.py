import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import tqdm
import torchvision.transforms as transforms


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

# scale: egyszeru szorzo a kimeneti meretekre

device = util.get_cuda(args.gpu_id[0])

# itt tortenik az adatok beolvasasa
dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)

data = dset[args.subset]
data_path = data["path"]
print("Data instance loaded:", data_path)

# ezek a kepek, nehannyal dolgozik, a tobbi meg nyilvan ellenorzesre van torch.Tensor(49, 3, 400, 300)
# torch.Tensor(hany kep van, 3 RGB, magassag, szelesseg)
# naluk 49 kepbol all egy jelenet, 400x300-as felbontasban
images = data["images"]  # (NV, 3, H, W)

# ezek az iranyok, ahonnan a kepek keszultek
poses = data["poses"]  # (NV, 4, 4)
focal = data["focal"]
if isinstance(focal, float):
    # Dataset implementations are not consistent about
    # returning float or scalar tensor in case of fx=fy
    focal = torch.tensor(focal, dtype=torch.float32)
focal = focal[None]

# kamera kozeppontja
c = data.get("c")
if c is not None:
    c = c.to(device=device).unsqueeze(0)

NV, _, H, W = images.shape

if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".format(
                args.scale, H, W
            )
        )
    H, W = Ht, Wt

# halo betoltese
net = make_model(conf["model"]).to(device=device)
net.load_weights(args)

# renderer betoltese aminek odaadjuk a halot
renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
).to(device=device)

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

print("Generating rays")

dtu_format = hasattr(dset, "sub_format") and dset.sub_format == "dtu"

# ez az if-else annyit csinal hogy valami dtu specifikus kamera poziciokat allit elo (ha dtu render van), amugy sima 360
# if dtu_format:
if False:
    print("Using DTU camera trajectory")
    # Use hard-coded pose interpolation from IDR for DTU

    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = torch.tensor(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = args.num_views // 5
    args.num_views = n_inter * 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).float()

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        new_q = new_q.unsqueeze(0)
        R = util.quat_to_rot(new_q)
        t = R[:, :, 2] * scale
        new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        new_pose[:, :3, :3] = R
        new_pose[:, :3, 3] = t
        render_poses.append(new_pose)
    render_poses = torch.cat(render_poses, dim=0)
else:
    print("Using default (360 loop) camera trajectory")
    if args.radius == 0.0:
        radius = (z_near + z_far) * 0.5
        print("> Using default camera radius", radius)
    else:
        radius = args.radius

    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            util.pose_spherical(angle, args.elevation, radius)
            for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

# a korabbi kamera poziciok alapjan letrehozza a sugarakat
render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal * args.scale,
    z_near,
    z_far,
    c=c * args.scale if c is not None else None,
).to(device=device)
# (NV, H, W, 8)

focal = focal.to(device=device)

# forras kepek kivalasztasa, vagy random
source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1
assert not (source >= NV).any()

# eredeti NeRF paperben van szo a coarse es fine renderrol, ez a rendereles gyorsitasa miatt jo
if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
    renderer.n_fine = 128


# ------------------------------------------------------------

image_dir = 'C:/Users/kofinandi/Downloads/test_set'

image_tensors = []
pose_tensors = []

for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        image = imageio.imread(image_path)[..., :3]

        print("image: " + filename)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        tensor = transform(image)

        image_tensors.append(tensor)

    if filename.endswith('.npy'):
        if filename.startswith('intrinsic'):
            print("intrinsic: " + filename)
            intrinsic = np.load(os.path.join(image_dir, filename))
        if filename.startswith('extrinsic'):
            print("extrinsic: " + filename)
            extrinsic = np.load(os.path.join(image_dir, filename))

            pose = np.identity(4)
            pose[0:3, 0:3] = extrinsic[0:3, 0:3].transpose()
            pose[0:3, 3] = extrinsic[0:3, 3]

            pose = extrinsic

            pose = (
                    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
            )

            pose_tensors.append(pose)

tensor_batch = torch.stack(image_tensors).to(device=device)

print('Tensor shape:', tensor_batch.shape)

pose_tensors = torch.stack(pose_tensors).to(device=device)

intrinsic = np.load('C:/Users/kofinandi/Downloads/test_set/intrinsic_0002.npy')
focal = torch.tensor(([intrinsic[0, 0], intrinsic[1, 1]]), dtype=torch.float32).unsqueeze(0).to(device=device)

# ------------------------------------------------------------



# nyilvan no grad mert test time van, forward propagation es kepek szamolasa megy
with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    # beadja a halonak a kepeket es a poziciokat, ezekhez mar csak a pontos sugarak (5d koordinatak) kellenek
    net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        # tensor_batch.unsqueeze(0),
        # pose_tensors.unsqueeze(0),
        # poses[43].unsqueeze(0).unsqueeze(0).to(device=device),
        focal,
        c=c,
    )

    print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    # ez a ciklus rendereli az egyes sugarakat
    for rays in tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
    ):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    # rgb_fine (V*H*W, 3)

    # megvan az osszes sugar egy nagy tombben, vissza kell alakitani a kepekbe
    frames = rgb_fine.view(-1, H, W, 3)

# innentol kiiras videoba
print("Writing video")
vid_name = "{:04}".format(args.subset)
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name, "video" + vid_name + ".mp4")
viewimg_path = os.path.join(
    args.visual_path, args.name, "video" + vid_name + "_view.jpg"
)
imageio.mimwrite(
    vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
)

img_np = (data["images"][src_view].permute(0, 2, 3, 1) * 0.5 + 0.5).numpy()
img_np = (img_np * 255).astype(np.uint8)
img_np = np.hstack((*img_np,))
imageio.imwrite(viewimg_path, img_np)

print("Wrote to", vid_path, "view:", viewimg_path)
