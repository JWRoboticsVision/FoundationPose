from pathlib import Path
import numpy as np
import torch
from torch.nn import Module, ModuleList
from manopth.manolayer import ManoLayer

PROJ_ROOT = Path(__file__).resolve().parent

NEW_MANO_FACES = [
    [92, 38, 234],
    [234, 38, 239],
    [38, 122, 239],
    [239, 122, 279],
    [122, 118, 279],
    [279, 118, 215],
    [118, 117, 215],
    [215, 117, 214],
    [117, 119, 214],
    [214, 119, 121],
    [119, 120, 121],
    [121, 120, 78],
    [120, 108, 78],
    [78, 108, 79],
]
NUM_MANO_VERTS = 778
NUM_MANO_FACES = 1538


class MANOLayerWrapper(Module):
    """Wrapper layer for manopath ManoLayer."""

    def __init__(self, side, betas):
        """
        Constructor.
        Args:
            side: MANO hand type. 'right' or 'left'.
            betas: A numpy array of shape [10] containing the betas.
        """
        super(MANOLayerWrapper, self).__init__()

        self._side = side
        self._betas = betas

        self._mano_layer = ManoLayer(
            center_idx=0,
            flat_hand_mean=True,
            ncomps=45,
            side=side,
            mano_root=str(PROJ_ROOT / "data/ManoModels"),
            use_pca=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
            robust_rot=True,
        )

        # register buffer for betas
        b = torch.from_numpy(betas).unsqueeze(0).float()
        self.register_buffer("b", b)

        # register buffer for faces
        f = self._mano_layer.th_faces
        self.register_buffer("f", f)

        # register buffer for root translation
        v = (
            torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(0, 1)).permute(
                2, 0, 1
            )
            + self._mano_layer.th_v_template
        )
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer("root_trans", r)

    def forward(self, p, t):
        """
        Forward function.
        Args:
            p: A tensor of shape [B, 48] containing the pose.
            t: A tensor of shape [B, 3] containing the trans.
        Returns:
            v: A tensor of shape [B, 778, 3] containing the vertices.
            j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)

        # Convert to meters.
        v /= 1000.0
        j /= 1000.0
        return v, j

    @property
    def th_hands_mean(self):
        return self._mano_layer.th_hands_mean

    @property
    def th_selected_comps(self):
        return self._mano_layer.th_selected_comps

    @property
    def th_v_template(self):
        return self._mano_layer.th_v_template

    @property
    def side(self):
        return self._side

    @property
    def num_verts(self):
        return 778


class MANOGroupLayer(Module):
    """Wrapper layer to hold a group of MANOLayers."""

    def __init__(self, sides, betas):
        """Constructor.

        Args:
            sides: A list of MANO sides. 'right' or 'left'.
            betas: A list of numpy/tensor arrays of shape [10] containing the betas.
        """
        super(MANOGroupLayer, self).__init__()

        self._sides = sides
        self._betas = betas
        self._num_obj = len(self._sides)

        self._layers = ModuleList(
            [MANOLayerWrapper(s, b) for s, b in zip(self._sides, self._betas)]
        )

        # register buffer for faces
        f = torch.cat([self._layers[i].f + 778 * i for i in range(self._num_obj)])
        self.register_buffer("f", f)

        # register buffer for root translation
        r = torch.cat([l.root_trans for l in self._layers])
        self.register_buffer("root_trans", r)

    def forward(self, p, inds=None):
        """Forward function.

        Args:
            p: A tensor of shape [B, D] containing the pose vectors.
            inds: A list of sub-layer indices.

        Returns:
            v: A tensor of shape [B, N, 3] containing the vertices.
            j: A tensor of shape [B, J, 3] containing the joints.
        """
        if inds is None:
            inds = range(self._num_obj)
        v = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        j = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        p, t = self.pose2pt(p)
        for i in inds:
            y = self._layers[i](p[:, i], t[:, i])
            v.append(y[0])
            j.append(y[1])
        v = torch.cat(v, dim=1)
        j = torch.cat(j, dim=1)
        return v, j

    def pose2pt(self, pose):
        """Extracts pose, betas, and trans from pose vectors.

        Args:
            pose: A tensor of shape [B, D] containing the pose vectors.

        Returns:
            p: A tensor of shape [B, O, 48] containing the pose.
            t: A tensor of shape [B, O, 3] containing the trans.
        """
        p = torch.stack(
            [pose[:, 51 * i + 0 : 51 * i + 48] for i in range(self._num_obj)], dim=1
        )
        t = torch.stack(
            [pose[:, 51 * i + 48 : 51 * i + 51] for i in range(self._num_obj)], dim=1
        )
        return p, t

    def get_f_from_inds(self, inds):
        """Gets faces from sub-layer indices.

        Args:
            inds: A list of sub-layer indices.

        Returns:
            f: A tensor of shape [F, 3] containing the faces.
            m: A tensor of shape [F] containing the face to index mapping.
        """
        f = [torch.zeros((0, 3), dtype=self.f.dtype, device=self.f.device)]
        m = [torch.zeros((0,), dtype=torch.int64, device=self.f.device)]
        for i, x in enumerate(inds):
            f.append(self._layers[x].f + 778 * i)
            m.append(x * torch.ones(1538, dtype=torch.int64, device=self.f.device))
        f = torch.cat(f)
        m = torch.cat(m)
        return f, m

    @property
    def num_obj(self):
        return self._num_obj


class RGBA:
    def __init__(self, red, green, blue, alpha=255):
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __str__(self):
        return "({},{},{},{})".format(self.red, self.green, self.blue, self.alpha)

    @property
    def rgb(self):
        return (self.red, self.green, self.blue)

    @property
    def bgr(self):
        return (self.blue, self.green, self.red)

    @property
    def rgb_norm(self):
        return (self.red / 255.0, self.green / 255.0, self.blue / 255.0)

    @property
    def bgr_norm(self):
        return (self.blue / 255.0, self.green / 255.0, self.red / 255.0)


COLORS = {
    "red": RGBA(255, 0, 0, 255),
    "dark_red": RGBA(128, 0, 0, 255),
    "green": RGBA(0, 255, 0, 255),
    "dark_green": RGBA(0, 128, 0, 255),
    "blue": RGBA(0, 0, 255, 255),
    "yellow": RGBA(255, 255, 0, 255),
    "magenta": RGBA(255, 0, 255, 255),
    "cyan": RGBA(0, 255, 255, 255),
    "orange": RGBA(255, 165, 0, 255),
    "purple": RGBA(128, 0, 128, 255),
    "brown": RGBA(165, 42, 42, 255),
    "pink": RGBA(255, 192, 203, 255),
    "lime": RGBA(0, 255, 127, 255),
    "navy": RGBA(0, 0, 128, 255),
    "teal": RGBA(0, 128, 128, 255),
    "olive": RGBA(128, 128, 0, 255),
    "maroon": RGBA(128, 0, 0, 255),
    "coral": RGBA(255, 127, 80, 255),
    "turquoise": RGBA(64, 224, 208, 255),
    "indigo": RGBA(75, 0, 130, 255),
    "violet": RGBA(238, 130, 238, 255),
    "gold": RGBA(255, 215, 0, 255),
    "skin": RGBA(192, 134, 107, 255),
    "white": RGBA(255, 255, 255, 255),
    "black": RGBA(0, 0, 0, 255),
    "gray": RGBA(128, 128, 128, 255),
    "darkgray": RGBA(64, 64, 64, 255),
    "lightgray": RGBA(192, 192, 192, 255),
    "tomato": RGBA(255, 99, 71, 255),
}

# RGB colors for Object classes
OBJ_CLASS_COLORS = [
    COLORS["black"],  # background
    COLORS["red"],  # object_id 1
    COLORS["green"],  # object_id 2
    COLORS["blue"],  # object_id 3
    COLORS["yellow"],  # object_id 4
    COLORS["magenta"],  # object_id 5
    COLORS["cyan"],  # object_id 6
]

# RGB colors for Hands
HAND_COLORS = [
    COLORS["black"],  # background
    COLORS["turquoise"],  # right
    COLORS["tomato"],  # left
]
