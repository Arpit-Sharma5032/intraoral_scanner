from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class BuiltinMeshSegNet(nn.Module):
    """MeshSegNet (Lian et al., 2020) with the layer names of the reference implementation.

    Layer names follow the public MeshSegNet repository (mlp1_*, fstn, glm1_*, mlp2_*, glm2_*,
    mlp3_*, output_conv) so its checkpoints can be loaded with ``strict=True``. If a checkpoint
    still does not load, ``segment.py`` prints the missing/unexpected keys rather than guessing.
    Input: x (B, C, N), a_s (B, N, N), a_l (B, N, N). Output: (B, N, num_classes) probabilities.
    """

    def __init__(
        self,
        num_classes: int = 15,
        num_channels: int = 15,
        with_dropout: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        self.mlp1_conv1 = nn.Conv1d(num_channels, 64, 1)
        self.mlp1_conv2 = nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)

        self.fstn = STNkd(k=64)

        self.glm1_conv1_1 = nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = nn.Conv1d(32 + 32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)

        self.mlp2_conv1 = nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)

        self.glm2_conv1_1 = nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = nn.Conv1d(128 * 3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)

        self.mlp3_conv1 = nn.Conv1d(64 + 512 + 512 + 512, 256, 1)
        self.mlp3_conv2 = nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)

        self.output_conv = nn.Conv1d(128, num_classes, 1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, a_s: torch.Tensor, a_l: torch.Tensor) -> torch.Tensor:
        n_pts = x.size(2)
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))

        trans_feat = self.fstn(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat)  # (B, N, 64)

        sap = torch.bmm(a_s, x).transpose(2, 1)
        x_ftm = x.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))

        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)

        x_t = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_t).transpose(2, 1)
        sap_2 = torch.bmm(a_l, x_t).transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))

        x = torch.max(x_glm2, 2, keepdim=True)[0]
        x = x.expand(-1, -1, n_pts)  # global feature broadcast to every cell
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)

        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))
        x = self.output_conv(x).transpose(2, 1).contiguous()
        return F.softmax(x, dim=-1)


def load_meshsegnet_class(external_repo_dir: str | Path | None = None):
    """Load an explicitly requested reference class, or use the bundled compatible model.

    The bundled implementation has the same parameter names and tensor shapes as the public
    Tai-Hsien reference implementation. An external checkout is only imported when the caller
    explicitly provides its path; this avoids accidentally importing an unrelated installed
    module named ``meshsegnet``.
    """
    external_repo_dir = external_repo_dir or os.environ.get("IOSCAN_MESHSEGNET_REPO")
    if external_repo_dir is None:
        return BuiltinMeshSegNet

    repo = Path(external_repo_dir).expanduser()
    module_path = repo / "meshsegnet.py" if repo.is_dir() else repo
    if not module_path.is_file():
        raise FileNotFoundError(
            f"MeshSegNet reference file not found: {module_path}. "
            "--repo-dir must point to a checkout containing meshsegnet.py or directly to that file."
        )

    spec = importlib.util.spec_from_file_location("_ioscan_external_meshsegnet", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import MeshSegNet reference implementation: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, "MeshSegNet", None)
    if model_class is None:
        raise ImportError(f"No MeshSegNet class found in {module_path}")
    return model_class
