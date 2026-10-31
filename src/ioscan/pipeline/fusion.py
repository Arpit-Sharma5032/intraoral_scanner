from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ioscan.calibration.models import CameraIntrinsics
from ioscan.pipeline.pose import transform_points


@dataclass
class FusionConfig:
    voxel_size_mm: float = 0.15       # 150 um voxel grid
    sdf_trunc_mm: float = 0.60        # truncation distance (4x voxel size)
    use_tsdf: bool = True             # integrate into ScalableTSDFVolume
    splat_half_width_px: int = 2      # vertical splat radius so TSDF rays see a Ribbon


class ScanFuser:
    """Accumulates 3D laser profiles into a world PointCloud and Open3D TSDF Volume."""

    def __init__(self, camera: CameraIntrinsics, cfg: FusionConfig = FusionConfig()):
        self.camera = camera
        self.cfg = cfg
        self._world_slices: list[np.ndarray] = []
        self._view_origins: list[np.ndarray] = []

        self._o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy
        )
        self.tsdf: o3d.pipelines.integration.ScalableTSDFVolume | None = None
        if cfg.use_tsdf:
            self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                # Open3D TSDF geometry is expressed in metres; this project stores all
                # scanner geometry in millimetres.
                voxel_length=cfg.voxel_size_mm * 1e-3,
                sdf_trunc=cfg.sdf_trunc_mm * 1e-3,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            )

    def integrate(self, pts_cam: np.ndarray, uv: np.ndarray, T_wc: np.ndarray) -> None:
        """Integrate a single triangulated profile into the global map."""
        if pts_cam.shape[0] == 0:
            return

        pts_w = transform_points(pts_cam, T_wc)
        self._world_slices.append(pts_w)
        self._view_origins.append(np.asarray(T_wc[:3, 3], dtype=np.float64).copy())

        if self.tsdf is not None:
            depth_m = self._profile_to_depth_image(pts_cam, uv)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)),
                o3d.geometry.Image(depth_m),
                depth_scale=1.0,
                depth_trunc=0.10,
                convert_rgb_to_intensity=False,
            )
            # Open3D expects extrinsic = T_cw (world-to-camera) with translation in meters
            T_wc_m = T_wc.copy()
            T_wc_m[:3, 3] *= 0.001
            T_cw_m = np.linalg.inv(T_wc_m)
            self.tsdf.integrate(rgbd, self._o3d_intrinsics, T_cw_m)

    def _profile_to_depth_image(self, pts_cam: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Splat a 1D laser line into a narrow 2D depth ribbon (in meters) for TSDF integration."""
        H, W = self.camera.height, self.camera.width
        depth_m = np.zeros((H, W), dtype=np.float32)

        u_int = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
        v_int = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
        z_m = (pts_cam[:, 2] * 0.001).astype(np.float32)

        hw = self.cfg.splat_half_width_px
        for dv in range(-hw, hw + 1):
            v_shifted = np.clip(v_int + dv, 0, H - 1)
            depth_m[v_shifted, u_int] = z_m
        return depth_m

    def get_preview_point_cloud(self, voxel_downsample_mm: float = 0.20) -> o3d.geometry.PointCloud:
        """Return a lightweight point cloud snapshot for live dashboard previews.

        Unlike :meth:`get_point_cloud`, this deliberately skips normal estimation and
        orientation so it can be refreshed frequently while profiles are being fused.
        """
        pcd = o3d.geometry.PointCloud()
        if not self._world_slices:
            return pcd
        pcd.points = o3d.utility.Vector3dVector(np.vstack(self._world_slices))
        if voxel_downsample_mm > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel_downsample_mm))
        return pcd

    def get_point_cloud(self, voxel_downsample_mm: float = 0.10) -> o3d.geometry.PointCloud:
        """Return the fused 3D point cloud in world coordinates (mm)."""
        pcd = o3d.geometry.PointCloud()
        if not self._world_slices:
            return pcd

        all_pts = np.vstack(self._world_slices)
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        if voxel_downsample_mm > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample_mm)
        if len(pcd.points) < 4:
            return pcd
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=40)
        )
        # A fixed world-space camera location is wrong for curved trajectories and arbitrary
        # session origins. Orient each normal toward the camera position that observed its
        # nearest point on the scan path.
        if self._view_origins:
            points = np.asarray(pcd.points)
            origins = np.asarray(self._view_origins, dtype=np.float64)
            _, nearest = cKDTree(origins).query(points, k=1)
            normals = np.asarray(pcd.normals).copy()
            to_camera = origins[nearest] - points
            flip = np.einsum("ij,ij->i", normals, to_camera) < 0.0
            normals[flip] *= -1.0
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            pcd.orient_normals_consistent_tangent_plane(k=min(30, len(pcd.points) - 1))
        return pcd