#!/usr/bin/env python3
"""Evaluate SAP tooth-dual checkpoints and paste reconstructions into arches."""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, dname)

from src.mesh_evaluator import MeshEvaluator
from src.data.tooth_dual import _load_vertices, _load_vertices_and_normals, _sample_fixed, _sample_surface, bbox_normalizer, sap_normalizer
from src.dpsr import DPSR
from src.model import Encode2Points
from src.utils import export_pointcloud, load_config, load_model_manual, mc_from_psr


def to_sap(points, center, scale):
    return (points - center[None, :]) / scale + 0.5


def sap_to_world(points, center, scale):
    return (points - 0.5) * scale + center[None, :]


def to_eval(points, center, scale):
    """指标坐标系：完整牙弓 X0 的包围盒，最长边映射到 [-1, 1]（仅用于算指标）。"""
    return (points - center[None, :]) / scale * 2.0


def hausdorff_distance(pc1, pc2):
    return max(directed_hausdorff(pc1, pc2)[0], directed_hausdorff(pc2, pc1)[0])


def scene_name(pred_path):
    return f"{Path(pred_path).parent.name}_{Path(pred_path).stem}"


def missing_arch_path(file0_path):
    candidate = Path(str(file0_path).replace("_0.ply", "_1.ply"))
    if not candidate.exists():
        raise FileNotFoundError(f"Defective arch X1 not found: {candidate}")
    return candidate


def read_entries(list_path):
    entries = []
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                entries.append(tuple(parts[:3]))
    if not entries:
        raise ValueError(f"No valid three-column entries in {list_path}")
    return entries


def write_combined(arch_path, tooth_vertices, tooth_faces, output_path, tooth_colors=None):
    arch_mesh = o3d.io.read_triangle_mesh(str(arch_path))
    if len(arch_mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(str(arch_path))
        arch_v = np.asarray(pcd.points)
        arch_f = np.zeros((0, 3), dtype=np.int32)
        arch_c = np.asarray(pcd.colors) if pcd.has_colors() else None
    else:
        arch_v = np.asarray(arch_mesh.vertices)
        arch_f = np.asarray(arch_mesh.triangles)
        arch_c = np.asarray(arch_mesh.vertex_colors) if arch_mesh.has_vertex_colors() else None

    tooth_v = np.asarray(tooth_vertices)
    tooth_f = np.asarray(tooth_faces, dtype=np.int32)
    all_v = np.concatenate([arch_v, tooth_v], axis=0)

    faces = []
    if arch_f.shape[0] > 0:
        faces.append(arch_f)
    if tooth_f.shape[0] > 0:
        faces.append(tooth_f + arch_v.shape[0])
    all_f = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int32)

    colors = np.zeros((all_v.shape[0], 3), dtype=np.float64)
    if arch_c is not None and arch_c.shape == arch_v.shape:
        colors[: arch_v.shape[0]] = np.clip(arch_c, 0.0, 1.0)
    else:
        colors[: arch_v.shape[0]] = [0.72, 0.72, 0.72]
    if tooth_colors is not None and tooth_colors.shape == tooth_v.shape:
        colors[arch_v.shape[0] :] = np.clip(tooth_colors, 0.0, 1.0)
    else:
        colors[arch_v.shape[0] :] = [1.0, 0.18, 0.12]

    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(all_v)
    combined.triangles = o3d.utility.Vector3iVector(all_f)
    combined.vertex_colors = o3d.utility.Vector3dVector(colors)
    combined.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), combined, write_ascii=False)
    return combined


def write_mesh(path, vertices, faces, colors=None):
    mesh = make_o3d_mesh(vertices, faces)
    if colors is not None and colors.shape == vertices.shape:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    return mesh


def make_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


def read_mesh_or_vertices(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        return None, points, normals
    mesh.compute_vertex_normals()
    return mesh, np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)


def normalize_o3d_mesh(mesh, center, scale):
    vertices = to_eval(np.asarray(mesh.vertices), center, scale)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(vertices)
    out.triangles = mesh.triangles
    out.compute_vertex_normals()
    return out


def read_colored_pointcloud(path):
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0 or not pcd.has_colors():
        return None, None
    return np.asarray(pcd.points, dtype=np.float64), np.asarray(pcd.colors, dtype=np.float64)


def transfer_vertex_colors(vertices, color_points, color_values):
    if color_points is None or color_values is None:
        return None
    if len(color_points) == 0 or color_points.shape != color_values.shape:
        return None
    tree = cKDTree(color_points)
    _, nearest = tree.query(vertices, k=1, workers=-1)
    return np.clip(color_values[nearest], 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAP tooth dual reconstruction.")
    parser.add_argument("config", help="训练时保存的 config.yaml，或 configs/learning_based/tooth_dual/*.yaml")
    parser.add_argument("--checkpoint", default=None, help="默认 <train.out_dir>/model_best.pt")
    parser.add_argument("--test_list", required=True, help="Stage I 导出的 test.lst（三列：predtooth file0 file3）")
    parser.add_argument("--output_dir", default=None, help="默认 <train.out_dir>/eval_test")
    parser.add_argument("--n_eval_pts", type=int, default=100000)
    parser.add_argument("--pointcloud_n", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（输入重采样与 GT 表面采样），保证评估可复现")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config, "configs/default.yaml")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg["train"]["out_dir"], "model_best.pt")
    if args.output_dir is None:
        args.output_dir = os.path.join(cfg["train"]["out_dir"], "eval_test")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"设备: {device}")
    print(f"checkpoint: {args.checkpoint}")

    out_dir = Path(args.output_dir)
    tooth_dir = out_dir / "tooth_mesh"
    combined_dir = out_dir / "combined"
    pcl_dir = out_dir / "pointcloud"
    input_dir = out_dir / "input"
    for d in [tooth_dir, combined_dir, pcl_dir, input_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = Encode2Points(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    load_model_manual(state["state_dict"], model)
    model.eval()

    resolution = cfg["generation"].get("psr_resolution", 0) or cfg["model"]["grid_res"]
    sigma = cfg["generation"].get("psr_sigma", 0) or cfg["model"]["psr_sigma"]
    zero_level = cfg["data"].get("zero_level", 0)
    pointcloud_n = args.pointcloud_n or cfg["data"]["pointcloud_n"]
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=sigma).to(device)

    entries = read_entries(args.test_list)
    total = len(entries) if args.max_samples == 0 else min(args.max_samples, len(entries))
    evaluator = MeshEvaluator(n_points=args.n_eval_pts, metric_names=MeshEvaluator.ESSENTIAL_METRICS)
    rows = []
    recon_time = 0.0

    with torch.no_grad():
        for idx, (pred_path, file0_path, file3_path) in enumerate(tqdm(entries[:total], ncols=100)):
            name = scene_name(pred_path)
            x1_path = missing_arch_path(file0_path)
            # 指标坐标系：完整牙弓 X0 的包围盒（仅用于算指标，不进入模型输入）
            center, scale = bbox_normalizer(_load_vertices(str(file0_path)))
            scale = float(scale)

            pred_points, pred_normals = _load_vertices_and_normals(pred_path)
            # SAP 推理坐标系：X1 ∪ 预测牙冠 的包围盒（推理时可得）
            sap_center, sap_scale = sap_normalizer(str(x1_path), pred_points)
            inputs = np.clip(to_sap(pred_points, sap_center, sap_scale), 0.0, 0.99).astype(np.float32)
            inputs, pred_normals = _sample_fixed(inputs, pred_normals, pointcloud_n)

            t0 = time.time()
            inputs_t = torch.from_numpy(inputs[None]).float().to(device)
            points, normals = model(inputs_t)
            psr_grid = dpsr(points, normals)
            verts_sap, faces, _ = mc_from_psr(psr_grid, zero_level=zero_level)
            recon_time += time.time() - t0

            verts_world = sap_to_world(verts_sap, sap_center, sap_scale)
            points_world = sap_to_world(points[0].detach().cpu().numpy(), sap_center, sap_scale)
            inputs_world = sap_to_world(inputs, sap_center, sap_scale)
            normals_np = normals[0].detach().cpu().numpy()
            # 牙冠颜色取自 Stage I 预测点云自带的 RGB（世界坐标）
            color_points, color_values = read_colored_pointcloud(pred_path)
            vertex_colors = transfer_vertex_colors(verts_world, color_points, color_values)

            tooth_path = tooth_dir / f"{name}.ply"
            write_mesh(tooth_path, verts_world, faces, vertex_colors)
            export_pointcloud(str(pcl_dir / f"{name}.ply"), points_world, normals_np)
            export_pointcloud(str(input_dir / f"{name}.ply"), inputs_world)
            combined_mesh = write_combined(
                missing_arch_path(file0_path),
                verts_world,
                faces,
                combined_dir / f"{name}_combined.ply",
                tooth_colors=vertex_colors,
            )

            gt_world, gt_normals = _sample_surface(file3_path, args.n_eval_pts)
            gt_eval = to_eval(gt_world, center, scale)
            verts_eval = to_eval(verts_world, center, scale)
            pred_mesh_eval = make_o3d_mesh(verts_eval, faces)
            eval_dict = evaluator.eval_mesh(pred_mesh_eval, gt_eval, gt_normals)

            gt_arch_world, gt_arch_normals = _sample_surface(file0_path, args.n_eval_pts)
            gt_arch_eval = to_eval(gt_arch_world, center, scale)
            combined_eval_mesh = normalize_o3d_mesh(combined_mesh, center, scale)
            combined_eval = evaluator.eval_mesh(combined_eval_mesh, gt_arch_eval, gt_arch_normals)

            hd_single = hausdorff_distance(verts_eval, gt_eval)
            combined_verts_eval = to_eval(np.asarray(combined_mesh.vertices), center, scale)
            hd_full = hausdorff_distance(combined_verts_eval, gt_arch_eval)

            row = {
                "scene": name,
                "chamfer-L1": float(eval_dict.get("chamfer-L1", -1)),
                "f-score": float(eval_dict.get("f-score", -1)),
                "normals": float(eval_dict.get("normals", -1)),
                "hd": float(hd_single),
                "combined_chamfer-L1": float(combined_eval.get("chamfer-L1", -1)),
                "combined_f-score": float(combined_eval.get("f-score", -1)),
                "combined_normals": float(combined_eval.get("normals", -1)),
                "combined_hd": float(hd_full),
            }
            rows.append(row)
            tqdm.write(
                f"[{idx + 1}/{total}] {name} "
                f"CD={row['chamfer-L1']:.4f} F={row['f-score']:.4f} N={row['normals']:.4f} HD={row['hd']:.4f} "
                f"FullCD={row['combined_chamfer-L1']:.4f} FullN={row['combined_normals']:.4f} FullHD={row['combined_hd']:.4f}"
            )
            if use_cuda:
                torch.cuda.empty_cache()

    valid = [r for r in rows if r["chamfer-L1"] >= 0]
    csv_path = out_dir / "result.csv"
    fields = [
        "scene",
        "chamfer-L1",
        "f-score",
        "normals",
        "hd",
        "combined_chamfer-L1",
        "combined_f-score",
        "combined_normals",
        "combined_hd",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        if valid:
            mean_row = {
                "scene": "MEAN",
                "chamfer-L1": float(np.mean([r["chamfer-L1"] for r in valid])),
                "f-score": float(np.mean([r["f-score"] for r in valid])),
                "normals": float(np.mean([r["normals"] for r in valid])),
                "hd": float(np.mean([r["hd"] for r in valid])),
                "combined_chamfer-L1": float(np.mean([r["combined_chamfer-L1"] for r in valid])),
                "combined_f-score": float(np.mean([r["combined_f-score"] for r in valid])),
                "combined_normals": float(np.mean([r["combined_normals"] for r in valid])),
                "combined_hd": float(np.mean([r["combined_hd"] for r in valid])),
            }
            writer.writerow(mean_row)
        else:
            mean_row = None

    print(f"\n完成 {len(valid)}/{len(rows)} 个样本")
    if mean_row is not None:
        print(f"平均 chamfer-L1 = {mean_row['chamfer-L1']:.4f}")
        print(f"平均 f-score    = {mean_row['f-score']:.4f}")
        print(f"平均 normals    = {mean_row['normals']:.4f}")
        print(f"完整 chamfer-L1 = {mean_row['combined_chamfer-L1']:.4f}")
        print(f"完整 normals    = {mean_row['combined_normals']:.4f}")
    print(f"重建时间: {recon_time:.1f}s ({recon_time / max(len(valid), 1):.2f}s/样本)")
    print(f"CSV: {csv_path}")
    print(f"单牙 mesh: {tooth_dir}")
    print(f"完整牙列: {combined_dir}")


if __name__ == "__main__":
    main()
