import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.data.tooth_dual import ToothDualDataset
from src.dpsr import DPSR
from src.model import Encode2Points
from src.utils import export_mesh, export_pointcloud, load_config, load_model_manual, mc_from_psr


def sap_to_world(points, center, scale):
    return (points - 0.5) * scale + center


def main():
    parser = argparse.ArgumentParser(description="Generate SAP tooth dual reconstructions.")
    parser.add_argument("config", type=str, help="Path to config file saved for training.")
    parser.add_argument("--test_list", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    args = parser.parse_args()

    cfg = load_config(args.config, "configs/default.yaml")
    if args.test_list is not None:
        cfg["data"]["test_list"] = args.test_list
    if cfg["data"].get("test_list") is None:
        raise ValueError("Missing data:test_list in config or --test_list.")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = ToothDualDataset(
        list_file=cfg["data"]["test_list"],
        pointcloud_n=cfg["data"]["pointcloud_n"],
        num_gt_points=cfg["data"]["num_gt_points"],
        global_num_points=cfg["data"]["global_num_points"],
        file0_surface_points=cfg["data"]["file0_surface_points"],
        global_dist_threshold=cfg["data"]["global_dist_threshold"],
        global_min_points=cfg["data"]["global_min_points"],
        grid_res=cfg["model"]["grid_res"],
        psr_sigma=cfg["model"]["psr_sigma"],
        cache_dir=cfg["data"].get("psr_cache_dir"),
        split="test",
    )

    model = Encode2Points(cfg).to(device)
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["train"]["out_dir"], "model_best.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    load_model_manual(state_dict["state_dict"], model)
    model.eval()

    resolution = cfg["generation"].get("psr_resolution", 0) or cfg["model"]["grid_res"]
    sigma = cfg["generation"].get("psr_sigma", 0) or cfg["model"]["psr_sigma"]
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=sigma).to(device)

    out_dir = args.output_dir or os.path.join(cfg["train"]["out_dir"], cfg["generation"]["generation_dir"])
    mesh_dir = os.path.join(out_dir, "meshes")
    pcl_dir = os.path.join(out_dir, "pointcloud")
    input_dir = os.path.join(out_dir, "input")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(pcl_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), ncols=100):
            item = dataset[idx]
            model_id = item["model_id"]
            inputs = torch.from_numpy(item["inputs"][None]).float().to(device)
            points, normals = model(inputs)
            psr_grid = dpsr(points, normals)
            verts, faces, _ = mc_from_psr(psr_grid, zero_level=cfg["data"]["zero_level"])

            center = item["norm_center"][None, :]
            scale = float(item["norm_scale"][0])
            verts_world = sap_to_world(verts, center, scale)
            points_world = sap_to_world(points[0].detach().cpu().numpy(), center, scale)
            inputs_world = sap_to_world(item["inputs"], center, scale)

            normals_np = normals[0].detach().cpu().numpy()
            export_mesh(os.path.join(mesh_dir, f"{model_id}.ply"), verts_world, faces)
            export_pointcloud(os.path.join(pcl_dir, f"{model_id}.ply"), points_world, normals_np)
            export_pointcloud(os.path.join(input_dir, f"{model_id}.ply"), inputs_world)


if __name__ == "__main__":
    main()
