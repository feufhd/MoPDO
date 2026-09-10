# inference_mopdo.py

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast

from models_mopdo.networks import CONFIGS, MoPDOVisionTransformer
from datasets_mopdo.VTABDataLoader import get_data
from datasets_mopdo.Explore_VTABConfig import DATA_CONFIGS
from utils import seed_torch, accuracy, AverageMeter, Logger, count_parameters
from torchvision.utils import save_image
from PIL import Image, ImageDraw
import shutil
import matplotlib.cm as cm
import numpy as np


def _load_pt_map_as_numpy(pt_path):
    """
    Load a saved map tensor and convert it to numpy array [H, W] in [0, 1].

    Expected possible shapes:
      [B, 1, H, W]
      [1, H, W]
      [H, W]
    """
    m = torch.load(pt_path, map_location="cpu")

    if isinstance(m, torch.Tensor):
        m = m.detach().float()
    else:
        raise TypeError(f"Expected tensor in {pt_path}, got {type(m)}")

    if m.dim() == 4:
        # [B, 1, H, W], use first sample
        m = m[0, 0]
    elif m.dim() == 3:
        # [1, H, W] or [B, H, W], use first channel/sample
        m = m[0]
    elif m.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported map shape {tuple(m.shape)} in {pt_path}")

    m = m.numpy()
    m = m - m.min()
    m = m / (m.max() + 1e-8)
    return m


def _overlay_heatmap_on_image(
    ori_img,
    heatmap,
    alpha=0.45,
    cmap_name="jet",
):
    """
    Overlay heatmap [H, W] in [0, 1] on ori_img.
    ori_img: PIL RGB image.
    Return: PIL RGB image.
    """
    ori_img = ori_img.convert("RGB").resize((224, 224))

    if heatmap.shape != (224, 224):
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((224, 224), resample=Image.BILINEAR)
        heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(heatmap)[:, :, :3]  # [H, W, 3], RGB in [0, 1]
    colored = (colored * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).convert("RGB")

    blended = Image.blend(ori_img, colored_img, alpha=alpha)
    return blended


def concat_block_visualizations(
    save_dir,
    num_blocks=12,
    alpha=0.45,
    cmap_name="jet",
):
    """
    For each block_i, read:
      ori.png
      block_i/q_base_map_224.pt
      block_i/q_hco_map_224.pt
      block_i/q_diff_map_224.pt

    Overlay each map on ori.png as a heatmap mask, then concatenate:
      Original | Q Base | Q MoPDO | Q Diff

    Save as:
      save_dir/vis_i.png

    Also save separate overlays as:
      block_i/vis_base.png
      block_i/vis_mopdo.png
      block_i/vis_diff.png
    """
    ori_path = os.path.join(save_dir, "ori.png")
    if not os.path.exists(ori_path):
        print(f"[Warning] Missing ori.png: {ori_path}")
        return

    ori_img = Image.open(ori_path).convert("RGB").resize((224, 224))

    columns = [
        ("Original", None),
        ("Q Base", "q_base_map_224.pt"),
        ("Q MoPDO", "q_hco_map_224.pt"),
        ("Q Diff", "q_diff_map_224.pt"),
    ]

    for i in range(num_blocks):
        block_dir = os.path.join(save_dir, f"block_{i}")

        pt_paths = []
        for title, filename in columns:
            if filename is None:
                pt_paths.append(None)
            else:
                pt_paths.append(os.path.join(block_dir, filename))

        missing = [p for p in pt_paths if p is not None and not os.path.exists(p)]
        if missing:
            print(f"[Warning] Skip block {i}, missing files: {missing}")
            continue

        imgs = []
        overlay_dict = {}

        for title, pt_path in zip([c[0] for c in columns], pt_paths):
            if pt_path is None:
                imgs.append(ori_img.copy())
            else:
                heatmap = _load_pt_map_as_numpy(pt_path)
                overlay = _overlay_heatmap_on_image(
                    ori_img,
                    heatmap,
                    alpha=alpha,
                    cmap_name=cmap_name,
                )
                imgs.append(overlay)
                overlay_dict[title] = overlay

        # Save separate overlays.
        if "Q Base" in overlay_dict:
            overlay_dict["Q Base"].save(os.path.join(block_dir, "vis_base.png"))
        if "Q MoPDO" in overlay_dict:
            overlay_dict["Q MoPDO"].save(os.path.join(block_dir, "vis_mopdo.png"))
        if "Q Diff" in overlay_dict:
            overlay_dict["Q Diff"].save(os.path.join(block_dir, "vis_diff.png"))

        pad = 8
        title_h = 26
        w, h = 224, 224
        canvas_w = len(imgs) * w + (len(imgs) - 1) * pad
        canvas_h = h + title_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for j, (title, _) in enumerate(columns):
            x0 = j * (w + pad)
            draw.text((x0 + 5, 5), title, fill=(0, 0, 0))
            canvas.paste(imgs[j], (x0, title_h))

        out_path = os.path.join(save_dir, f"vis_{i}.png")
        canvas.save(out_path)


def compute_erank_from_q_diff(save_dir, num_blocks=12, remove_cls=True, eps=1e-12):
    """
    Compute effective rank for q_diff.pt in each block.

    q_diff.pt shape is expected to be [B, N, C].
    Return:
        results: dict, block_idx -> erank value or None.
    """
    results = {}

    for i in range(num_blocks):
        q_diff_path = os.path.join(save_dir, f"block_{i}", "q_diff.pt")

        if not os.path.exists(q_diff_path):
            print(f"[Warning] Missing q_diff.pt for block {i}: {q_diff_path}")
            results[i] = None
            continue

        q_diff = torch.load(q_diff_path, map_location="cpu")

        if not isinstance(q_diff, torch.Tensor):
            print(f"[Warning] q_diff is not a tensor for block {i}: {type(q_diff)}")
            results[i] = None
            continue

        if q_diff.dim() != 3:
            print(f"[Warning] Expected q_diff shape [B, N, C], got {tuple(q_diff.shape)}")
            results[i] = None
            continue

        # Usually q_diff shape is [1, 197, 768]. Remove CLS token.
        if remove_cls and q_diff.shape[1] > 1:
            q_diff = q_diff[:, 1:, :]

        erank_list = []
        for b in range(q_diff.shape[0]):
            mat = q_diff[b].float()  # [N, C]
            s = torch.linalg.svdvals(mat)
            p = s / (s.sum() + eps)
            erank = torch.exp(-(p * torch.log(p + eps)).sum())
            erank_list.append(erank.item())

        # If B=1, this is just one value. If B>1, use mean.
        erank_mean = sum(erank_list) / len(erank_list)
        results[i] = erank_mean

    return results


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="MoPDO inference")
    parser.add_argument("--dataset_name", default="cifar")
    parser.add_argument("--model_type", default="ViT-B_16")
    parser.add_argument("--dataset_dir", default="./data/vtab-1k")
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="ViT-B_16.npz",
        help="Path to the original pre-trained ViT npz checkpoint."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to the fine-tuned MoPDO checkpoint, e.g., xxx.pth."
    )
    parser.add_argument("--output_dir", default="output/vtab_mopdo_eval", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--num_classes", default=100, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--simple_aug", default=True, type=bool)

    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Only run several batches for visualization/debug. Default: run all."
    )

    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Whether to save logits/preds/labels."
    )

    args = parser.parse_args()
    return args


def frozen_param(model, frozen_list=("",)):
    for name, param in model.named_parameters():
        if any(item in name for item in frozen_list):
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_params = count_parameters(model)
    print("Total trainable parameters: \t%2.3fM" % num_params)


def setup_model(args, frozen_list=("head", "scale", "shift", "hco")):
    config = CONFIGS[args.model_type]

    model = MoPDOVisionTransformer(
        config,
        args.img_size,
        zero_head=True,
        num_classes=args.num_classes,
        drop_path=args.drop_path,
    )

    print(f"Loading pre-trained weights from: {args.pretrained_dir}")
    model.load_from(np.load(args.pretrained_dir))

    frozen_param(model, frozen_list)
    return model


def load_finetuned_checkpoint(model, resume_path):
    if resume_path is None or resume_path == "":
        print("No fine-tuned checkpoint is loaded. Using only pre-trained weights.")
        return model

    print(f"Loading fine-tuned checkpoint from: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu")

    # Your training save_model() saves a state_dict containing selected keys directly.
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model


@torch.no_grad()
def inference(model, test_loader, device, max_batches=None, save_logits=False):
    model.eval()

    top1 = AverageMeter("Acc@1", ":6.2f")
    losses = AverageMeter("Loss", ":.4e")
    criterion = nn.CrossEntropyLoss()

    saved_outputs = []

    for batch_idx, batch in enumerate(tqdm(test_loader)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Original VTABDataLoader usually returns (x, label).
        # If you later modify it to return (x, label, ori_path), this also supports it.
        if len(batch) >= 2:
            x, label = batch[0], batch[1]
        else:
            raise ValueError("Expected batch to contain at least x and label.")

        x = x.to(device)
        label = label.to(device)

        # Save original natural image before model inference.
        # x is normalized by mean/std, so we need to de-normalize it first.
        save_dir = os.path.join(save_root, "vis_outputs", f"{batch_idx}")
        os.makedirs(save_dir, exist_ok=True)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x_vis = x.detach() * std + mean
        x_vis = x_vis.clamp(0, 1)

        save_image(x_vis[0].cpu(), os.path.join(save_dir, "ori.png"))

        for block_idx, block in enumerate(model.transformer.encoder.layer):
            block.vis_save_dir = save_dir
            block.vis_block_idx = block_idx

        with autocast():
            output = model(x)

        if True:
            save_dir = os.path.join(save_root, "vis_outputs", f"{batch_idx}")
            concat_block_visualizations(save_dir, num_blocks=len(model.transformer.encoder.layer), alpha=0.45, cmap_name="jet")

            erank_results = compute_erank_from_q_diff(
                save_dir,
                num_blocks=len(model.transformer.encoder.layer),
                remove_cls=True,
            )

            txt_path = os.path.join(save_dir, "erank.txt")
            with open(txt_path, "w") as f:
                f.write(f"batch_idx: {batch_idx}\n")
                f.write("Effective rank computed from q_diff.pt\n")
                f.write("remove_cls: True\n\n")

                for block_idx, erank_value in erank_results.items():
                    if erank_value is None:
                        line = f"block_{block_idx}: None\n"
                    else:
                        line = f"block_{block_idx}: {erank_value:.6f}\n"

                    print(f"[batch {batch_idx}] {line.strip()}")
                    f.write(line)

            print(f"[batch {batch_idx}] Saved erank results to: {txt_path}")

        loss = criterion(output, label)
        acc1 = accuracy(output, label, topk=(1,))

        top1.update(acc1[0].item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if save_logits:
            saved_outputs.append({
                "batch_idx": batch_idx,
                "logits": output.detach().cpu(),
                "pred": output.argmax(dim=1).detach().cpu(),
                "label": label.detach().cpu(),
            })

    print("Inference:", losses, top1)
    return top1.avg, losses.avg, saved_outputs


def main(args):
    config = DATA_CONFIGS[args.dataset_name]

    args.data_path = os.path.join(args.dataset_dir, args.dataset_name)
    args.num_classes = config["num_classes"]
    args.learning_rate = config["lr"]
    args.min_lr = config["min_lr"]
    args.drop_path = config["drop_path"]
    args.warmup_lr = config["warmup_lr"]
    args.weight_decay = config["weight_decay"]

    if args.batch_size is None:
        args.batch_size = config["batch_size"]

    args.simple_aug = config["simple_aug"]

    save_root = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(save_root, exist_ok=True)

    sys.stdout = Logger(
        sys.stdout,
        os.path.join(save_root, f"{args.dataset_name}_inference.txt")
    )

    print(args)
    print(config)

    print("Building test dataloader...")
    _, test_loader = get_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        simple_aug=args.simple_aug,
    )

    print("Building model...")
    model = setup_model(args, frozen_list=("head", "scale", "shift", "hco"))
    model = load_finetuned_checkpoint(model, args.resume)

    model.to(args.device)

    acc, loss, saved_outputs = inference(
        model=model,
        test_loader=test_loader,
        device=args.device,
        max_batches=args.max_batches,
        save_logits=args.save_logits,
    )

    print(f"Final Acc@1: {acc:.4f}")
    print(f"Final Loss: {loss:.6f}")

    if args.save_logits:
        save_path = os.path.join(save_root, "inference_outputs.pt")
        torch.save(
            {
                "acc": acc,
                "loss": loss,
                "outputs": saved_outputs,
                "args": vars(args),
                "config": config,
            },
            save_path,
        )
        print(f"Saved logits/preds/labels to: {save_path}")


if __name__ == "__main__":
    args = get_args_parser()
    seed_torch(args.seed)
    args.device = torch.device(args.device)
    main(args)
