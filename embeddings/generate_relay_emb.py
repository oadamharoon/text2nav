#!/usr/bin/env python3
"""
Compute `tasks_dict` from replay buffer and save it into a new pickle.
Each task is mapped properly to (episode, step) frame.
"""

import pickle
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from transformers import SiglipProcessor, SiglipModel

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_PKL  = Path("/work/mech-ai-scratch/shreyang/me592/text2nav/embeddings/data/replay_buffer.pkl")
OUTPUT_PKL = INPUT_PKL.parent / "replay_buffer_with_tasks.pkl"

# ---------------------------------------------------------------------
# Helper Functionsss
# ---------------------------------------------------------------------
def grid_cell(cx, cy, W, H):
    col = "left"   if cx <  W/3 else "centre" if cx < 2*W/3 else "right"
    row = "top"    if cy <  H/3 else "middle" if cy < 2*H/3 else "bottom"
    return (f"{row}-{col}"
            .replace("middle-centre", "centre")
            .replace("middle-", "")
            .replace("-centre", ""))

def detect_balls(img, threshold=30):
    COLOR_CENTRES = {
        'blue'  : np.array([5,   5, 192]),
        'yellow': np.array([212,212, 22]),
        'green' : np.array([22, 214, 21]),
        'red'   : np.array([196, 15, 13])
    }
    out  = img.copy()
    H, W = img.shape[:2]
    dets = []

    for colour, centre in COLOR_CENTRES.items():
        lo = np.clip(centre - threshold, 0, 255)
        hi = np.clip(centre + threshold, 0, 255)
        mask = cv2.inRange(img, lo, hi)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 10:
                continue
            cx, cy = x + w//2, y + h//2
            dets.append((colour, (x, y, w, h), grid_cell(cx, cy, W, H)))

    return out, dets

class SigLIPMatcher:
    def __init__(self, ckpt="google/siglip-so400m-patch14-384", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc   = SiglipProcessor.from_pretrained(ckpt)
        self.model  = SiglipModel.from_pretrained(ckpt).to(self.device).eval()

    @staticmethod
    def _l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    @torch.no_grad()
    def get_joint_embeddings(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        if not prompts:
            h = self.model.config.projection_dim
            return np.empty((0, h), dtype=np.float32)

        img_inp = self.proc(images=image, return_tensors="pt").to(self.device)
        img_feat = self._l2(self.model.get_image_features(**img_inp))
        txt_inp = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        txt_feat = self._l2(self.model.get_text_features(**txt_inp))
        joint = self._l2(img_feat + txt_feat)
        return joint.cpu().numpy().copy()

# ---------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------

def main():
    print("🚀 Loading replay buffer...")
    with INPUT_PKL.open("rb") as f:
        buffer = pickle.load(f)

    if not (hasattr(buffer, "observations") and isinstance(buffer.observations, dict)):
        print("❌ Buffer does not have a dict-like 'observations' field!")
        return

    rgb = buffer.observations["rgb"]
    assert rgb.ndim == 5, "Expected (episodes, steps, channels, height, width)"

    num_episodes, num_steps = rgb.shape[0], rgb.shape[1]
    print(f"✅ Loaded buffer with {num_episodes} episodes, {num_steps} steps each.")

    matcher = SigLIPMatcher()

    tasks_dict = {}

    print("🔍 Detecting balls and computing tasks...")
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        tasks_dict[ep] = {}
        for st in range(num_steps):
            img_np = rgb[ep, st]  # (3, 256, 256)
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.cpu().numpy()
            img_np = img_np.transpose(1, 2, 0).astype(np.uint8)  # (H, W, C)

            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]

            drawn_img, detections = detect_balls(img_np)

            task_list = []
            for colour, bbox, loc in detections:
                task_text = f"move to the {loc} {colour} ball"
                task_list.append({
                    "task": task_text,
                    "location": loc,
                    "colour": colour,
                    "bbox": bbox
                })

            tasks_dict[ep][st] = task_list

    # Compute embeddings
    print("💬 Computing embeddings...")
    for ep in tqdm(range(num_episodes), desc="Embedding Episodes"):
        for st in range(num_steps):
            task_list = tasks_dict[ep][st]
            if not task_list:
                continue

            img_np = rgb[ep, st]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.cpu().numpy()
            img_np = img_np.transpose(1, 2, 0).astype(np.uint8)
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]

            img_pil = Image.fromarray(img_np)
            prompts = [t["task"] for t in task_list]

            embs = matcher.get_joint_embeddings(img_pil, prompts)
            for t, e in zip(task_list, embs):
                t["embedding"] = e

    # Save into buffer
    print("💾 Saving updated buffer...")
    buffer.observations["tasks"] = tasks_dict

    with OUTPUT_PKL.open("wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Done. New pickle with tasks saved to: {OUTPUT_PKL}")

if __name__ == "__main__":
    main()
