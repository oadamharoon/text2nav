#!/usr/bin/env python3
"""
Compute goal-based embeddings from replay buffer and save into a new pickle.
Embeddings are generated from RGB frames and text tasks matched with goal_index.
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
OUTPUT_PKL = INPUT_PKL.parent / "replay_buffer_with_embeddings.pkl"

# ---------------------------------------------------------------------
# Helper Functions
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

def compute_goal_embeddings(rgb_array, tasks_dict, goal_index, color_index, embedding_size):
    """
    Compute embeddings based on tasks matching the goal color for each frame.
    Returns a numpy array of shape (episodes, steps, embedding_size).
    """
    num_episodes, num_steps = rgb_array.shape[0], rgb_array.shape[1]
    all_embeddings = []

    for batch_id in range(num_episodes):
        batch_embeddings = []
        task_batch = tasks_dict[batch_id]
        goal_batch = goal_index[batch_id]

        for step_id in range(num_steps):
            task_list = task_batch[step_id]
            embedding = None
            if task_list:
                for task in task_list:
                    if color_index.get(task.get("colour", None), None) == goal_batch[step_id][0]:
                        embedding = task.get("embedding", None)
            if embedding is None:
                embedding = np.zeros(embedding_size)

            batch_embeddings.append(embedding)

        all_embeddings.append(np.array(batch_embeddings))

    return np.array(all_embeddings)

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
    next_rgb = buffer.next_observations.get("rgb", None)
    goal_index = buffer.observations.get("goal_index", None)

    if rgb is None or next_rgb is None or goal_index is None:
        print("❌ Missing required data in buffer: rgb, next_rgb or goal_index.")
        return

    num_episodes, num_steps = rgb.shape[0], rgb.shape[1]
    print(f"✅ Loaded buffer with {num_episodes} episodes, {num_steps} steps each.")

    matcher = SigLIPMatcher()
    tasks_dict = {}

    print("🔍 Detecting balls and computing tasks...")
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        tasks_dict[ep] = {}
        for st in range(num_steps):
            img_np = rgb[ep, st]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.cpu().numpy()
            img_np = img_np.transpose(1, 2, 0).astype(np.uint8)
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]

            _, detections = detect_balls(img_np)

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

    print("💬 Computing joint embeddings...")
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

    # Compute final embeddings based on goal_index
    print("📦 Computing goal-based embeddings...")
    embedding_size = matcher.model.config.projection_dim
    color_index = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3,
        "white": 4
    }

    embeddings = compute_goal_embeddings(rgb, tasks_dict, goal_index, color_index, embedding_size)
    next_embeddings = compute_goal_embeddings(next_rgb, tasks_dict, goal_index, color_index, embedding_size)

    buffer.observations["embeddings"] = embeddings
    buffer.observations["next_embeddings"] = next_embeddings
    buffer.observations.pop("tasks", None)

    # Save
    print("💾 Saving final buffer with embeddings...")
    with OUTPUT_PKL.open("wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Done. Embeddings saved to: {OUTPUT_PKL}")

if __name__ == "__main__":
    main()