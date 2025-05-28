#!/usr/bin/env python3
"""
Compute goal-based embeddings from replay buffer and save into a new pickle.
Embeddings are generated from RGB frames and text tasks matched with goal_index.
"""

# import pickle
# import pickle
import pickle
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import requests
import requests

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_PKL  = Path("/work/mech-ai-scratch/nitesh/workspace/text2nav/replay_buffer_5.pkl")
OUTPUT_PKL = INPUT_PKL.parent / "replay_buffer_with_embeddings_5.pkl"

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
    pass

# ---------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------

def main():
    matcher = SigLIPMatcher()
    matcher = SigLIPMatcher()
    print("🚀 Loading replay buffer...")
    with INPUT_PKL.open("rb") as f:
        buffer = pickle.load(f)

    # test_limit = 1
    print(f"Buffer size: {len(buffer.buffer)}")
    
    observations = np.array([experience[0] for experience in buffer.buffer])#[:test_limi
    actions = np.array([experience[1] for experience in buffer.buffer])#[:test_limit]
    rewards = np.array([experience[2] for experience in buffer.buffer])#[:test_limit]
    # next_observations = np.array([experience[3] for experience in buffer.buffer])#[:test_limit]
    dones = np.array([experience[4] for experience in buffer.buffer])#[:test_limit]

    rgb = np.array([obs['rgb'] for obs in observations]).transpose(0, 1, 4, 2, 3)  # (episodes, steps, channels, height, width)
    # next_rgb = np.array([obs['rgb'] for obs in next_observations]).transpose(0, 1, 4, 2, 3)  # (episodes, steps, channels, height, width)
    goal_index = np.array([obs['goal_index'] for obs in observations])  # (episodes, steps, 1)

    if rgb is None or goal_index is None:
        print("❌ Missing required data in buffer: rgb, next_rgb or goal_index.")
        return

    num_episodes, num_steps = rgb.shape[0], rgb.shape[1]
    print(f"✅ Loaded buffer with {num_episodes} episodes, {num_steps} steps each.")

    print("💬 Computing joint embeddings...")
    for ep in tqdm(range(num_episodes), desc="Embedding Episodes"):
        for st in range(num_steps):
            task_list = tasks_dict[ep][st]

            img_np = rgb[ep, st]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.cpu().numpy()
            img_np = img_np.transpose(1, 2, 0).astype(np.uint8)
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]

            img_pil = Image.fromarray(img_np)

            if not task_list:
                # No object in frame → Add "move around" task
                prompts = ["No balls in frame, move around"]
                embs = matcher.get_joint_embeddings(img_pil, prompts)
                tasks_dict[ep][st] = [{
                    "task": prompts[0],
                    "location": None,
                    "colour": None,
                    "bbox": None,
                    "embedding": embs[0]
                }]
            else:
                prompts = [t["task"] for t in task_list]
                embs = matcher.get_joint_embeddings(img_pil, prompts)
                for t, e in zip(task_list, embs):
                    t["embedding"] = e

            if not task_list:
                # No object in frame → Add "move around" task
                prompts = ["No balls in frame, move around"]
                embs = matcher.get_joint_embeddings(img_pil, prompts)
                tasks_dict[ep][st] = [{
                    "task": prompts[0],
                    "location": None,
                    "colour": None,
                    "bbox": None,
                    "embedding": embs[0]
                }]
            else:
                prompts = [t["task"] for t in task_list]
                embs = matcher.get_joint_embeddings(img_pil, prompts)
                for t, e in zip(task_list, embs):
                    t["embedding"] = e

    # Compute final embeddings based on goal_index
    print("📦 Computing goal-based embeddings...")
    embedding_size = matcher.model.config.text_config.projection_size
    print(f"Embedding size: {embedding_size}")
    color_index = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3,
        "magenta": 4
    }

    embeddings = compute_goal_embeddings(rgb, tasks_dict, goal_index, color_index, embedding_size)

    buffer_with_embeddings = []
    buffer_with_embeddings.append(embeddings)
    buffer_with_embeddings.append(actions)
    buffer_with_embeddings.append(rewards)
    buffer_with_embeddings.append(dones)

    # Save
    print("💾 Saving final buffer with embeddings...")
    with OUTPUT_PKL.open("wb") as f:
        pickle.dump(buffer_with_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(buffer_with_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Done. Embeddings saved to: {OUTPUT_PKL}")
    print(f"✅ Done. Embeddings saved to: {OUTPUT_PKL}")

if __name__ == "__main__":
    main()