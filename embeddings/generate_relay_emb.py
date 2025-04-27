import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import cv2 

# ---------------------------------------------------------------------------
#  Detection helpers
# ---------------------------------------------------------------------------

COLOR_CENTRES = {
    "blue":   np.array([5,   5, 192]),
    "yellow": np.array([212,212, 22]),
    "green":  np.array([22, 214, 21]),
    "red":    np.array([196, 15, 13]),
}
THRESH = 32  # strictness for cv2.inRange


def grid_cell(cx: int, cy: int, W: int, H: int) -> str:
    col = "left"   if cx <  W/3 else "centre" if cx < 2*W/3 else "right"
    row = "top"    if cy <  H/3 else "middle" if cy < 2*H/3 else "bottom"
    return (
        f"{row}-{col}"
        .replace("middle-centre", "centre")
        .replace("middle-", "")
        .replace("-centre", "")
    )


def detect_balls(rgb: np.ndarray):
    """Return list[(colour, (x,y,w,h), location_label)]. *rgb* is H×W×3."""
    dets = []
    H, W = rgb.shape[:2]
    for colour, centre in COLOR_CENTRES.items():
        lo = np.clip(centre - THRESH, 0, 255)
        hi = np.clip(centre + THRESH, 0, 255)
        mask = cv2.inRange(rgb, lo, hi)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 10:
                continue
            cx, cy = x + w // 2, y + h // 2
            dets.append((colour, (x, y, w, h), grid_cell(cx, cy, W, H)))
    return dets


# ---------------------------------------------------------------------------
#  SIGLIP embedding helper
# ---------------------------------------------------------------------------

from transformers import SiglipProcessor, SiglipModel        # new imports


class SigLIPMatcher:
    def __init__(self,
                 ckpt="google/siglip-so400m-patch14-384",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc   = SiglipProcessor.from_pretrained(ckpt)
        self.model  = SiglipModel.from_pretrained(ckpt).to(self.device).eval()

    @staticmethod
    def _l2(x):          # row-wise normalisation
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    @torch.no_grad()
    def get_joint_embeddings(self, image: Image.Image,
                             prompts: list[str]) -> np.ndarray:
        if not prompts:
            h = self.model.config.projection_dim
            return np.empty((0, h), dtype=np.float32)

        # 1) IMAGE → GPU *before* forward
        img_inp = self.proc(images=image, return_tensors="pt").to(self.device)
        img_feat = self._l2(self.model.get_image_features(**img_inp))          # [1, D]

        # 2) TEXT  → GPU *before* forward
        txt_inp = self.proc(text=prompts, return_tensors="pt",
                            padding=True).to(self.device)
        txt_feat = self._l2(self.model.get_text_features(**txt_inp))           # [N, D]

        # 3) Fusion
        joint = self._l2(img_feat + txt_feat)                                  # [N, D]
        return joint.cpu().numpy().copy()


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def annotate_buffer(buf, matcher: SigLIPMatcher, obs_key: str = "rgb"):
    if "tasks" not in buf.observations:
        # create parallel structure (same length as other obs)
        buf.observations["tasks"] = [None] * len(buf.observations[obs_key])

    for idx, img_tensor in enumerate(buf.observations[obs_key]):
        # ─── convert to H×W×3 uint8 ─────────────────────────────────────────
        img_np = (
            img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
        )
        img_np = img_np.transpose(1, 2, 0) if img_np.shape[0] in (3, 4) else img_np
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        # ─── detections ─────────────────────────────────────────────────────
        detections = detect_balls(img_np)
        if not detections:
            buf.observations["tasks"][idx] = []
            continue

        img_pil = Image.fromarray(img_np)
        tasks_for_frame = []

        for colour, _bbox, loc in detections:
            prompt = f"Go to the {colour} ball"
            emb = matcher.embed(img_pil, prompt).astype("float16")
            tasks_for_frame.append({
                "task": prompt,
                "location": loc,
                "embedding": emb.tolist(),  # JSON / pickle safe
            })

        buf.observations["tasks"][idx] = tasks_for_frame
        if idx % 1000 == 0:
            print(f"Annotated frame {idx}/{len(buf.observations[obs_key])}")

    return buf


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="path to replay_buffer.pkl")
    parser.add_argument("--out", dest="out", required=True, help="output .pkl or .pkl.gz")
    parser.add_argument("--obs_key", default="rgb", help="key of RGB frames in observations dict")
    args = parser.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    print(f"Loading replay buffer from {in_path}…")
    with open(in_path, "rb") as f:
        buffer = pickle.load(f)

    matcher = SigLIPMatcher()
    buffer = annotate_buffer(buffer, matcher, obs_key=args.obs_key)

    print(f"Saving augmented buffer to {out_path}…")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".gz":
        with gzip.open(out_path, "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(out_path, "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ✅")


if __name__ == "__main__":
    main()
