# embedding_pipeline.py

from .embedding_utils import SigLIPMatcher
from PIL import Image
import argparse
import torch

COLOR_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
    "pink": 4
}

INDEX_COLOR = {v: k for k, v in COLOR_INDEX.items()}

class EmbeddingPipeline:
    def __init__(self, matcher_type="siglip"):
        self.matcher = SigLIPMatcher()

    def generate(self, image: torch.tensor, prompts):
        """
        Generate joint embeddings for the given image and goal index.
        Args:
            image (torch.tensor): Input image tensor. (N, 3, 256, 256)
            goal_index (torch.tensor): Index of the goal color.
        Returns:
            embeddings (torch.tensor): Tensor of joint embeddings.
        """
        # prompts = [f"Move towards {INDEX_COLOR[int(idx)]} ball" for idx in goal_index]
        embeddings = self.matcher.get_joint_embeddings(image, prompts)
        return embeddings
