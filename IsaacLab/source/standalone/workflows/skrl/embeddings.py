# embedding_pipeline.py

from detector import ObjectDetector
from prompt_generator import load_task_templates, generate_prompts
from embedding_utils import BLIPMatcher, SigLIPMatcher
from PIL import Image
import argparse
import torch

COLOR_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
    "magenta": 4
}

class EmbeddingPipeline:
    def __init__(self):
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

def generate_relative_prompts(y, goal_index, INDEX_COLOR, y_thresh=0.5, base="Move toward the ball."):
    """
    goal_vec_robot: (B, 3) tensor, goal vector in robot frame
    y_thresh: float, threshold for deciding clear left/right
    Returns: List of strings (prompts)
    """
    prompts = []

    for i in range(len(y)):
        if y[i] > y_thresh:
            prompts.append(f"The target is {INDEX_COLOR[int(goal_index[i])]} ball which is to your left. {base}")
        elif y[i] < -y_thresh:
            prompts.append(f"The target is {INDEX_COLOR[int(goal_index[i])]} ball which is to your right. {base}")
        else:
            prompts.append(f"The target is {INDEX_COLOR[int(goal_index[i])]} ball which is straight ahead. {base}")
    
    return prompts   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--matcher", type=str, choices=["blip", "siglip"], default="siglip")
    parser.add_argument("--goal", type=int, choices=range(0, 5), help="Goal index (0=red, 1=green, 2=blue, 3=yellow, 4=white)")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(matcher_type=args.matcher)
    actions, embeddings = pipeline.generate(args.image, goal_index=args.goal)

    print("Actions:")
    for a in actions:
        print(f"- {a}")

    print("\nJoint Embeddings:")
    for i, vec in enumerate(embeddings):
        print(f"{actions[i]} => Embedding dim: {len(vec)}")