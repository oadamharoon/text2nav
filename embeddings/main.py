from detector import ObjectDetector
from prompt_generator import generate_prompts
from blip_utils import BLIPMatcher
from PIL import Image

import warnings
import os
from transformers.utils import logging as hf_logging
from prompt_generator import load_task_templates, generate_prompts

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
hf_logging.set_verbosity_error()


def run_pipeline(image_path, top_n=5):
    detector = ObjectDetector()
    detections = detector.detect(image_path, top_n=top_n)

    template = load_task_templates("task_templates.json", key="nav_task")
    prompts = generate_prompts(detections, template)

    matcher = BLIPMatcher()
    image = Image.open(image_path).convert("RGB")
    joint_embeddings = matcher.get_joint_embeddings(image, prompts)

    return prompts, joint_embeddings

if __name__ == "__main__":
    image_path = "data/env.png"
    actions, embeddings = run_pipeline(image_path)

    print("Actions:")
    for a in actions:
        print(f"- {a}")

    print("Joint Embeddings:")
    for i, vec in enumerate(embeddings):
        print(f"{actions[i]} => Embedding dim: {len(vec)}")