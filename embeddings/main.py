# main.py
from detector import ObjectDetector
from prompt_generator import load_task_templates, generate_prompts
from embedding_utils import BLIPMatcher, SigLIPMatcher
from PIL import Image
import argparse, os, warnings
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
hf_logging.set_verbosity_error()

def run_pipeline(image_path, matcher_type="siglip", top_n=5):
    detector = ObjectDetector()
    detections = detector.detect(image_path, top_n=top_n)

    template = load_task_templates("task_templates.json", key="nav_task")
    prompts = generate_prompts(detections, template)

    image = Image.open(image_path).convert("RGB")
    matcher = SigLIPMatcher() if matcher_type == "siglip" else BLIPMatcher()
    embeddings = matcher.get_joint_embeddings(image, prompts)

    return prompts, embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--matcher", type=str, choices=["blip", "siglip"], default="siglip", help="Matcher model")
    args = parser.parse_args()

    actions, embeddings = run_pipeline(args.image, matcher_type=args.matcher)

    print("Actions:")
    for a in actions:
        print(f"- {a}")

    print("\nJoint Embeddings:")
    for i, vec in enumerate(embeddings):
        print(f"{actions[i]} => Embedding dim: {len(vec)}")