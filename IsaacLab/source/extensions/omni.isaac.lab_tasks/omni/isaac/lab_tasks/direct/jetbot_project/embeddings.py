from .detector import ObjectDetector
from .prompt_generator import load_task_templates, generate_prompts
from .embedding_utils import BLIPMatcher, SigLIPMatcher
from PIL import Image
import os, warnings
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
hf_logging.set_verbosity_error()

def run_pipeline(image, matcher_type="siglip", top_n=5):
    detector = ObjectDetector()
    detections = detector.detect(image, top_n=top_n)

    template = load_task_templates("task_templates.json", key="nav_task")
    prompts = generate_prompts(detections, template)

    image = Image.fromarray(image).convert("RGB")
    matcher = SigLIPMatcher() if matcher_type == "siglip" else BLIPMatcher()
    embeddings = matcher.get_joint_embeddings(image, prompts)

    return prompts, embeddings