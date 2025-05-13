# embedding_pipeline.py

from detector import ObjectDetector
from prompt_generator import load_task_templates, generate_prompts
from embedding_utils import BLIPMatcher, SigLIPMatcher
from PIL import Image
import argparse

COLOR_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
    "pink": 4
}

INDEX_COLOR = {v: k for k, v in COLOR_INDEX.items()}

class EmbeddingPipeline:
    def __init__(self, matcher_type="siglip", task_template_path="/home/nitesh/workspace/offline_rl_test/text2nav/embeddings/task_templates.json", task_key="nav_task", top_n=5):
        self.detector = ObjectDetector()
        self.matcher = SigLIPMatcher() if matcher_type == "siglip" else BLIPMatcher()
        self.template = load_task_templates(task_template_path, key=task_key)
        self.top_n = top_n

    def generate(self, image_path, goal_index=None):
        detections = self.detector.detect(image_path, top_n=self.top_n)
        prompts = generate_prompts(detections, self.template)
        image = Image.open(image_path).convert("RGB")
        embeddings = self.matcher.get_joint_embeddings(image, prompts)

        if goal_index is not None:
            filtered_prompts = []
            filtered_embeddings = []
            for i, d in enumerate(detections):
                color = d.get("colour", "").lower()
                if COLOR_INDEX.get(color) == goal_index:
                    filtered_prompts.append(prompts[i])
                    filtered_embeddings.append(embeddings[i])
            prompts = filtered_prompts
            embeddings = filtered_embeddings
        
        if not prompts:
            prompts = [f"{INDEX_COLOR[goal_index]} not in frame, move around to see"]
            embeddings = self.matcher.get_joint_embeddings(image, prompts)

        return prompts, embeddings
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--matcher", type=str, choices=["blip", "siglip"], default="siglip")
    parser.add_argument("--goal", type=int, choices=range(0, 5), help="Goal index (0=red, 1=green, 2=blue, 3=yellow, 4=pink)")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(matcher_type=args.matcher)
    actions, embeddings = pipeline.generate(args.image, goal_index=args.goal)

    print("Actions:")
    for a in actions:
        print(f"- {a}")

    print("\nJoint Embeddings:")
    for i, vec in enumerate(embeddings):
        print(f"{actions[i]} => Embedding dim: {len(vec)}")