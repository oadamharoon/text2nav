# blip_utils.py

import torch
from PIL import Image
from transformers import BlipProcessor, BlipModel

class BLIPMatcher:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BLIP model on {self.device}...")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_joint_embeddings(self, image: Image.Image, prompts: list[str]):
        embeddings = []

        for prompt in prompts:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                # 👇 Correct access to text hidden states
                joint_emb = outputs.text_model_output.last_hidden_state[:, 0, :]  # shape: [1, hidden_dim]
                embeddings.append(joint_emb.squeeze(0).cpu().numpy())

        return embeddings
