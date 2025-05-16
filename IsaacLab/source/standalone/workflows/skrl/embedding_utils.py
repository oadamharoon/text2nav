# embedding_utils.py

import torch
from PIL import Image
from transformers import BlipProcessor, BlipModel, SiglipProcessor, SiglipModel

class BLIPMatcher:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipModel.from_pretrained(model_name).to(self.device).eval()

    def get_joint_embeddings(self, image: Image.Image, prompts: list[str]):
        embeddings = []
        for prompt in prompts:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                emb = outputs.text_model_output.last_hidden_state[:, 0, :]
                embeddings.append(emb.squeeze(0).cpu().numpy())
        return embeddings


class SigLIPMatcher:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()

    def _l2(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def get_joint_embeddings(self, image: Image.Image, prompts: list[str]):
        if not prompts:
            return []
        img_inp = self.processor(images=image, return_tensors="pt").to(self.device)
        img_feat = self._l2(self.model.get_image_features(**img_inp))
        txt_inp = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        txt_feat = self._l2(self.model.get_text_features(**txt_inp))
        joint = self._l2(img_feat + txt_feat)
        return joint.cpu().numpy()