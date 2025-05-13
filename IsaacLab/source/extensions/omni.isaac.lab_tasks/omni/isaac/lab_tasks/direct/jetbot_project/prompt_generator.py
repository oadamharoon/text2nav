import json

def load_task_templates(json_path="task_templates.json", key="nav_task"):
    with open(json_path, "r") as f:
        template_dict = json.load(f)

    templates = template_dict.get(key, [])
    if not templates:
        raise ValueError(f"No templates found for task key: '{key}'")
    return templates

def generate_prompts(detections, templates):
    prompts = []
    for det in detections:
        for template in templates:
            prompt = template.replace("<OBJECT>", det["label"])
            prompts.append(prompt)
    return prompts
