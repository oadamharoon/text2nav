import json

def load_task_templates(json_path="/home/nitesh/workspace/offline_rl_test/text2nav/IsaacLab/source/standalone/workflows/skrl/task_templates.json", key="nav_task"):
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
