
This pipeline detects objects in an image, generates text prompts from a task template, and computes joint image-text embeddings using BLIP.

* * * * *

✅ 1. Install Dependencies
-------------------------

`pip install -r requirements.txt`


▶️ 2. Run the Pipeline
----------------------

Run the main script:

`python main.py`

* * * * *

📤 Output
---------

The script prints:

-   A list of generated task prompts (e.g., `"Go to the chair"`)

-   A list of joint image-text embeddings (one per prompt, `dim=768`)