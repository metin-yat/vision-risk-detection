import logging
import torch
from transformers import pipeline
from PIL import Image
from config import Config
import time
import os
from transformers import AutoProcessor, AutoModelForVision2Seq

class VLMAnalyzer:
    def __init__(self):
        # SmolVLM is a 'Vision2Seq' model. 
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_PATH)
        self.model = AutoModelForVision2Seq.from_pretrained(
            Config.MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

    def process_event(self, event_data):
        snapshots = event_data.get("snapshots", [])
        if not snapshots:
            return "No snapshots found."

        images = []
        metadata_summary = []

        for i, snap in enumerate(snapshots):
            # Path coming from Redis (Might be absolute: e.g., /app/scenes/event_x/1.jpg)
            original_path = snap.get("image_path")
            
            # Extracting only the filename and the event folder
            # Example: original_path -> "scenes/event_2026/1.jpg"
            # path_parts -> ["scenes", "event_2026", "1.jpg"]
            path_parts = original_path.strip("/").split("/")
            
            # Extracting the last two parts (folder + filename)
            if len(path_parts) >= 2:
                relative_path = os.path.join(path_parts[-2], path_parts[-1])
            else:
                relative_path = path_parts[-1]

            # Actual path inside the container: /app/scenes + event_x + 1.jpg
            img_path = os.path.join(Config.SCENES_DIR, relative_path)

            try:
                if not os.path.exists(img_path):
                    logging.error(f"File does not exist physically: {img_path}")
                    continue
                    
                images.append(Image.open(img_path).convert("RGB"))
                alerts = snap.get("metadata", {}).get("critical_alerts", [])
                alert_reasons = ", ".join([a['reason'] for a in alerts]) if alerts else "None"
                metadata_summary.append(f"Frame {i+1} alerts: {alert_reasons}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        # Constructing the message structure as a list
        # The processor will automatically convert this using the model's chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{Config.SYSTEM_PROMPT}\nContext: {' | '.join(metadata_summary)}"},
                    # Adding an image dictionary for each image in the list
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": "Based on these frames, is this person violating helmet safety rules: ONLY YES OR NO & WHY."}
                ]
            }
        ]

        # Apply Chat Template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Prepare Inputs
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.model.device)

        start_time = time.time()
        
        # Inference
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=Config.MAX_NEW_TOKENS)

        # Decode & Get the response only
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - start_time
        logging.info(f"Total processing latency: {latency:.2f} seconds")

        return generated_text.split("Assistant:")[-1].strip()