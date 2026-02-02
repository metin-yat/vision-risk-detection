import logging
import torch
import os
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from config import Config

# Standard logging setup
logger = logging.getLogger(__name__)

class VLMAnalyzer:
    def __init__(self):
        """
        Initializes the model with bfloat16 for memory efficiency.
        """
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_PATH)
        self.model = AutoModelForVision2Seq.from_pretrained(
            Config.MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

    def _build_prompt(self, p_id, frames_info, is_bulk=False):
        """
        Her kareyi bağımsız inceler. Kırmızı kutudaki insan varlığını ve 
        kask yokluğunu (violation) kesin bir dille sorgular.
        """
        system_instructions = (
            "You are a Safety Compliance Expert. Your ONLY task is to inspect the RED BOX for a helmet violation.\n"
            "DEFINITION: VIOLATION = Human head is visible WITHOUT a helmet.\n"
            "DEFINITION: SAFE = Human is wearing a helmet OR no human is in the box.\n"
        )

        few_shot = (
            "EXAMPLE 1 (Violation):\n"
            "Red Box: P1 [200, 400, 800, 600]\n"
            "Analysis:\n"
            "1. Focus: Inspecting red box P1.\n"
            "2. Human Presence: Yes, person identified.\n"
            "3. Helmet Check: No helmet. Bare head is visible.\n"
            "4. Decision: DECISION: YES\n"
            "5. Reason: Human head is exposed; no helmet detected.\n"
        )

        # Tekil kare analizi için context
        bbox = frames_info[0]['bbox']
        context = (
            f"CURRENT TASK: Look at the red box labeled {p_id} at coordinates {bbox}.\n"
            "Is there a person with a bare head (no helmet) in this exact box?"
        )

        instruction = (
            "Output Format:\n"
            "1. Focus: (Target red box)\n"
            "2. Human Presence: (Yes/No)\n"
            "3. Helmet Check: (Specifically check for helmet or bare head)\n"
            "4. Decision: (Write 'DECISION: YES' for violation, 'DECISION: NO' if safe)\n"
            "5. Reason: (Brief explanation)"
        )

        return f"{system_instructions}\n{few_shot}\n{context}\n{instruction}"

    def _get_vlm_response(self, images, prompt):
        """
        Triggers VLM inference and returns raw text.
        """
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=formatted_prompt, images=images, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=Config.MAX_NEW_TOKENS)
        
        raw_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return raw_output.split("Assistant:")[-1].strip()

    
    def process_event(self, event_data):
        event_id = event_data.get("event_id", "")
        
        # 1. EOS veya geçersiz eventleri sessizce atla
        if not event_data.get("snapshots"):
            return None

        snapshots = event_data.get("snapshots", [])
        
        # 2. DOSYA KONTROLÜ: Eğer dosyalardan biri bile eksikse direkt çık (Log yok)
        for snap in snapshots:
            if not os.path.exists(snap["image_path"]):
                return None

        unique_p_ids = {m["p_id"] for s in snapshots for m in s["metadata"]["compliance_map"]}
        final_results = []

        for p_id in unique_p_ids:
            is_suspicious = any(
                m["status"] == "violation" 
                for s in snapshots 
                for m in s["metadata"]["compliance_map"] if m["p_id"] == p_id
            )

            if is_suspicious:
                yes_votes = 0
                reasons = []
                evidence_img = None

                for snap in snapshots:
                    # 'next' hatalarını önlemek için varsayılan değer (None) ekledik
                    meta = next((m for m in snap["metadata"]["compliance_map"] if m["p_id"] == p_id), None)
                    if not meta: continue # Bu karede bu kişi yoksa geç
                    
                    alert = next((a for a in snap["metadata"]["critical_alerts"] if a["id"] == p_id), None)
                    
                    img = Image.open(snap["image_path"]).convert("RGB")
                    info = [{"bbox": alert["norm_bbox"] if alert else [0,0,0,0], "overlap": meta["overlap"]}]
                    res = self._get_vlm_response([img], self._build_prompt(p_id, info))
                    
                    if "DECISION: YES" in res.upper():
                        yes_votes += 1
                        evidence_img = snap["image_path"]
                    
                    reasons.append(res.split("Reason:")[-1].strip() if "Reason:" in res else "No reason")

                if reasons: # Eğer en az bir kare işlenebildiyse
                    final_status = "VIOLATION" if yes_votes >= 2 else "SAFE"
                    final_results.append({
                        "p_id": p_id, "status": final_status, 
                        "reason": reasons[-1], "evidence": evidence_img or snapshots[0]["image_path"]
                    })
            else:
                # Branch B: Bulk Verifier (Dosya kontrolü yukarıda yapıldığı için güvenli)
                imgs = [Image.open(s["image_path"]).convert("RGB") for s in snapshots]
                res = self._get_vlm_response(imgs, self._build_prompt(p_id, [], is_bulk=True))
                
                final_status = "VIOLATION" if "DECISION: YES" in res.upper() else "SAFE"
                final_results.append({
                    "p_id": p_id, "status": final_status, 
                    "reason": res.split("Reason:")[-1].strip() if "Reason:" in res else "Verified safe",
                    "evidence": snapshots[0]["image_path"]
                })

        return final_results if final_results else None

        return final_results