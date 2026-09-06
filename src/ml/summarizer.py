from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

class CaseSummarizer:
    def __init__(self):
        print(f"Loading summarization model: {MODEL_NAME} (first run downloads ~300MB)")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.eval()

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        if not text or len(text.strip()) < 100:
            return "Text too short to summarize."

        truncated = text[:4000]

        try:
            inputs = self.tokenizer(
                truncated,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return f"Summarization failed: {e}"