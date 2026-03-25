from transformers import pipeline
from PIL import Image

class FurnitureRouter:
    def __init__(self):
        print("Loading CLIP classifier...")
        self.classifier = pipeline(
            "zero-shot-image-classification", 
            model="openai/clip-vit-base-patch32",
            device=0  # GPU
        )
        
        self.clip_labels = [
            "a bed", "a chair", "a sofa", "a swivel office chair",
            "a table", "a cabinet", "a bookshelf", "a desk",
            "a bench", "a wardrobe"
        ]
        
        self.label_to_category = {
            "a bed": "bed", "a chair": "chair", "a sofa": "sofa",
            "a swivel office chair": "swivelchair", "a table": "table",
            "a cabinet": "cabinet", "a bookshelf": "bookshelf",
            "a desk": "desk", "a bench": "bench", "a wardrobe": "wardrobe",
        }
        print("Router ready!")
    
    def classify(self, image_path):
        image = Image.open(image_path).convert("RGB")
        results = self.classifier(image, candidate_labels=self.clip_labels)
        
        top_label = results[0]['label']
        confidence = results[0]['score']
        category = self.label_to_category[top_label]
        
        return category, confidence

# Usage:
# router = FurnitureRouter()
# category, conf = router.classify("Chairs/example1.png")
# print(f"{category}: {conf*100:.1f}%")