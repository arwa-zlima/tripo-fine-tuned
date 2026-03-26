'''
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
'''
"""
Zero-shot furniture classifier using CLIP.

Classifies an input image into one of the 10 furniture categories and
returns the category key that matches the LoRAAdapterManager adapter names
used in app.py and the team assignment table.

BUG FIXES applied:
  1. device="cuda:0" hard-coded → crash on CPU-only machines. Now auto-detects.
  2. classify() accepted only a file path string → broke when called from
     app.py which passes a PIL.Image directly. Now accepts both.
  3. No error handling when classifier returns an unexpected label → silent
     KeyError. Added a safe fallback.
  4. CLIP candidate labels used natural-language phrases ("a bed") but
     app.py called classify_image() expecting plain category keys ("bed").
     The label_to_category mapping worked for FurnitureRouter.classify() but
     was bypassed in app.py's classify_image() which passed CATEGORIES
     (plain keys) directly to the pipeline. Unified so the standalone class
     and the app helper both use the same label set and mapping.
  5. No confidence threshold / low-confidence warning. Added optional
     threshold parameter so callers can detect ambiguous images.
  6. __init__ did not expose the classifier as a reusable callable compatible
     with the LoRAAdapterManager.route_to_adapter() API (which expects a
     callable that takes an image tensor and returns a category string).
     Added __call__ and a get_router_fn() adapter method.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import pipeline


# ---------------------------------------------------------------------------
# Canonical label set — must match the keys used in LoRAAdapterManager
# and the CATEGORIES list in app.py.
# ---------------------------------------------------------------------------
CLIP_CANDIDATE_LABELS = [
    "a bed",
    "a chair",
    "a sofa",
    "a swivel office chair",
    "a dining table",       # BUG FIX 7: "a table" is ambiguous in CLIP;
    "a cabinet",            # "a dining table" scores higher for furniture.
    "a bookshelf",
    "a desk",
    "a bench",
    "a wardrobe",
]

LABEL_TO_CATEGORY: Dict[str, str] = {
    "a bed":               "bed",
    "a chair":             "chair",
    "a sofa":              "sofa",
    "a swivel office chair": "swivelchair",
    "a dining table":      "table",
    "a cabinet":           "cabinet",
    "a bookshelf":         "bookshelf",
    "a desk":              "desk",
    "a bench":             "bench",
    "a wardrobe":          "wardrobe",
}


class FurnitureRouter:
    """Zero-shot CLIP-based furniture category classifier.

    Args:
        confidence_threshold: If the top prediction score is below this
            value the classifier still returns a result but also sets the
            ``low_confidence`` flag in classify(). Useful to surface
            ambiguous inputs in the UI.
        device: "cuda", "cpu", or None (auto-detect).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        # BUG FIX 1: original hard-coded device=0 (GPU index), crashing on
        # CPU-only machines and on machines where CUDA is unavailable at import
        # time. Auto-detect instead.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP classifier on {device}...")
        self.classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=device,
        )

        self.clip_labels = CLIP_CANDIDATE_LABELS
        self.label_to_category = LABEL_TO_CATEGORY
        self.confidence_threshold = confidence_threshold

        print("Router ready!")

    # ------------------------------------------------------------------
    # Core classify method
    # ------------------------------------------------------------------

    def classify(
        self,
        image: Union[str, "Image.Image"],
    ) -> Tuple[str, float, bool, Dict[str, float]]:
        """Classify a furniture image.

        Args:
            image: A file path string OR a PIL.Image object.
                   BUG FIX 2: original only accepted file paths; app.py
                   passes PIL.Image objects directly → TypeError.

        Returns:
            category      (str):  Category key, e.g. "chair"
            confidence    (float): Top score in [0, 1]
            low_confidence(bool):  True when confidence < threshold
            all_scores    (dict):  {category: score} for all categories
        """
        # BUG FIX 2: accept both path strings and PIL.Image objects
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(
                f"image must be a file path str or PIL.Image, got {type(image)}"
            )

        results = self.classifier(pil_image, candidate_labels=self.clip_labels)

        top_label = results[0]["label"]
        confidence = results[0]["score"]

        # BUG FIX 3: original had no fallback for unmapped labels → KeyError
        category = self.label_to_category.get(top_label)
        if category is None:
            # Fallback: pick closest key by partial string match, else "chair"
            for label, cat in self.label_to_category.items():
                if any(word in top_label for word in label.split()):
                    category = cat
                    break
            if category is None:
                category = "chair"

        # Build scores dict keyed by category (not raw CLIP label)
        all_scores: Dict[str, float] = {}
        for r in results:
            cat = self.label_to_category.get(r["label"])
            if cat:
                all_scores[cat] = r["score"]

        low_confidence = confidence < self.confidence_threshold

        return category, confidence, low_confidence, all_scores

    # ------------------------------------------------------------------
    # Convenience: make the router directly callable
    # BUG FIX 6: LoRAAdapterManager.route_to_adapter() expects a callable
    # that takes an image and returns a category string. Added __call__.
    # ------------------------------------------------------------------

    def __call__(self, image: Union[str, "Image.Image"]) -> str:
        """Return just the category string (for use as a router callable)."""
        category, _, _, _ = self.classify(image)
        return category

    def get_router_fn(self):
        """Return a plain function compatible with route_to_adapter().

        Usage:
            from tsr.models.transformer.lora import route_to_adapter
            router_fn = FurnitureRouter().get_router_fn()
            route_to_adapter(image_tensor, router_fn, model, adapter_manager)
        """
        def _route(image):
            return self(image)
        return _route


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    router = FurnitureRouter()

    # From file path
    category, conf, low_conf, scores = router.classify("examples/chair.png")
    print(f"Category   : {category}")
    print(f"Confidence : {conf * 100:.1f}%")
    if low_conf:
        print("WARNING: low confidence prediction")
    print("All scores :")
    for cat, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {cat:15s} {score * 100:.1f}%")

    # From PIL image (as app.py does)
    img = Image.open("examples/chair.png")
    category2 = router(img)
    print(f"\nVia __call__: {category2}")
