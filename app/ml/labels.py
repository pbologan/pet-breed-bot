import json
from pathlib import Path
from typing import Dict, List


class LabelManager:
    """Manages label mappings for pet breed classification"""

    def __init__(self, label_map_path: Path):
        """
        Initialize label manager

        Args:
            label_map_path: Path to JSON file containing label mappings
        """
        self.label_map_path = label_map_path
        self.idx_to_label: Dict[int, str] = {}
        self.label_to_idx: Dict[str, int] = {}
        self.load_labels()

    def load_labels(self):
        """Load label mappings from JSON file"""
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found at {self.label_map_path}")

        with open(self.label_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Support both formats: {"0": "label"} or {"label_to_idx": {...}, "idx_to_label": {...}}
        if "idx_to_label" in data:
            self.idx_to_label = {int(k): v for k, v in data["idx_to_label"].items()}
            self.label_to_idx = data["label_to_idx"]
        else:
            self.idx_to_label = {int(k): v for k, v in data.items()}
            self.label_to_idx = {v: int(k) for k, v in data.items()}

    def get_label(self, idx: int) -> str:
        """Get label name from index"""
        return self.idx_to_label.get(idx, "unknown")

    def get_index(self, label: str) -> int:
        """Get index from label name"""
        return self.label_to_idx.get(label, -1)

    def get_all_labels(self) -> List[str]:
        """Get list of all label names"""
        return [self.idx_to_label[i] for i in sorted(self.idx_to_label.keys())]

    def num_classes(self) -> int:
        """Get total number of classes"""
        return len(self.idx_to_label)

    def save_labels(self, output_path: Path = None):
        """Save current label mappings to JSON file"""
        if output_path is None:
            output_path = self.label_map_path

        data = {
            "idx_to_label": {str(k): v for k, v in self.idx_to_label.items()},
            "label_to_idx": self.label_to_idx
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def create_sample_label_map(output_path: Path):
    """
    Create a sample label map for testing purposes
    This should be replaced with actual labels from your dataset
    """
    sample_labels = {
        "idx_to_label": {
            "0": "golden_retriever",
            "1": "labrador_retriever",
            "2": "german_shepherd",
            "3": "beagle",
            "4": "bulldog",
            "5": "siamese_cat",
            "6": "persian_cat",
            "7": "maine_coon",
            "8": "british_shorthair",
            "9": "sparrow",
            "10": "pigeon",
            "11": "parrot"
        }
    }

    # Add reverse mapping
    sample_labels["label_to_idx"] = {v: int(k) for k, v in sample_labels["idx_to_label"].items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_labels, f, ensure_ascii=False, indent=2)
