from pathlib import Path
import shutil
import json
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

SYNTHETIC_DIR = DATASETS_DIR / "synthetic-bags" / "ImageClassesCombinedWithCOCOAnnotations"
COCO_DIR = DATASETS_DIR / "_synthetic-bags_processed"
IMAGES_DIR = COCO_DIR / "images"
ANNOTATIONS_DIR = COCO_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

console.print("\n[bold cyan]Processing Synthetic Bags dataset[/bold cyan]\n")

annotations_file = SYNTHETIC_DIR / "coco_instances.json"
if not annotations_file.exists():
    console.print("[red]Error: coco_instances.json not found[/red]")
    exit(1)

console.print("[blue]Loading annotations...[/blue]")
with open(annotations_file) as f:
    annotations_data = json.load(f)

console.print("[blue]Copying images from images_raw...[/blue]")
images_raw_dir = SYNTHETIC_DIR / "images_raw"

all_images = list(images_raw_dir.glob("*"))
file_name_mapping = {}

copied_count = 0
for img_file in tqdm(all_images, desc="Copying images",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
    if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        normalized_name = f"{img_file.stem}".lower() + img_file.suffix.lower()
        new_path = IMAGES_DIR / normalized_name
        
        shutil.copy(img_file, new_path)
        
        file_name_mapping[img_file.name] = normalized_name
        copied_count += 1

console.print(f"[green]Copied {copied_count} images[/green]\n")

console.print("[blue]Updating image paths in annotations...[/blue]")
updated_images = []
missing_images = []

if "images" in annotations_data:
    for img_info in annotations_data["images"]:
        original_file_name = img_info.get("file_name", "")
        file_name_only = Path(original_file_name).name
        
        normalized_name = None
        if file_name_only in file_name_mapping:
            normalized_name = file_name_mapping[file_name_only]
        else:
            for orig, norm in file_name_mapping.items():
                if orig.lower() == file_name_only.lower():
                    normalized_name = norm
                    break
        
        if normalized_name:
            img_info["file_name"] = normalized_name
            updated_images.append(img_info)
        else:
            missing_images.append(original_file_name)

if missing_images:
    console.print(f"[yellow]Warning: {len(missing_images)} images from annotations not found[/yellow]")

if "annotations" in annotations_data:
    valid_image_ids = {img["id"] for img in updated_images}
    updated_annotations = [ann for ann in annotations_data["annotations"] if ann["image_id"] in valid_image_ids]
    annotations_data["annotations"] = updated_annotations
    console.print(f"[green]Kept {len(updated_annotations)} annotations for {len(updated_images)} images[/green]\n")

annotations_data["images"] = updated_images

output_json = ANNOTATIONS_DIR / "annotations.json"
with open(output_json, "w") as f:
    json.dump(annotations_data, f, indent=2)

console.print("[green]Annotations saved[/green]\n")

table = Table(title="Synthetic Bags Processing Summary", show_header=True, header_style="bold cyan")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")

table.add_row("Total images", str(copied_count))
table.add_row("Total annotations", str(len(updated_annotations)))
table.add_row("Missing images", str(len(missing_images)))

console.print(table)

console.print(Panel(
    "[green]Synthetic Bags preprocessing complete[/green]",
    title="[bold green]Success[/bold green]",
    border_style="green"
))