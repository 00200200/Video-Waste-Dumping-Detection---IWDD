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

WASTEINSIGHT_DIR = DATASETS_DIR / "WasteInsight.v2i.coco"
COCO_DIR = DATASETS_DIR / "_wasteinsight_processed"
IMAGES_DIR = COCO_DIR / "images"
ANNOTATIONS_DIR = COCO_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

console.print("\n[bold cyan]Processing WasteInsight dataset[/bold cyan]\n")

splits = ["train", "valid", "test"]
total_images = 0
total_annotations = 0

for split in splits:
    split_dir = WASTEINSIGHT_DIR / split
    json_file = split_dir / "_annotations.coco.json"
    
    if not split_dir.exists():
        console.print(f"[red]Error: {split} directory not found[/red]")
        continue
    
    console.print(f"[cyan]Processing {split} split...[/cyan]")
    
    all_images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.jpeg")) + list(split_dir.glob("*.png"))
    
    console.print(f"[blue]Copying {len(all_images)} images...[/blue]")
    for img_file in tqdm(all_images, desc=f"  {split} images", 
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
        shutil.copy(img_file, IMAGES_DIR / img_file.name)
    
    total_images += len(all_images)
    
    if json_file.exists():
        output_json_name = f"annotations_{split}.json"
        shutil.copy(json_file, ANNOTATIONS_DIR / output_json_name)
        console.print(f"[green]Copied annotations_{split}.json[/green]\n")
    else:
        console.print(f"[yellow]Warning: {json_file.name} not found in {split}[/yellow]\n")

console.print("[blue]Merging annotations from all splits...[/blue]")

merged_images = []
merged_annotations = []
merged_categories = []
image_id_offset = 0
annotation_id_offset = 0

for split in splits:
    json_file = ANNOTATIONS_DIR / f"annotations_{split}.json"
    
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        
        if data.get("categories"):
            for cat in data["categories"]:
                if cat not in merged_categories:
                    merged_categories.append(cat)
        
        if data.get("images"):
            for img in data["images"]:
                img["id"] = img["id"] + image_id_offset
                merged_images.append(img)
            image_id_offset += len(data["images"])
        
        if data.get("annotations"):
            for ann in data["annotations"]:
                ann["id"] = ann["id"] + annotation_id_offset
                ann["image_id"] = ann["image_id"] + image_id_offset - len(data["images"])
                merged_annotations.append(ann)
            annotation_id_offset += len(data["annotations"])
            total_annotations += len(data["annotations"])

merged_coco = {
    "images": merged_images,
    "annotations": merged_annotations,
    "categories": merged_categories
}

output_json = ANNOTATIONS_DIR / "annotations.json"
with open(output_json, "w") as f:
    json.dump(merged_coco, f, indent=2)

console.print("[green]Merged annotations saved[/green]\n")

console.print("[blue]Cleaning up temporary files...[/blue]")
for split in splits:
    json_file = ANNOTATIONS_DIR / f"annotations_{split}.json"
    if json_file.exists():
        json_file.unlink()

console.print("[green]Temporary files removed[/green]\n")

table = Table(title="WasteInsight Processing Summary", show_header=True, header_style="bold cyan")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")

table.add_row("Total images", str(total_images))
table.add_row("Total annotations", str(total_annotations))
table.add_row("Total categories", str(len(merged_categories)))

console.print(table)

console.print(Panel(
    "[green]WasteInsight preprocessing complete[/green]",
    title="[bold green]Success[/bold green]",
    border_style="green"
))