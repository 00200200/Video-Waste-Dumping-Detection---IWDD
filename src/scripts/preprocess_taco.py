from pathlib import Path
import shutil
import json
from tqdm import tqdm
import re
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

TACO_DIR = DATASETS_DIR / "taco-trash-dataset"
COCO_DIR = DATASETS_DIR / "_taco-trash-dataset_processed"
IMAGES_DIR = COCO_DIR / "images"
ANNOTATIONS_DIR = COCO_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

console.print("\n[bold cyan]Processing TACO dataset[/bold cyan]\n")

original_annotations_file = TACO_DIR / "annotations.json"

if not original_annotations_file.exists():
    console.print("[red]Error: annotations.json not found[/red]")
    exit(1)

console.print("[blue]Loading original annotations...[/blue]")
with open(original_annotations_file) as f:
    annotations_data = json.load(f)

batch_dirs = sorted([d for d in TACO_DIR.iterdir() if d.is_dir() and d.name.startswith("batch_")],
                    key=lambda x: int(re.search(r'\d+', x.name).group()))

console.print(f"[cyan]Found {len(batch_dirs)} batches[/cyan]\n")

console.print("[blue]Building file mapping from annotations...[/blue]")
file_name_mapping = {}

for img_info in annotations_data.get("images", []):
    original_file_path = img_info.get("file_name", "")
    file_name_only = Path(original_file_path).name
    
    batch_num = None
    if "batch_" in original_file_path:
        match = re.search(r'batch_(\d+)', original_file_path)
        if match:
            batch_num = int(match.group(1))
    
    found = False
    
    if batch_num:
        batch_name = f"batch_{batch_num}"
        batch_dir = TACO_DIR / batch_name
        if batch_dir.exists():
            potential_files = list(batch_dir.glob(f"{Path(file_name_only).stem}.*"))
            if potential_files:
                actual_file = potential_files[0]
                normalized_name = f"{Path(actual_file).stem}".lower() + Path(actual_file).suffix.lower()
                new_name = f"{batch_name}_{normalized_name}"
                file_name_mapping[original_file_path] = (batch_dir, new_name, actual_file)
                found = True
    
    if not found:
        for batch_dir in batch_dirs:
            potential_files = list(batch_dir.glob(f"{Path(file_name_only).stem}.*"))
            if potential_files:
                actual_file = potential_files[0]
                normalized_name = f"{Path(actual_file).stem}".lower() + Path(actual_file).suffix.lower()
                new_name = f"{batch_dir.name}_{normalized_name}"
                file_name_mapping[original_file_path] = (batch_dir, new_name, actual_file)
                found = True
                break
    
    if not found:
        console.print(f"[yellow]Warning: File not found in any batch: {original_file_path}[/yellow]")

console.print(f"[green]Mapped {len(file_name_mapping)} files from annotations[/green]\n")

console.print("[blue]Copying images...[/blue]")
image_count = 0
for original_path, (batch_dir, new_name, actual_file) in tqdm(file_name_mapping.items(), 
                                                                desc="Copying images",
                                                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
    new_path = IMAGES_DIR / new_name
    shutil.copy(actual_file, new_path)
    image_count += 1

console.print(f"[green]Copied {image_count} images with unique names[/green]\n")

console.print("[blue]Updating image paths in annotations...[/blue]")
updated_images = []

if "images" in annotations_data:
    for img_info in annotations_data["images"]:
        original_file_path = img_info.get("file_name", "")
        
        if original_file_path in file_name_mapping:
            _, new_name, _ = file_name_mapping[original_file_path]
            img_info["file_name"] = new_name
            updated_images.append(img_info)

console.print(f"[green]Updated {len(updated_images)} image paths[/green]")

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

table = Table(title="TACO Processing Summary", show_header=True, header_style="bold cyan")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")

table.add_row("Total batches", str(len(batch_dirs)))
table.add_row("Total images", str(image_count))
table.add_row("Total annotations", str(len(updated_annotations)))

console.print(table)

console.print(Panel(
    "[green]TACO preprocessing complete[/green]",
    title="[bold green]Success[/bold green]",
    border_style="green"
))