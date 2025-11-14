from pathlib import Path
import yaml
import shutil
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets_yolo"

DATASETS = ["taco-trash-dataset_yolo", "synthetic-bags_yolo", "wasteinsight_yolo"]
OUTPUT_DIR = DATASETS_DIR / "combined_all"

console.print("\n[bold cyan]Merging YOLO datasets into combined_all[/bold cyan]\n")

console.print("[blue]Creating directory structure...[/blue]")
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

console.print("[green]Directory structure created[/green]\n")

total_images = 0
total_labels = 0

for dataset_name in DATASETS:
    dataset_path = DATASETS_DIR / dataset_name
    console.print(f"[cyan]Processing dataset: {dataset_name}[/cyan]")
    
    splits_to_process = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "test": "test"
    }
    
    for source_split, target_split in splits_to_process.items():
        source_images = dataset_path / "images" / source_split
        source_labels = dataset_path / "labels" / source_split
        
        target_images = OUTPUT_DIR / "images" / target_split
        target_labels = OUTPUT_DIR / "labels" / target_split
        
        if not source_images.exists():
            continue
        
        image_files = list(source_images.glob("*"))
        console.print(f"[blue]Copying {len(image_files)} images from {source_split}[/blue]")
        
        for img_file in tqdm(image_files, desc=f"  {dataset_name} {source_split}",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
            shutil.copy(img_file, target_images / img_file.name)
            total_images += 1
        
        if source_labels.exists():
            label_files = list(source_labels.glob("*.txt"))
            console.print(f"[blue]Converting {len(label_files)} labels from {source_split}[/blue]")
            
            for label_file in tqdm(label_files, desc=f"  {dataset_name} {source_split} labels",
                                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
                with open(label_file) as f:
                    lines = f.readlines()
                
                converted_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        converted_lines.append("0 " + " ".join(parts[1:]) + "\n")
                
                target_label_file = target_labels / label_file.name
                with open(target_label_file, "w") as f:
                    f.writelines(converted_lines)
                total_labels += 1
    
    console.print(f"[green]Completed: {dataset_name}[/green]\n")

console.print("[blue]Creating unified data.yaml...[/blue]")

data_yaml = {
    "path": str(OUTPUT_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": {0: "trash"}
}

output_yaml = OUTPUT_DIR / "data.yaml"
with open(output_yaml, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

console.print("[green]Configuration file created[/green]\n")

table = Table(title="Dataset Merge Summary", show_header=True, header_style="bold cyan")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")

table.add_row("Total datasets", str(len(DATASETS)))
table.add_row("Total images", str(total_images))
table.add_row("Total labels", str(total_labels))
table.add_row("Classes", "1 (trash)")

console.print(table)

console.print(Panel(
    "[green]Datasets merged successfully[/green]",
    title="[bold green]Success[/bold green]",
    border_style="green"
))