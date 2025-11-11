from ultralytics import YOLO
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import argparse

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets_yolo"

def train():
    console.print("\n[bold cyan]Starting YOLO training for trash detection[/bold cyan]")
    console.print(f"[dim]Dataset: {DATASETS_DIR / 'combined_all' / 'data.yaml'}[/dim]\n")
    
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(DATASETS_DIR / "combined_all" / "data.yaml"),
        patience=20,
        lrf=0.01,
        epochs=100,
        imgsz=640,
        device='0',
        project=str(PROJECT_ROOT / "runs"),
        name="yolov8n_trash_detector",
        verbose=True
    )
    
    console.print("\n[bold green]Training completed[/bold green]")
    console.print(Panel(
        f"Model saved at: [yellow]{PROJECT_ROOT}/runs/yolov8n_trash_detector/weights/best.pt[/yellow]",
        title="[bold green]Training Success[/bold green]",
        border_style="green"
    ))

def validate():
    console.print("\n[bold cyan]Loading best model and running validation...[/bold cyan]")
    console.print(f"[dim]Dataset: {DATASETS_DIR / 'combined_all' / 'data.yaml'}[/dim]\n")
    
    model = YOLO("runs/yolov8n_trash_detector/weights/best.pt")
    
    results = model.val(
        data=str(DATASETS_DIR / "combined_all" / "data.yaml"),
        imgsz=640,
        batch=16,
        device='0'
    )
    
    table = Table(title="Validation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    
    table.add_row("mAP50", f"{results.box.map50:.3f}")
    table.add_row("mAP50-95", f"{results.box.map:.3f}")
    table.add_row("Precision", f"{results.box.mp:.3f}")
    table.add_row("Recall", f"{results.box.mr:.3f}")
    
    console.print("\n")
    console.print(table)
    
    console.print(Panel(
        f"Model path: [yellow]{PROJECT_ROOT}/runs/yolov8n_trash_detector/weights/best.pt[/yellow]",
        title="[bold green]Validation Complete[/bold green]",
        border_style="green"
    ))

def main():
    parser = argparse.ArgumentParser(
        description="Train or validate YOLO trash detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python train_yolo.py --mode train\n  python train_yolo.py --mode validate"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "validate"],
        default="train",
        help="Mode: train or validate (default: train)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "validate":
        validate()

if __name__ == "__main__":
    main()