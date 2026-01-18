import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from src.model.model import VideoClassificationModel
from src.data.dataset import IWDDDataModule
from src.utils.utils import get_model_config

def coding_rate(Z, eps=1e-4):
    n, d = Z.shape
    _, rate = np.linalg.slogdet(np.eye(d) + (1.0 / (n * eps)) * (Z.T @ Z))
    return 0.5 * rate

def transrate(Z, y, eps=1e-4):
    Z = Z - Z.mean(axis=0, keepdims=True)
    rz = coding_rate(Z, eps)
    rzy = 0.0
    k = int(y.max() + 1)
    for i in range(k):
        Zi = Z[y == i]
        if Zi.shape[0] > 1:
            rzy += coding_rate(Zi, eps)
    return rz - rzy / k

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    model_config = get_model_config("videomae_ssv2")
    model = VideoClassificationModel(model_config=model_config)
    model.to(device)
    model.eval()

    all_blocks = list(model.model.encoder.layer)
    target_layers = all_blocks[-5:]
    
    captured_outputs = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        mean_output = output.mean(dim=1).detach().cpu().numpy()
        captured_outputs.append(mean_output)

    hooks = []
    for layer in target_layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    dm = IWDDDataModule(
        model_config=model_config,
        batch_size=2,
        num_workers=0,
        clip_duration=3,
        stride=1,
        persistent_workers=False,
        train_split=0.1,
        num_frames=16,
        val_split=0.15,
        use_yolo=False,
    )
    dm.setup("validate")
    loader = dm.val_dataloader()

    all_features = [[] for _ in range(5)]
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 50:
                break
            
            x = batch["pixel_values"].to(device)
            y = batch["labels"]
            
            captured_outputs.clear()
            
            model({"pixel_values": x})
            
            if len(captured_outputs) > 5:
                current_batch_features = captured_outputs[-5:]
            else:
                current_batch_features = captured_outputs
            
            for i, out in enumerate(current_batch_features):
                all_features[i].append(out)
            
            if torch.is_tensor(y):
                all_labels.append(y.cpu().numpy())
            else:
                all_labels.append(np.array(y))

    for h in hooks:
        h.remove()

    y_full = np.concatenate(all_labels)
    if y_full.ndim > 1:
        y_full = np.argmax(y_full, axis=1)

    transrate_values = []
    for i in range(5):
        Z_layer = np.concatenate(all_features[i], axis=0)
        Z_layer = Z_layer / (np.linalg.norm(Z_layer, axis=1, keepdims=True) + 1e-8)
        tr_val = transrate(Z_layer, y_full, eps=1e-4)
        transrate_values.append(tr_val)

    plt.figure(figsize=(10, 6))
    layers_labels = ["L-4", "L-3", "L-2", "L-1", "L"]
    plt.plot(layers_labels, transrate_values, marker='o', color='royalblue', linewidth=2)
    plt.title("Transrate: Last 5 Layers of VideoMAE")
    plt.xlabel("Layer Index")
    plt.ylabel("Transrate Score")
    plt.grid(True, alpha=0.3)
    plt.savefig("transrate_plot.png")
    plt.show()

if __name__ == "__main__":
    main()