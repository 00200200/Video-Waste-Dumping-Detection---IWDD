import lightning as L
import torch
import torch.nn as nn
from torchmetrics.functional import f1_score, precision, recall
from transformers import AutoModel
from tokenizers import Tokenizer

from src.utils.metrics import calculate_metrics


class VideoTextClassificationModel(L.LightningModule):
    def __init__(
        self,
        model_config,
        learning_rate=1e-5,
        num_classes=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.video_processor = AutoModel.from_pretrained(model_config["video_model_name"])
        self.tokenizer = Tokenizer.from_pretrained(model_config["text_model_name"])
        self.text_encoder = AutoModel.from_pretrained(model_config["text_model_name"])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        video_hidden_size = self.video_processor.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size 

        self.fusion = FusionModule(text_hidden_size, video_hidden_size, 0.1, 256)
        self.classifier = nn.Linear(256, num_classes)


    def forward(self, text, pixel_values):
        encoded_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        text_outputs = self.text_encoder(
            input_ids=encoded_text["input_ids"],
            attention_mask=encoded_text["attention_mask"],
        )
        text_emb = text_outputs.last_hidden_state[:, 0, :]
        
        video_outputs = self.video_processor(pixel_values)
        video_emb = video_outputs.last_hidden_state[:, 0]

        sequence_emb = torch.cat([text_emb, video_emb], dim=-1)

        fusion = self.fusion(sequence_emb)
        logits = self.classifier(fusion)
        return logits

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["pixel_values"], train_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        prec_score = precision(preds, y, task="multiclass", num_classes=2)
        recall_score = recall(preds, y, task="multiclass", num_classes=2)
        self.log("train_accuracy", acc, prog_bar=True, batch_size=x.size(0))
        self.log("train_f1", f1, prog_bar=True, batch_size=x.size(0))
        self.log("train_precision", prec_score, prog_bar=True, batch_size=x.size(0))
        self.log("train_recall", recall_score, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["pixel_values"], val_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.size(0))
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        prec_score = precision(preds, y, task="multiclass", num_classes=2)
        recall_score = recall(preds, y, task="multiclass", num_classes=2)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=x.size(0))
        self.log("val_f1", f1, prog_bar=True, batch_size=x.size(0))
        self.log("val_precision", prec_score, prog_bar=True, batch_size=x.size(0))
        self.log("val_recall", recall_score, prog_bar=True, batch_size=x.size(0))
        self.clip_outputs.append(
            {
                "preds": preds,
                "targets": y,
                "video_ids": val_batch["video_ids"],
                "start_times": val_batch["start_times"],
                "end_times": val_batch["end_times"],
                "video_labels": val_batch["video_labels"],
                "video_timestamps": val_batch["video_timestamps"],
            }
        )
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch["pixel_values"], test_batch["labels"]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_validation_epoch_end(self):
        metrics = calculate_metrics(self.clip_outputs)
        self.log("val_video_precision", metrics["precision"], prog_bar=True)
        self.log("val_video_recall", metrics["recall"], prog_bar=True)
        self.log("val_video_f1", metrics["f1"], prog_bar=True)
        self.clip_outputs = []

class FusionModule(nn.Module):
    def __init__(self, text_hidden_size, video_hidden_size, dropout, output_size):
        super().__init__()
        final_dim = text_hidden_size + video_hidden_size
        self.model = nn.Sequential(
            nn.Conv1d(final_dim, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Linear(512, output_size)
        )
    