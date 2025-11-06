# import lightning as L
# from lightning.pytorch.callbacks import ModelCheckpoint

# # from src.data.dataio import ImageDataModule
# # from src.model.yolo import YoloModel


# def main():
#     L.seed_everything(42)

#     # model = YoloModel()
#     # data = ImageDataModule()

#     # checkpoint_callback = ModelCheckpoint(
#     #     monitor="val_loss", filename="best_model-{epoch:02d}-{val_loss:.2f}"
#     # )

#     # trainer = L.Trainer(
#     #     max_epochs=20,
#     #     accelerator="auto",
#     #     log_every_n_steps=1,
#     #     deterministic=True,
#     #     callbacks=[checkpoint_callback],
#     # )
#     # trainer.fit(model, data)


# if __name__ == "__main__":
#     main()

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    patience=20,
    lrf=0.01,
    data='kaggle/working/taco_yolo/data.yaml', 
    epochs=100, 
    imgsz=640,
    device='mps'
    )