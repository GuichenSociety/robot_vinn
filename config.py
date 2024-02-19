import torch
img_size = 224
train_model = dict(
    model_weight = "chkpts/BYOL_20_1.pt",
    wandb = True,
    ### dataset
    img_size = img_size,
    folder = "./datasets/push/train",
    ###
    device = "cuda" if torch.cuda.is_available() else "cpu",
    epochs = 1000,
    batch_size = 26,
    save_dir = "./chkpts/",
    version = 1
)