import os
import numpy as np
import argparse
from datetime import datetime


import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from bams.data import KeypointsDataset
from bams.models import BAMS
from bams import HoALoss

# Customized for Alice dataset
def load_data(f, path):
    segment = 60 # in seconds
    fz = 500
    sample_period = int(f.split(".")[0].split("samp")[-1])
    step = fz // sample_period * 10 # 10 second as a step
    
    # load raw train data (with annotations for 2 tasks)
    data_train = np.load(
        os.path.join(path, f), allow_pickle=True
    )

    print(data_train.keys())
    min_len = min(map(lambda x: x.shape[0], data_train.values()))
    print(min_len)
    total_sample = segment * sample_period
    keypoints = np.array([[data[start * step : start * step + total_sample] 
                           for start in range((min_len - total_sample) // step)] 
                          for data in data_train.values()])

    keypoints = keypoints.reshape(keypoints.shape[0] * keypoints.shape[1], keypoints.shape[2], keypoints.shape[3])
    print(keypoints.shape)

    return keypoints


def train_loop(model, device, loader, optimizer, criterion, writer, step, log_every_step):
    model.train()

    for data in tqdm(loader, position=1, leave=False):
        # todo convert to float
        input = data["input"].float().to(device)  # (B, N, L)
        target = data["target_hist"].float().to(device)
        ignore_weights = data["ignore_weights"].to(device)

        # forward pass
        optimizer.zero_grad()
        embs, hoa_pred, byol_preds = model(input)

        # prediction task
        hoa_loss = criterion(target, hoa_pred, ignore_weights)

        # contrastive loss: short term
        batch_size, sequence_length, emb_dim = embs["short_term"].size()
        skip_frames, delta = 60, 5
        view_1_id = (
            torch.randint(sequence_length - skip_frames - delta, (batch_size,))
            + skip_frames
        )
        view_2_id = torch.randint(delta + 1, (batch_size,)) + view_1_id
        view_2_id = torch.clip(view_2_id, 0, sequence_length)

        view_1 = byol_preds["short_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["short_term"][torch.arange(batch_size), view_2_id]

        byol_loss_short_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # contrastive loss: long term
        batch_size, sequence_length, emb_dim = embs["long_term"].size()
        skip_frames = 100
        view_1_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )
        view_2_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )

        view_1 = byol_preds["long_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["long_term"][torch.arange(batch_size), view_2_id]

        byol_loss_long_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # backprop
        loss = 5e2 * hoa_loss + 0.5 * byol_loss_short_term + 0.5 * byol_loss_long_term

        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every_step == 0:
            writer.add_scalar("train/hoa_loss", hoa_loss.item(), step)
            writer.add_scalar(
                "train/byol_loss_short_term", byol_loss_short_term.item(), step
            )
            writer.add_scalar(
                "train/byol_loss_long_term", byol_loss_long_term.item(), step
            )
            writer.add_scalar("train/total_loss", loss.item(), step)

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="24chans.pkl")
    parser.add_argument("--data_root", type=str, default="./data/alice")
    parser.add_argument("--cache_path", type=str, default="./data/alice/custom_dataset")
    parser.add_argument("--hoa_bins", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--log_every_step", type=int, default=50)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--job",
        default="train",
        const="train",
        nargs="?",
        choices=["train", "compute_representations"],
        help="select task",
    )
    args = parser.parse_args()

    if args.job == "train":
        train(args)
    elif args.job == "compute_representations":
        compute_representations(args)

def train(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    keypoints = load_data(args.input_file, args.data_root)

    dataset = KeypointsDataset(
        keypoints=keypoints,
        hoa_bins=args.hoa_bins,
        cache_path=args.cache_path,
        cache=False,
    )
    print("Number of sequences:", len(dataset))

    # prepare dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 64, 64), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 64, 64), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins)
        ),
    ).to(device)

    print(model)

    model_name = f"bams-custom-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    writer = SummaryWriter("runs/" + model_name)

    main_params = [p for name, p in model.named_parameters() if "byol" not in name]
    byol_params = list(model.byol_predictors.parameters())

    optimizer = optim.AdamW(
        [{"params": main_params}, {"params": byol_params, "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    criterion = HoALoss(hoa_bins=args.hoa_bins, skip_frames=100)

    step = 0
    for epoch in tqdm(range(1, args.epochs + 1), position=0):
        step = train_loop(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            writer,
            step,
            args.log_every_step,
        )
        scheduler.step()

        if epoch % 50 == 0:
            torch.save(model.state_dict(), model_name + ".pt")

def compute_representations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keypoints = load_data(args.input_file, args.data_root)

    keypoints, split_mask, batch = load_mice_triplet(args.data_root)

    # dataset
    if not Dataset.cache_is_available(args.cache_path, args.hoa_bins):
        print("Processing data...")
        input_feats, target_feats, ignore_frames = mouse_feature_extractor(keypoints)
    else:
        print("No need to process data")
        input_feats = target_feats = ignore_frames = None

    # only use

    dataset = Dataset(
        input_feats=input_feats,
        target_feats=target_feats,
        ignore_frames=ignore_frames,
        cache_path=args.cache_path,
        hoa_bins=args.hoa_bins,
        hoa_window=30,
    )

    print("Number of sequences:", len(dataset))

    # build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 32, 32), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 32, 32), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins),
        ),  # frame rate = 30, 6 steps = 200ms
    ).to(device)

    if args.ckpt_path is None:
        raise ValueError("Please specify a checkpoint path")

    # load checkpoint
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    loader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=32,
        num_workers=16,
        pin_memory=True,
    )

    # compute representations
    short_term_emb, long_term_emb = [], []

    for data in loader:
        input = data["input"].float().to(device)  # (B, N, L)

        with torch.inference_mode():
            embs, _, _ = model(input)

            short_term_emb.append(embs["short_term"].detach().cpu())
            long_term_emb.append(embs["long_term"].detach().cpu())

    short_term_emb = torch.cat(short_term_emb)
    long_term_emb = torch.cat(long_term_emb)

    embs = torch.cat([short_term_emb, long_term_emb], dim=2)

    # the learned representations are at the individual mouse level, we want to compute
    # the mouse triplet-level representation
    # embs: (B, L, N)
    batch_size, seq_len, num_feats = embs.size()
    embs = embs.reshape(-1, 3, seq_len, num_feats)

    embs_mean = embs.mean(1)
    embs_max = embs.max(1).values
    embs_min = embs.min(1).values

    embs = torch.cat([embs_mean, embs_max - embs_min], dim=-1)

    # normalize embeddings
    mean, std = embs.mean(0, keepdim=True), embs.std(0, unbiased=False, keepdim=True)
    embs = (embs - mean) / std

    frame_number_map = np.load(
        os.path.join(args.data_root, "mouse_triplet_frame_number_map.npy"),
        allow_pickle=True,
    ).item()

    # only take submission frames
    embs = embs.numpy()[~split_mask].reshape(-1, embs.shape[-1])

    submission_dict = dict(
        frame_number_map=frame_number_map,
        embeddings=embs,
    )

    model_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    np.save(f"{model_name}_submission.npy", submission_dict)

if __name__ == "__main__":
    main()
