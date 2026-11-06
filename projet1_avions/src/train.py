"""Entraînement. Exemple : python -m src.train --model hier --seed 0

--model flat : une tête à 100 variantes, cross-entropy (baseline)
--model hier : 3 têtes, somme pondérée de 3 cross-entropies
--model soft : une tête à 100 variantes, soft labels hiérarchiques (Bertinetto et al., 2020)
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .dataset import LEVELS, make_loader
from .engine import RUNS_DIR, AIRCRAFT_DIR, get_device, heads_for, load_data, predict
from .metrics import compute_metrics, soft_targets
from .model import AircraftClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["flat", "hier", "soft"], default="flat")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--frozen-epochs", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20, help="epochs de fine-tuning, après la phase gelée")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--w-coarse", type=float, default=0.5, help="poids des têtes constructeur et famille")
    p.add_argument("--beta", type=float, default=10.0, help="température des soft labels")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="test rapide : limite le nombre de batchs par epoch")
    p.add_argument("--name", default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_loss(logits, labels, args, soft):
    if args.model == "hier":
        fine = F.cross_entropy(logits["variant"], labels["variant"], label_smoothing=args.label_smoothing)
        coarse = sum(F.cross_entropy(logits[l], labels[l], label_smoothing=args.label_smoothing)
                     for l in ("manufacturer", "family"))
        return fine + args.w_coarse * coarse
    if args.model == "soft":
        return F.cross_entropy(logits["variant"], soft[labels["variant"]])
    return F.cross_entropy(logits["variant"], labels["variant"], label_smoothing=args.label_smoothing)


def make_optimizer(model, args):
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.heads.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)


def run_epoch(model, loader, optimizer, scheduler, args, soft, device, frozen):
    model.train()
    if frozen:
        model.backbone.eval()  # backbone gelé : ses BatchNorm ne doivent pas bouger non plus
    total, n = 0.0, 0
    for b, (images, labels) in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        images = images.to(device)
        labels = {l: labels[l].to(device) for l in LEVELS}
        loss = compute_loss(model(images), labels, args, soft)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total += loss.item() * len(images)
        n += len(images)
    return total / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    name = args.name or f"{args.model}_{args.backbone}_{args.img_size}_seed{args.seed}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb")

    df, classes, v2f, f2m, height = load_data()
    train_loader = make_loader(df, AIRCRAFT_DIR, "train", True, args.img_size, args.batch_size, args.num_workers)
    val_loader = make_loader(df, AIRCRAFT_DIR, "val", False, args.img_size, args.batch_size, args.num_workers)
    soft = soft_targets(height, args.beta).to(device)

    model = AircraftClassifier(heads_for(args.model, classes), args.backbone).to(device)
    history, best_acc = [], -1.0
    total_epochs = args.frozen_epochs + args.epochs
    optimizer = scheduler = None

    for epoch in range(total_epochs):
        frozen = epoch < args.frozen_epochs
        if epoch == 0 or epoch == args.frozen_epochs:  # début de chaque phase
            model.freeze_backbone(frozen)
            optimizer = make_optimizer(model, args)
            steps = (args.max_batches or len(train_loader)) * (args.frozen_epochs if frozen else args.epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, args, soft, device, frozen)
        train_time = time.time() - t0
        probs, targets = predict(model, val_loader, args.model, device, v2f, f2m, classes)
        m = compute_metrics(probs, targets, v2f, f2m, height)
        row = {"epoch": epoch, "frozen": frozen, "train_loss": train_loss, "train_time": train_time,
               "total_time": time.time() - t0, **{f"val_{k}": v for k, v in m.items()}}
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        writer.add_scalar("loss/train", train_loss, epoch)
        for k in ("acc_variant", "acc_family", "acc_manufacturer", "severity_mean_all"):
            writer.add_scalar(f"val/{k}", m[k], epoch)
        print(f"[{name}] epoch {epoch + 1}/{total_epochs} loss {train_loss:.3f} | val variante {m['acc_variant']:.3f} "
              f"famille {m['acc_family']:.3f} constructeur {m['acc_manufacturer']:.3f} "
              f"gravité {m['severity_mean_error']:.2f} | {row['total_time']:.0f} s", flush=True)
        if m["acc_variant"] > best_acc:  # le meilleur modèle est choisi sur la validation, jamais sur le test
            best_acc = m["acc_variant"]
            torch.save({"state_dict": model.state_dict(), "config": vars(args), "epoch": epoch, "val": m}, run_dir / "best.pt")
    writer.close()
    if not args.max_batches:
        (run_dir / ".done").touch()  # src.evaluate ne prend que les runs terminés
    print(f"[{name}] terminé, meilleure accuracy variante en validation : {best_acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
