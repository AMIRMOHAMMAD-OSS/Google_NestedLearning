import os, argparse, yaml, math, time
import torch, torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from nl.config import RootCfg, ModelCfg, TrainCfg, CMSLevelCfg
from nl.data.text import LMTextDataModule
from nl.models.hope import HOPELM

def load_cfg(path_model, path_data):
    with open(path_model, "r") as f:
        cfg = yaml.safe_load(f)

    # Optional external data override
    if path_data:
        with open(path_data, "r") as f:
            data_over = yaml.safe_load(f)
        cfg["data"] = {**cfg["data"], **data_over["data"]}

    # --- Build dataclasses explicitly to avoid double-passing cms_levels ---
    # Convert list of dicts -> list[CMSLevelCfg]
    cms_levels = [CMSLevelCfg(**lvl) for lvl in cfg["model"]["cms_levels"]]

    model_cfg = ModelCfg(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        n_layers=cfg["model"]["n_layers"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
        d_kv=cfg["model"]["d_kv"],
        cms_levels=cms_levels,
        inner_lr=cfg["model"]["inner_lr"],
        inner_scale_xtx=cfg["model"]["inner_scale_xtx"],
        inner_apply_during_eval=cfg["model"]["inner_apply_during_eval"],
        inner_apply_during_sampling=cfg["model"]["inner_apply_during_sampling"],
    )

    train_cfg = TrainCfg(**cfg["train"])

    root = RootCfg(
        exp_name=cfg["exp_name"],
        seed=cfg["seed"],
        model=model_cfg,
        train=train_cfg,
        data=cfg["data"],
    )
    return root, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    root, cfg_dict = load_cfg(args.config, args.data)
    torch.manual_seed(root.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    dm = LMTextDataModule(
        tokenizer_name=root.data["tokenizer_name"],
        dataset_name=root.data["dataset_name"],
        dataset_config=root.data["dataset_config"],
        text_field=root.data["text_field"],
        max_seq_len=cfg_dict["model"]["max_seq_len"],
        batch_size=cfg_dict["train"]["batch_size"]
    )
    (train_loader, val_loader), tokenizer = dm.dataloaders()

    # model
    cms_levels = cfg_dict["model"]["cms_levels"]
    model = HOPELM(
        vocab_size=cfg_dict["model"]["vocab_size"],
        d_model=cfg_dict["model"]["d_model"],
        d_ff=cfg_dict["model"]["d_ff"],
        n_layers=cfg_dict["model"]["n_layers"],
        d_kv=cfg_dict["model"]["d_kv"],
        max_seq_len=cfg_dict["model"]["max_seq_len"],
        dropout=cfg_dict["model"]["dropout"],
        cms_levels=cms_levels
    ).to(device)

    # optimizer for outer (slow) weights
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg_dict["train"]["lr"],
        weight_decay=cfg_dict["train"]["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    inner_lr = cfg_dict["model"]["inner_lr"]
    inner_scale_xtx = cfg_dict["model"]["inner_scale_xtx"]

    out_dir = os.path.join("outputs", root.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    global_step, running_loss = 0, 0.0

    model.train()
    for epoch in range(1_000_000):  # by steps
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            model.reset_fast_states(B=x.size(0), device=device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            scaler.scale(loss).backward()

            # Apply inner updates (Eq. 28–29) to Q/K/V and optionally CMS fc layers.
            model.apply_inner_updates(inner_lr=inner_lr, inner_scale_xtx=inner_scale_xtx)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += loss.item()

            # Logging
            if global_step % cfg_dict["train"]["log_every"] == 0:
                ppl = math.exp(min(20.0, running_loss / cfg_dict["train"]["log_every"]))
                print(f"step={global_step} loss={running_loss/cfg_dict['train']['log_every']:.4f} ppl≈{ppl:.2f}")
                running_loss = 0.0

            # Eval
            if global_step % cfg_dict["train"]["eval_every"] == 0:
                val_ppl = evaluate(model, val_loader, device)
                print(f"[eval] step={global_step} val_ppl={val_ppl:.2f}")

            # Checkpoint
            if global_step % cfg_dict["train"]["ckpt_every"] == 0:
                ckpt_path = os.path.join(out_dir, f"step{global_step}.ckpt")
                torch.save({"model": model.state_dict(), "tokenizer": tokenizer.name_or_path}, ckpt_path)

            if global_step >= cfg_dict["train"]["max_steps"]:
                torch.save({"model": model.state_dict(), "tokenizer": tokenizer.name_or_path},
                           os.path.join(out_dir, "last.ckpt"))
                return

def evaluate(model, val_loader, device):
    model.eval()
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            model.reset_fast_states(B=x.size(0), device=device)
            logits = model(x)
            nll += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
            n_tok += y.numel()
    model.train()
    return math.exp(nll / n_tok)

if __name__ == "__main__":
    main()

