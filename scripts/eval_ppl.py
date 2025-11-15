import argparse, math, torch, yaml
from nl.data.text import LMTextDataModule
from nl.models.hope import HOPELM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--config_model", default="configs/hope_small.yaml")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config_model, "r") as f:
        cfg = yaml.safe_load(f)

    dm = LMTextDataModule(
        tokenizer_name=cfg["data"]["tokenizer_name"],
        dataset_name=cfg["data"]["dataset_name"],
        dataset_config=cfg["data"]["dataset_config"],
        text_field=cfg["data"]["text_field"],
        max_seq_len=cfg["model"]["max_seq_len"],
        batch_size=cfg["train"]["batch_size"]
    )
    (_, val_loader), tok = dm.dataloaders()

    cms_levels = cfg["model"]["cms_levels"]
    model = HOPELM(
        vocab_size=cfg["model"]["vocab_size"], d_model=cfg["model"]["d_model"], d_ff=cfg["model"]["d_ff"],
        n_layers=cfg["model"]["n_layers"], d_kv=cfg["model"]["d_kv"], max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"], cms_levels=cms_levels
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            model.reset_fast_states(B=x.size(0), device=device)
            logits = model(x)
            nll += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
            n_tok += y.numel()
    print(f"val ppl: {math.exp(nll/n_tok):.2f}")

if __name__ == "__main__":
    main()
