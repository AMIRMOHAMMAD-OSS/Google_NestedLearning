import argparse, torch, yaml
from nl.models.hope import HOPELM
from nl.utils.sampling import sample_autoregressive
from transformers import AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--config_model", default="configs/hope_small.yaml")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config_model, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    cms_levels = cfg["model"]["cms_levels"]
    model = HOPELM(
        vocab_size=cfg["model"]["vocab_size"], d_model=cfg["model"]["d_model"], d_ff=cfg["model"]["d_ff"],
        n_layers=cfg["model"]["n_layers"], d_kv=cfg["model"]["d_kv"], max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"], cms_levels=cms_levels
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    text = sample_autoregressive(
        model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        apply_inner=cfg["model"]["inner_apply_during_sampling"], inner_lr=cfg["model"]["inner_lr"], inner_scale=cfg["model"]["inner_scale_xtx"],
        device=device
    )
    print(text)

if __name__ == "__main__":
    main()
