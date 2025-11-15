import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class LMTextDataModule:
    def __init__(self, tokenizer_name: str, dataset_name: str, dataset_config: str, text_field: str, max_seq_len: int, batch_size: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ds = load_dataset(dataset_name, dataset_config)
        self.text_field = text_field
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def _tokenize(self, examples):
        return self.tokenizer(examples[self.text_field])

    def _group_texts(self, examples):
        # Concatenate & chunk to fixed seq length
        concat = []
        for ids in examples["input_ids"]:
            concat.extend(ids + [self.tokenizer.eos_token_id])
        total_len = (len(concat) // self.max_seq_len) * self.max_seq_len
        input_ids = [concat[i:i+self.max_seq_len] for i in range(0, total_len, self.max_seq_len)]
        return {"input_ids": input_ids}

    def dataloaders(self):
        tokenized = self.ds.map(self._tokenize, batched=True, remove_columns=self.ds["train"].column_names)
        chunked = tokenized.map(self._group_texts, batched=True)
        def collate(batch):
            x = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
            return x[:, :-1], x[:, 1:]   # inputs, targets (teacher forcing)
        return (
            torch.utils.data.DataLoader(chunked["train"], batch_size=self.batch_size, shuffle=True, collate_fn=collate),
            torch.utils.data.DataLoader(chunked["validation"], batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        ), self.tokenizer