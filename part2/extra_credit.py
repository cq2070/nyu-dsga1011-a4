"""Extra-credit T5 training and final decoding.

Run from part2 with the assignment environment. See EXTRA_CREDIT.md.
Training uses only training examples. The submitted weights were obtained in
several notebook processes, so retraining can differ from the recorded scores.
"""
import argparse
import json
import os
import pickle
import random
import re
import sqlite3
import time
from pathlib import Path


def repair_end(query):
    tokens = re.findall(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[^'\"]", query)
    end = len(tokens)
    while end and (tokens[end - 1].isspace() or tokens[end - 1] == ")"):
        end -= 1
    depth = 0
    for token in tokens[:end]:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                return query
    return "".join(tokens[:end]).rstrip() + " )" * depth


def repair_outputs(sql_path, record_path, destination, stem):
    """Apply the submitted rule only to queries with cached execution errors."""
    queries = Path(sql_path).read_text().splitlines()
    with Path(record_path).open("rb") as handle:
        records, errors = pickle.load(handle)
    if not len(queries) == len(records) == len(errors):
        raise ValueError("SQL and record counts differ")
    conn = sqlite3.connect("file:data/flight_database.db?mode=ro", uri=True)
    repaired = 0
    try:
        for index, error in enumerate(errors):
            if not error:
                continue
            candidate = repair_end(queries[index])
            if candidate == queries[index]:
                continue
            deadline = time.monotonic() + 2
            conn.set_progress_handler(
                lambda: int(time.monotonic() > deadline), 10000
            )
            try:
                conn.execute("EXPLAIN " + candidate)
                rows = conn.execute(candidate).fetchall()
            except sqlite3.Error:
                continue
            queries[index] = candidate
            records[index] = rows
            errors[index] = ""
            repaired += 1
    finally:
        conn.close()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sql_out = destination / (stem + ".sql")
    record_out = destination / (stem + ".pkl")
    sql_out.write_text("\n".join(queries) + "\n")
    with record_out.open("wb") as handle:
        pickle.dump((records, errors), handle)
    print(json.dumps({
        "examples": len(queries), "repaired": repaired,
        "execution_errors": sum(bool(error) for error in errors),
    }), flush=True)
    return sql_out, record_out


def validate_sql_tokenizer(tokenizer):
    """Check the corpus assumption used by the submitted unknown-token repair."""
    for split in ("train", "dev"):
        queries = Path("data", split + ".sql").read_text().splitlines()
        original = tokenizer(queries, add_special_tokens=False)["input_ids"]
        removed = tokenizer(
            [query.replace("<", "") for query in queries],
            add_special_tokens=False,
        )["input_ids"]
        unknowns = sum(ids.count(tokenizer.unk_token_id) for ids in original)
        remaining = sum(ids.count(tokenizer.unk_token_id) for ids in removed)
        if unknowns != sum(query.count("<") for query in queries) or remaining:
            raise ValueError("Unknown SQL tokens cannot uniquely represent '<'")


def generate_queries(model, tokenizer, encoder, mask, starts,
                     max_new_tokens=512):
    import torch
    with torch.no_grad():
        generated = model.generate(
            input_ids=encoder.to(model.device),
            attention_mask=mask.to(model.device),
            decoder_input_ids=starts.to(model.device),
            decoder_start_token_id=tokenizer.convert_tokens_to_ids("<extra_id_0>"),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )
    removable = set(tokenizer.all_special_ids)
    removable.discard(tokenizer.unk_token_id)
    queries = []
    for sequence in generated.tolist():
        ids = [token for token in sequence if token not in removable]
        text = tokenizer.decode(
            ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        queries.append(text.replace(tokenizer.unk_token, "<").strip())
    return queries


def make_optimizer(model, ordinary_lr, position_lr):
    import torch
    positions, others = [], []
    for name, parameter in model.named_parameters():
        if "relative_attention_bias" in name:
            positions.append(parameter)
        else:
            others.append(parameter)
    if not positions:
        raise ValueError("Missing relative attention bias parameters")
    return torch.optim.AdamW([
        {"params": others, "lr": ordinary_lr},
        {"params": positions, "lr": position_lr},
    ], weight_decay=0.0)


def warmup(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import T5Config, T5ForConditionalGeneration
    from load_data import T5Dataset, normal_collate_fn
    from train_t5 import generate_queries as greedy_queries

    dataset = T5Dataset("data", "train")
    indices = random.Random(42).sample(range(len(dataset)), 32)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True,
                        collate_fn=normal_collate_fn)
    evaluation = DataLoader(subset, batch_size=4, shuffle=False,
                            collate_fn=normal_collate_fn)
    config = T5Config.from_pretrained("google-t5/t5-small")
    config.dropout_rate = 0.0
    model = T5ForConditionalGeneration(config).to(args.device)
    optimizer = make_optimizer(model, 0.0003, 0.005)
    iterator = iter(loader)
    history = []
    total_loss = 0.0
    for step in range(1, 1001):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        encoder, mask, decoder, targets, _ = [
            tensor.to(args.device) for tensor in batch
        ]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=encoder, attention_mask=mask,
            decoder_input_ids=decoder,
            labels=targets.masked_fill(targets.eq(0), -100),
        ).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if step % 250:
            continue
        model.eval()
        predictions, references = [], []
        with torch.no_grad():
            for encoder, mask, _, targets, starts in evaluation:
                predictions.extend(greedy_queries(
                    model, dataset.tokenizer, encoder, mask, starts
                ))
                references.extend(dataset.tokenizer.batch_decode(
                    targets, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ))
        matches = sum(p.strip() == g.strip()
                      for p, g in zip(predictions, references))
        history.append({"step": step, "loss": total_loss / 250,
                        "matches": matches, "indices": indices})
        model.save_pretrained(args.output / ("step-" + str(step)))
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(history[-1], flush=True)
        total_loss = 0.0
        if matches == 32:
            break


def train_stage(args):
    import torch
    import train_t5
    from load_data import load_t5_data
    from t5_utils import initialize_scheduler
    from transformers import T5ForConditionalGeneration

    refining = args.stage == "refine"
    ordinary_lr, position_lr = (0.0001, 0.001) if refining else (0.0003, 0.005)
    settings = argparse.Namespace(
        finetune=False, use_wandb=False,
        experiment_name="ec_reproduction_refine" if refining else "ec_reproduction_full",
        learning_rate=ordinary_lr, batch_size=16, test_batch_size=16,
        max_n_epochs=15 if refining else 10,
        patience_epochs=5 if refining else 10,
        scheduler_type="cosine", num_warmup_epochs=1,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        str(args.checkpoint)
    ).to(args.device)
    train_loader, dev_loader, _ = load_t5_data(16, 16)
    optimizer = make_optimizer(model, ordinary_lr, position_lr)
    scheduler = initialize_scheduler(settings, optimizer, len(train_loader))
    train_t5.DEVICE = torch.device(args.device)
    for directory in ("results", "records"):
        Path(directory).mkdir(exist_ok=True)

    def save_model(checkpoint_dir, model, best):
        model.save_pretrained(args.output / ("best" if best else "last"))
    train_t5.save_model = save_model
    # Keep the historical greedy decoder for training checkpoint selection.
    train_t5.train(settings, model, train_loader, dev_loader, optimizer, scheduler)
    model = T5ForConditionalGeneration.from_pretrained(
        str(args.output / "best")
    ).to(args.device)
    values = train_t5.eval_epoch(
        settings, model, dev_loader, "data/dev.sql",
        str(args.output / "dev.sql"), "records/ground_truth_dev.pkl",
        str(args.output / "dev.pkl"),
    )
    (args.output / "metrics.json").write_text(json.dumps(dict(zip(
        ["loss", "record_f1", "record_em", "sql_em", "error_rate"], values
    )), indent=2))


def inference(args):
    import train_t5
    from load_data import get_dataloader
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from utils import compute_metrics

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    validate_sql_tokenizer(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(
        str(args.checkpoint)
    ).to(args.device).eval()
    train_t5.generate_queries = generate_queries
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    stem = "t5_ft_experiment_ec_test" if args.split == "test" else "dev"
    sql_path, record_path = raw / (stem + ".sql"), raw / (stem + ".pkl")
    if args.split == "test":
        train_t5.test_inference(
            args, model, get_dataloader(4, "test"),
            str(sql_path), str(record_path),
        )
    else:
        train_t5.eval_epoch(
            args, model, get_dataloader(4, "dev"),
            "data/dev.sql", str(sql_path),
            "records/ground_truth_dev.pkl", str(record_path),
        )
    final_sql, final_records = repair_outputs(
        sql_path, record_path, args.output, stem
    )
    with final_records.open("rb") as handle:
        records, errors = pickle.load(handle)
    expected = len(Path("data", args.split + ".nl").read_text().splitlines())
    if not len(records) == len(errors) == len(final_sql.read_text().splitlines()) == expected:
        raise ValueError("Output count mismatch")
    if args.split == "dev":
        sql_em, record_em, record_f1, _ = compute_metrics(
            "data/dev.sql", str(final_sql),
            "records/ground_truth_dev.pkl", str(final_records),
        )
        (args.output / "metrics.json").write_text(json.dumps({
            "sql_em": sql_em, "record_em": record_em, "record_f1": record_f1,
            "execution_errors": sum(bool(error) for error in errors),
        }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["warmup", "full", "refine", "infer", "repair"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--sql", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("full", "refine", "infer") and args.checkpoint is None:
        parser.error("--checkpoint is required for this stage")
    if args.stage == "repair":
        if args.sql is None or args.records is None:
            parser.error("--sql and --records are required for repair")
        stem = "t5_ft_experiment_ec_test" if args.split == "test" else "dev"
        repair_outputs(args.sql, args.records, args.output, stem)
    elif args.stage == "warmup":
        import torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        warmup(args)
    elif args.stage == "infer":
        inference(args)
    else:
        train_stage(args)


if __name__ == "__main__":
    main()

