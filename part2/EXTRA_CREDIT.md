# Extra credit — T5 from scratch

Final dev scores: Record F1 61.5172%, Record EM 56.8670%, SQL EM 45.7082%, execution errors 6/466. Final hidden-test Record F1 is 50.27%, with 11/432 execution errors.

Use Python 3.11 and the assignment's part2 requirements. Run inside part2. The script imports the existing load_data.py, train_t5.py, t5_utils.py and utils.py. No additional dependencies are required.

## Reproduce predictions from the submitted checkpoint

Mount Google Drive in Colab first. The submitted checkpoint is backed up at /content/drive/MyDrive/DSGA1011-A4/EC-full-position-v2/best.

```bash
python extra_credit.py --stage infer --split dev \
  --checkpoint /content/drive/MyDrive/DSGA1011-A4/EC-full-position-v2/best \
  --output /content/drive/MyDrive/DSGA1011-A4/EC-reproduced-dev

python extra_credit.py --stage infer --split test \
  --checkpoint /content/drive/MyDrive/DSGA1011-A4/EC-full-position-v2/best \
  --output /content/drive/MyDrive/DSGA1011-A4/EC-reproduced-test
```

Final test filenames are t5_ft_experiment_ec_test.sql and t5_ft_experiment_ec_test.pkl. The raw directory retains predictions before trailing-parenthesis correction. The pickle contains (records, errors), matching the assignment format.

Final inference uses beam width 4 and 512 maximum new tokens. It retains unknown tokens while removing other special tokens, then restores '<'. The script verifies on train/dev SQL that unknown tokens correspond only to '<'. Queries with cached execution errors undergo the submitted trailing-parenthesis repair. A candidate must pass SQLite EXPLAIN and execute within two seconds. Valid original queries are retained. No reference SQL or reference records are used by the repair rule.

The inherited raw SQL executor uses a 120-second batch timeout. Wall-clock timeouts can cause record outputs to vary across machines.

## Repair cached predictions directly

```bash
python extra_credit.py --stage repair --split test \
  --sql /content/drive/MyDrive/DSGA1011-A4/EC-full-position-v2/test-beam4-restore-lt/t5_ft_experiment_ec_test.sql \
  --records /content/drive/MyDrive/DSGA1011-A4/EC-full-position-v2/test-beam4-restore-lt/t5_ft_experiment_ec_test.pkl \
  --output /content/drive/MyDrive/DSGA1011-A4/EC-repaired-test
```

## Reproduce the training procedure

```bash
python extra_credit.py --stage warmup --output /content/ec_run/warmup
python extra_credit.py --stage full --checkpoint /content/ec_run/warmup/step-1000 --output /content/ec_run/full
python extra_credit.py --stage refine --checkpoint /content/ec_run/full/best --output /content/ec_run/refine
```

Warmup initializes T5-small with random weights and dropout 0, samples 32 training examples with seed 42, and trains up to 1,000 steps. Batch size is 4, ordinary/relative-bias learning rates are 0.0003/0.005, weight decay is 0, and gradient norm is clipped to 1.0. If all 32 queries are memorized early, substitute the final saved step directory for step-1000.

Full-data training runs for 10 epochs with batch size 16 and the same learning rates. Refinement loads its best checkpoint and runs for up to 15 epochs with learning rates 0.0001/0.001 and patience 5. Both stages use AdamW, cosine scheduling, one warmup epoch and greedy dev F1 for checkpoint selection, matching the historical selection procedure. Optimizer and scheduler states reset at stage boundaries.

The diagnostic warmup used PyTorch seed 42. Later notebook processes did not explicitly reset their seeds. Retraining can produce different weights and scores. Use the submitted checkpoint for direct prediction reproduction. Best and last are weight/configuration checkpoints; optimizer state is not saved. The failed earlier uniform-learning-rate trials are outside the final weight lineage.

## Written report

Add A4_Extra_Credit.tex next to main.tex in Overleaf. Insert \input{A4_Extra_Credit} immediately before \end{document}. The appendix uses the main report's booktabs, tabularx, array and hyperref packages.

