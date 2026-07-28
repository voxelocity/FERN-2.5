# FERN-2.5 (eco)

A small language model I've been building, sized so it actually trains on
one consumer GPU (a 5070, 12GB) instead of a rented cluster.

## How it works

FERN is a sparse model. There's a big pool of small "expert" sub-networks grouped
by rough role (language, math, code, and so on), but only the top two fire for
any given token, so the compute per token stays tiny even as the total parameter
count grows.

Two other things make it a bit different from a plain transformer:

- **It reuses one reasoning block instead of stacking layers.** Depth becomes
  something the model spends when a token is hard, not a fixed number baked in.
- **It generates by block diffusion.** Rather than emitting tokens strictly one
  at a time, it denoises a small block in parallel, which also makes infilling
  fall out naturally.

The "eco" part is just the stuff that keeps it on a 12GB card:

- reversible blocks, so activation memory doesn't grow with reasoning depth,
- GaLore, which keeps the optimizer state in a low-rank subspace (roughly half
  the memory it'd otherwise need),
- a deliberately small default preset of about 8M params total, ~2.5M active per token.

There's a bit more detail in [`ECO_NOTES.md`](ECO_NOTES.md).

## Running it

```bash
pip install -r requirements.txt
python smoke_test.py                                   # sanity check
python train.py --preset eco --reversible --galore --amp --steps 2000
```

That trains on a tiny built-in corpus so you can watch the loss move without any
setup. To train on your own data, tokenize it and point `--bin` at the result:

```bash
python train.py --preset eco --reversible --galore --amp \
    --bin your_data.bin --block 1024 --batch 24 --steps 100000
```

## Status

Early and experimental. It trains, the loss goes down, and the pieces fit
together, but I haven't pushed it at anything serious yet, so expect rough edges.

## License

MIT, see [LICENSE](LICENSE).
