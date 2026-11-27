# Matches 7–10 annotation review

Open [review.html](review.html) to inspect every sampled gameplay frame. Blue boxes mark friendly troops; yellow boxes mark enemy troops. Green **SCREENED** frames are in this corrected dataset. Red **NEEDS REVIEW** frames are shown with the original model-generated boxes and are withheld from training.

This corrects obvious missed troops and false boxes from the `troops-v3` draft. I screened all 111 sampled gameplay frames and retained 55 with clearer labels; 56 crowded, obscured, or post-match transition frames remain for manual annotation. The retained frames are still machine-assisted and are **not** a fully hand-verified gold standard. The exact frame lists and corrections are in [manifest.json](manifest.json).

The `images/`, `labels/`, `metadata/`, `sources/`, and `classes.txt` files form a standalone copy of the retained annotations. The local ignored `dataset/` directory has the same corrected frames alongside the older matches. The earlier model-assisted training snapshot remains under `runs/troops-v3/dataset/`; the v3 weights were trained on that snapshot and have not been retrained on these corrections. Matches 5 and 6 remain in `dataset/holdout` and were not used for these labels.

To inspect or edit a saved box at full size, run for example:

```sh
python -m clash_royale annotate recordings/match07.mp4 --dataset dataset --review
```

Repeat with `match08.mp4`, `match09.mp4`, or `match10.mp4`. Save edits with **S**. Re-run `prepare-dataset` before any later training run because the dataset changed.
