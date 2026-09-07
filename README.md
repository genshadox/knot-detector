# Wood Board Knot Detection

Detects knots in wood boards from sequential camera frames using YOLO11n,
and outputs knot bounding boxes per board (board-space, not frame-space).

## How it works

Frames are named `{boardIndex}_{frameNumber}.png`.

1. Group frames by board, sort by frame number, stitch into one board image.
2. Slide a fixed-size window (`WINDOW_SIZE`/`WINDOW_STRIDE`) across the
   board image, running YOLO inference per window.
3. Shift each window's detections into board-space coordinates.
4. Merge overlapping cross-window detections of the same knot (see
   below), then normalize to YOLO format.
5. Save per board:
   - `{board_id}.txt` - detections in YOLO format, board-space
   - `{board_id}.png` - stitched board image with boxes drawn

## Output format

Bounding boxes (YOLO xywh - board space), not traced polygons - no segmentation ground
truth exists to train or validate against.

## Cross-window merging

A knot near a frame/window boundary can be detected more than once
across overlapping windows. Duplicate detections are merged via IoA
(intersection-over-area, either direction) above `POST_MERGE_IOA_THRESHOLD`
(`params.py`) - boxes that substantially contain one another are unioned
into one box. This also subsumes the old fully-nested box-in-box
duplicate case (see Known limitations below).

## Known limitations

- **Fully-nested / cross-window duplicates**: handled by the IoA merge
  step above; the merge threshold was tuned via sweeps against real val-set metrics,
  cross-checked against the dataset's own GT overlap patterns
- **Duplicate detections within a window**: handled by NMS tuning (see
  below).

## Training

- YOLO11n, pretrained weights. Config: `model/args.yaml`.
- Board 1043 had a corrupted label - excluded from stats, split into train/val
  manually.
- Train/val split is per-board (not per-frame, since frames within a
  board overlap in the training set) and stratified by max knot size per
  board, targeting ~70/30.
- Inference NMS `iou=0.5` (default 0.7, see `params.py`) - lowered to cut
  duplicate detections. Checked against GT: real knots never exceed 0.5
  IoU with each other, so this doesn't risk merging distinct knots.

## Test mode

`MODE=test` runs the full pipeline and compares board-space output
against a ground-truth test set (rather than validating the model alone,
frame-space, like `model.val()` would).

**Input** (`TEST_DIR_PATH`): frames + one `{board_id}.txt` GT label file
per board, same board-space format as this program's output.

**Output** (`compare/` subfolder):
- GT boxes drawn over `{board_id}.png` for visual comparison
- `{board_id}_compare.json`: `matched` / `missed` / `false_positives` /
  `mean_iou_matched` / `precision` / `recall`, via IoU matching (`box_iou` + Hungarian assignment,
  avoids double-counting duplicates) at IoU ≥ 0.5
- `summary.json`: pooled totals across the test set

## Tests

Unit tests cover `filter_boxes_ioa_merge` and `compare_labels`. Kept in `tests/`,
outside `src/`, so `pytest` and test files never end up in the prod
image.

```bash
docker compose -f docker-compose.dev.yml up
docker compose -f docker-compose.dev.yml exec runner pytest /tests -v
```

## Running

Inference runs on CPU (no CUDA/GPU dependency required).

```bash
docker compose up                              # production
docker compose -f docker-compose.dev.yml up    # development
```

Configure via `.env`:

| Variable         | Purpose                                  |
|------------------|-------------------------------------------|
| `MODE`           | `infer` or `test`                        |
| `INPUT_DIR_PATH` | input frames (infer)                     |
| `OUTPUT_DIR_PATH`| inference output                         |
| `TEST_DIR_PATH`  | test set (frames + GT labels)            |
| `LOG_DIR_PATH`   | logs                                     |
| `CLEANUP`        | delete results from previous run         |
| `UID` / `GID`    | container user/group (match host)        |


## Project structure

- `main.py` - entry point, mode branching
- `model_runner.py` - YOLO inference, per-window sliding, per-board error isolation
- `board.py` - `Board`/`Frame`/`Window` model, board-space stitching and merge
- `params.py` - inference/NMS/merge thresholds
- `test.py` - test-mode comparison
- `util.py` - path/label/coordinate helpers
- `logger.py` - logging setup
- `model/args.yaml` - training config
- `tests/` - unit tests (pytest, dev-only, not in prod image)

## Future work

- Further model tuning
- Re-examine merge threshold if Smarti's real test set behaves
  differently than our val set
- Clean up / re-review training labels