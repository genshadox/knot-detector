# Wood Board Knot Detection

Detects knots in wood boards from sequential camera frames using YOLO11n,
and outputs knot bounding boxes per board (board-space, not frame-space).

## How it works

Frames are named `{boardIndex}_{frameNumber}.png`. Assumed non-overlapping,
no gaps, per the task brief.

1. Group frames by board, sort by frame number.
2. Run YOLO inference per frame.
3. Stitch detections into board space (accumulate x-offset per frame,
   normalize at the end).
4. Save per board:
   - `{board_id}.txt` - detections in YOLO format, board-space
   - `{board_id}.png` - stitched board image with boxes drawn
   - `{board_id}/` - per-frame results

## Output format

Bounding boxes (YOLO xywh - board space), not traced polygons - no segmentation ground
truth exists to train or validate against.

## Known limitations

- **Cross-frame knots**: a knot at a frame boundary can be detected in
  both adjacent frames; not currently merged.
- **Duplicate detections**: near-duplicate boxes are handled by NMS
  tuning (see below). Fully-nested box-in-box duplicates are not - IoU is
  low for these regardless of NMS threshold, so they need a separate
  containment check (not implemented). See [Future work](#future-work).

## Training

- YOLO11n, pretrained weights. Config: `model/args.yaml`.
- Board 1043 had a corrupted label - excluded from stats, split into train/val
  manually.
- Train/val split is per-board (not per-frame, since frames within a
  board overlap in the training set) and stratified by max knot size per
  board, targeting ~70/30.
- Inference NMS `iou=0.5` (default 0.7) - lowered to cut duplicate
  detections. Checked against GT: real knots never exceed 0.5 IoU with
  each other, so this doesn't risk merging distinct knots.

## Test mode

`MODE=test` runs the full pipeline and compares board-space output
against a ground-truth test set (rather than validating the model alone,
frame-space, like `model.val()` would).

**Input** (`TEST_DIR_PATH`): frames + one `{board_id}.txt` GT label file
per board, same board-space format as this program's output.

**Output** (`compare/` subfolder, cleared each run):
- GT boxes drawn over `{board_id}.png` for visual comparison
- `{board_id}_compare.json`: `matched` / `missed` / `false_positives` /
  `mean_iou_matched` / `precision` / `recall`, via IoU matching (`box_iou` + Hungarian assignment,
  avoids double-counting duplicates) at IoU ≥ 0.5
- `summary.json`: pooled totals across the test set

## Running

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
| `UID` / `GID`    | container user/group (match host)        |

## Project structure

- `main.py` - entry point, mode branching
- `model_runner.py` - YOLO inference, per-board error isolation
- `board.py` - `Board`/`Frame` model, board-space stitching
- `test.py` - test-mode comparison
- `util.py` - path/label/coordinate helpers
- `logger.py` - logging setup
- `model/args.yaml` - training config

## Future work

- Containment-based filter for nested duplicate boxes (`bbox_ioa`)
- Merge knots split across a frame boundary
- Further model tuning
- Use frame overlap so boundary knots appear whole in at least one frame