# Wood Board Knot Detection

A Python program that detects knots in wood boards from sequential camera
frames, using an Ultralytics YOLO11n model, and outputs knot bounding boxes
per board in board space (not frame space).

## How it works

Boards are captured as a sequence of overlapping frames, named
`{boardIndex}_{frameNumber}.png`. The program:

1. Groups frames by board index and sorts them by frame number.
2. Runs YOLO inference on each frame independently.
3. Stitches per-frame detections into board space by accumulating each
   frame's pixel offset along the board's long axis, then normalizing to the
   final board width/height.
4. Saves, per board:
   - `{board_id}.txt` - detected boxes in YOLO format (`cls x_center y_center
     width height`, normalized to the full board), the same format as this
     project's own training labels
   - `{board_id}.png` - the stitched board image with detected boxes drawn
   - `{board_id}/` - per-frame results

## Output format: xywh (YOLO) bounding boxes

The dataset's ground-truth labels are YOLO-style bounding boxes, so the
output uses the same format. No segmentation masks exist to train or
validate a true polygon-tracing model against, so that approach wasn't
pursued.

## Output space: per-board, not per-frame

Output is composed at the board level, not per-frame. All box coordinates
in `{board_id}.txt` are normalized to the full stitched board, not to any
individual frame.

## Known limitations

- **Knots spanning two frames.** Frames overlap, and a knot near a frame
  boundary can be detected once in each of the two adjacent frames. These
  are not currently merged - they appear as two separate (overlapping)
  boxes in the board-space output. See [Future work](#future-work).
- **Duplicate detections.** Independent of the above, the model
  occasionally produces more than one detection for the same knot within a
  single frame - an NMS tuning issue (`iou=` in `model.predict()`), not a
  board-stitching issue. See [Future work](#future-work).

## Training

- Model: YOLO11n, trained from Ultralytics pretrained weights. Full
  training configuration is in `src/model/args.yaml`, shipped alongside the
  model weights.
- Board 1043 had a corrupted label and was excluded from training/val
  statistics; it was split into the validation set manually.
- The train/val split is done at the **board** level, not frame level,
  since frames within a board overlap - a frame-level split would leak the
  same physical knots into both sets.
- The split is stratified by knot size (max knot area per board, bucketed
  into quartiles) and targets roughly 70/30 train/val, so both sets get a
  proportional mix of boards with small and large knots rather than
  relying on random luck.

## Test mode

Set `MODE=test` (vs `MODE=infer`) to compare this program's board-space
output against a ground-truth test set, rather than validating the model
in isolation (which `model.val()` would do, frame-space only).

**Test input** (`TEST_DIR_PATH`): frames in the usual
`{boardIndex}_{frameNumber}.png` naming, plus one ground-truth label file
per board, `{board_id}.txt`, in the same board-space YOLO format as this
program's own output.

**Test output**, into a `compare/` subfolder (cleared at the start of each
run): the usual inference output, plus per board:

- ground-truth boxes drawn on top of `{board_id}.png` in a distinct
  color/thickness, for visual comparison
- `{board_id}_compare.json` - `matched`, `missed`, `false_positives`, and
  `mean_iou_matched`, from IoU-based matching (`ultralytics.utils.metrics.box_iou`
  + Hungarian assignment via `scipy.optimize.linear_sum_assignment`, so a
  duplicate detection can't double-count a match) at an IoU threshold of
  0.5 (configurable)

A `summary.json` aggregates pooled counts across the whole test set.

## Running

Configure via `.env` (see `.env.example`):

| Variable         | Purpose                                              |
|------------------|-------------------------------------------------------|
| `MODE`           | `infer` for normal operation, `test` for test mode    |
| `INPUT_DIR_PATH` | Host path to input frames (inference mode)            |
| `OUTPUT_DIR_PATH`| Host path for inference output                        |
| `TEST_DIR_PATH`  | Host path to the test set (frames + GT labels)         |
| `LOG_DIR_PATH`   | Host path for log output                               |
| `UID` / `GID`    | Container user/group ID (match host user for file perms) |


Build and run via Docker Compose. Two compose files are provided:

- `docker-compose.yml` - production image, runs pre-built code baked into
  the image (no source bind mount, no debug port)
- `docker-compose.dev.yml` - development image, bind-mounts `./src` and
  exposes a debug port

```bash
# production
docker compose up

# development
docker compose -f docker-compose.dev.yml up
```


## Project structure

- `main.py` - entry point, mode branching (`infer` vs `test`)
- `model_runner.py` - wraps the YOLO model; runs inference per board,
  per-frame, with per-board error isolation (a corrupted or failing frame
  means that frame's board can't be reconstructed in board space, since its
  width is unknown - the whole board is skipped and logged, not just that
  frame)
- `board.py` - `Board`/`Frame` data model; board grouping, board-space
  coordinate accumulation, stitched-image + label output
- `test.py` - test-mode comparison logic (label loading, IoU-based
  matching, JSON reporting)
- `util.py` - stateless helpers: path resolution, label I/O, coordinate
  format conversion, box drawing
- `logger.py` - logging setup (rotating file handler + console output)
- `model/args.yaml` - training configuration for the shipped model weights

## Future work

- **`iou=` tuning** in `model.predict()` to reduce remaining duplicate
  detections within a single frame (separate from the `conf=` threshold)
- **Cross-frame knot merging** - knots spanning a frame boundary currently
  appear as two overlapping boxes rather than one merged box; a reasonable
  approach would be merging boxes that touch/overlap near a frame edge
- Further model tuning/training iteration