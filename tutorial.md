# Training a custom AutoMorph model

The current `/custom` workflow trains a YOLO oriented-box detector and one dlib
landmark predictor per class. It differs from the older standalone shape-predictor
workflow shown in [the Drosophila recording](drosophila_training.webm).

1. **Create Project:** open **Train Custom Model Wizard** from the landing page;
   enter a project name and organism.
2. **Add Data:** upload a ZIP, or select one TPS/XML annotation file and every
   referenced image together. ZIPs may contain subdirectories; annotation paths
   must resolve unambiguously. Use at least two annotated images, consistent class
   names and landmark names/order, and decoded image dimensions matching the
   annotations. Retain specimen identifiers separately for study-level splitting.
3. **Check Data:** inspect every displayed box and landmark overlay. Confirm image
   orientation, labels, and padding; do not proceed with reflected, displaced, or
   missing points. TPS coordinate conversion depends on the actual image height.
4. **Train Model:** choose a preset or expert parameters, then **Start Training Job**.
   Defaults include nu 0.1, tree depth 4, cascade depth 15, oversampling 5,
   feature pool 400, candidate tree splits 20, and validation fraction 0.2.
   Candidate tree splits are a model hyperparameter, not cross-validation folds.
5. **Follow Progress:** keep the job ID. Training runs in a separate process;
   concurrent submissions to the same orchestrator are rejected as busy, not queued.
   Cancelling or closing the backend interrupts training; retain the error/log if
   the job fails. Do not treat the presence of a model file as proof of completion.
6. **Review Results:** inspect detector metrics and per-class landmark error.
   The landmark value is an internal crop-space error, not an end-to-end accuracy
   guarantee. Read the [metric limitations](docs/RESEARCH_HANDOFF.md#interpreting-results).
7. **Publish & Use:** select the registered model for prediction. Verify predictions
   on new images and export annotations through the UI. Preserve unedited outputs
   separately from human corrections.

The generic upload limit defaults to 1 GiB locally and 512 MiB with
`AUTOMORPH_HOSTED=true` (legacy alias `LIZARDMORPH_HOSTED`). Override with
`TRAINING_MAX_BYTES`. Archive defaults are 20,000 files and 1 GiB uncompressed,
configured by `TRAINING_MAX_ARCHIVE_FILES` and `TRAINING_MAX_UNCOMPRESSED_BYTES`.
Older `/train_predictor` endpoints have different limits; do not use their old
100 MB/500-file documentation for this wizard.

For reproducible work, archive the complete training job folder and follow the
[research handoff guide](docs/RESEARCH_HANDOFF.md). The tiny sample dataset is for
workflow checks, not an accuracy study.
