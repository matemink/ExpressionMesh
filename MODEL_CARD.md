# Model card

## Model details

The committed `model` artifact is a scikit-learn `RandomForestClassifier`. It expects 1,404
features: the x, y, and z coordinates of 468 MediaPipe Face Mesh landmarks after subtracting
the minimum value on each axis.

| Property | Value |
| --- | --- |
| Classes | happy (0), sad (1), surprised (2) |
| Feature extractor | MediaPipe Face Mesh, 468 landmarks |
| Estimator | Random Forest classifier, 100 trees |
| Serialized with | scikit-learn 1.4.1.post1 |
| Training data | Stable Diffusion v1.5 synthetic portraits |

## Intended use

This artifact supports a local computer-vision experiment and webcam demonstration. It is
useful for exploring a complete synthetic-data-to-inference pipeline. It is not intended for
psychological assessment, identity inference, surveillance, hiring, access control, or any
safety-critical decision.

## Evaluation status

The original source images, train/test split, and metrics were not committed with the legacy
artifact. Therefore, the repository does not claim a validated accuracy for this model. The
original webcam code establishes the artifact's mapping as happy (0), sad (1), and surprised
(2). A later generation script also created angry images, but the committed model was not
retrained with a fourth class. The current pipeline keeps the verified three-class contract.

The current training command writes its split, seed, accuracy, and confusion matrix to a JSON
file so future artifacts can be reproduced and compared.

## Limitations

- Facial expression is not a reliable measurement of a person's internal emotional state.
- Training portraits were generated rather than collected from a representative real-world
  dataset, so domain shift is expected on webcam images.
- Stable Diffusion and MediaPipe may introduce demographic, pose, lighting, and occlusion bias.
- Minimum subtraction provides translation normalization but not rotation or scale invariance.
- The three labels cannot represent mixed, subtle, culturally dependent, or neutral expressions.

## Security and privacy

The webcam demo processes frames locally and does not save or transmit them. Python pickle can
execute code while loading, so use `expression-mesh-model-info` and
`expression-mesh-webcam` only with a model artifact you trust.
