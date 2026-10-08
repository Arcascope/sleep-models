One of the three networks of
[arcasleep-bidoze-adult](https://huggingface.co/arcascope/arcasleep-bidoze-adult).

Load the repository root, not this folder: the released model averages the three networks and
applies the stage divisor in `composite.json`. One network by itself scores 4-stage κ
0.402–0.431 on SleepAccel (cross-validation), against 0.437 for the three together. The
repository's README gives the model's performance, intended use and limitations.
