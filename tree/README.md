# Gradient Boosted Tree

A classical tree model, defined and trained using LightGBM. Features are based on heart rate (bpm) and triaxial accelerometer. These are aggregated and summarized in different ways to form a large family of features. These are saved in large parquet format tables; this format allows us to load individual columns without scanning all rows (of which there are many!) and also benefits from format improvements for handling such large tabular data.

Use this with `arcasleep tree` commands, like `predict`, `train`, or `publish`. To make a file that can be shared and run on watches like the Garmin Instinct 3, use `arcasleep tree monkey-c` (named after the Monkey-C language used in ConnectIQ).