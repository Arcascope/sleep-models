# Arcasleep

Arcascope's open source, open weights sleep models.

---

This is the top-level repository containing several Python packages. They are available on
PyPI as separate packages.

We release them in separate packages for a clean and easy to separate structure, and for ease of publishing model versions clearly on HuggingFace.

## BiDoze

Published on Hugging Face as [`arcascope/arcasleep-bidoze-pediatric`](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric); the package lives in [`bidoze/`](bidoze/).

Wake/Light/Deep/REM predictions, plus a 5th head for identifying gaps in accelerometer. 

1. **Architecture:** `[CNN encoder] -> [ALiBi Transformers] -> [5-head MLP]`.
2. Developed via a semi-autonomous research loop using LLMs.
    1. Agents were given the goal to optimize model architecture and training regime with fixed train/test splits. 
    2. "Best" as measured by the geometric mean of TST and WASO MAPE. 
3. The current production training is based on a clinical pediatric pool with varying levels of sleep apnea.
    1. We are grateful to Glenn Weaver and his group for sharing their data with us.  
        These data is described in [Weaver et al. 2026](https://pubmed.ncbi.nlm.nih.gov/40676371/).