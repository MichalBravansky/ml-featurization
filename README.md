# Automated Featurization Pipeline

This repository provides a pipeline for generating and selecting important features from a set of strings. The process helps reduce perplexity by identifying key patterns and elements within the strings. The pipeline operates in two stages: **feature generation** and **featurization**.

## Overview

1. **Feature Generation**: Proposes a set of possible features for the given strings and creates a table that indicates whether each feature is present in each string.
   
2. **Featurization**: Selects the most important features by iteratively adding them, minimizing perplexity, and producing a final set of features.

---

## Configuration

Before running the pipeline, ensure that `config.py` is properly set up. This file contains:

- **API_KEY**: Your OpenAI API key for generating and verifying features.
- **get_string_template**: A string template used to generate prompts for measuring perplexity. Customize this to fit your data format.

---

## Feature Generation

In this first stage, we generate and verify potential features. To start:

1. **Prepare Input Data**: Provide a CSV file containing the strings you want to analyze. You may need to adjust the code depending on your data structure.
  
2. **Run Generation**: Execute `generation.py` to generate features, cluster them, and verify their presence within the strings. This process can be run on any machine and does not require GPUs.

### Outputs:
- **features.txt**: A list of all proposed features.
- **filtered_features.txt**: Features filtered by clustering and removal of duplicates.
- **verified_df.csv**: A verification table that shows whether a given feature is present in each string.

To run the generation step:
    **python generation.py --input_dir data/generation**

## Featurization

The second stage is **featurization**, where we select the most impactful features based on their ability to reduce perplexity. This process is GPU-accelerated, iterating over features and adding them one by one until perplexity can no longer be reduced.

### Outputs:
- A final set of features.

To run the featurization step:
    **python featurization.py --input_dir data/generation -output_dir data/featurization**


## Example Test Run

You can test the pipeline using a sample dataset of 50 instruction-response pairs from **ChatArena**.

### Steps:

1. Install the required dependencies:
   **pip install -r requirements.txt**

2. Execute the feature generation step:
   **python generation.py**

3. Run the featurization process to select the most relevant features:
   **python featurization.py**