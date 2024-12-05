import os
import sys
import pandas as pd
import ast
import argparse
from config import WANDB_PROJECT, WANDB_USERNAME, WANDB_MODE
from utils.generator import Generator
from utils.verifier import Verifier
from utils.filtration import Filter
import wandb

def main(input_dir, output_dir, dataset):
    wandb.init(project=WANDB_PROJECT, entity=WANDB_USERNAME, mode=WANDB_MODE)

    generator = Generator()
    verifier = Verifier()
    filtration = Filter()

    df = pd.read_csv(os.path.join(input_dir, dataset))
    df["string"] = df["response"]

    generated_df = pd.DataFrame()
    generated_df["string"] = df.apply(lambda x: "Instruction: " + x["instruction"][-1] + "\nResponse: " + str(x["response"]), axis=1)

    features = generator.analyze(generated_df)

    features_file_path = os.path.join(output_dir, "features.txt")
    with open(features_file_path, "w") as file:
        file.write("\n".join(features))

    wandb.save(features_file_path)

    filtered_features = filtration.filter(features)

    filtered_features_file_path = os.path.join(output_dir, "filtered_features.txt")
    with open(filtered_features_file_path, "w") as file:
        file.write("\n".join(filtered_features))

    wandb.save(filtered_features_file_path)

    verified_df = verifier.process(df["string"].to_list(), filtered_features)
    verified_df_path = os.path.join(output_dir, "verified_df.csv")
    verified_df.to_csv(verified_df_path, index=False)

    wandb.save(verified_df_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate feature propositions for a dataset.")
    parser.add_argument("input_dir", nargs='?', type=str, help="Directory to get the initial dataset and features from", default = "data/generation")
    parser.add_argument("output_dir", nargs='?', type=str, help="Directory to save the features produced by the system", default= "data/generation")
    parser.add_argument("dataset", nargs='?', type=str, help="Name of the dataset with the strings to be analyzed", default= "instruction_dataset.csv")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.dataset)