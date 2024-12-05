import os
import sys
import pandas as pd
import ast
import statistics
import wandb
from datetime import datetime
from config import WANDB_PROJECT, WANDB_USERNAME, WANDB_MODE
from utils.generator import Generator
from utils.verifier import Verifier
from utils.filtration import Filter
from utils.perplexity_evaluator import Evaluator
import argparse

def main(experiment_name, input_dir, output_dir, dataset, num_iterations, batch_size):

    experiment_name = f"{experiment_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_USERNAME, mode=WANDB_MODE, name=experiment_name)
    wandb.run.log_code(".")

    evaluator = Evaluator(batch_size = batch_size)

    df = pd.read_csv(os.path.join(input_dir, dataset))
    df["instruction"] = df["instruction"].apply(ast.literal_eval)
    df["string"] = df["response"]

    df.reset_index(drop = True, inplace = True)
    evaluator.init_cached_prompts(df)

    best_features = []
    df = df[[column for column in df.columns if "_property" not in column]]
    losses = []

    best_features_file = os.path.join(output_dir, "best_features.txt")
    wandb.save(best_features_file)

    verified_df = load_verified_df(df, input_dir)

    for _ in range(num_iterations):

        if best_features:
            best_feature = best_features[-1]
            sub_df = df[df[best_feature + "_property"]]
            temp_evaluated_df = evaluator.evaluate(verified_df[verified_df["string"].isin(sub_df["string"])].reset_index(drop=True), sub_df.reset_index(drop=True), list(sub_df.index), feature_names=best_features)
            temp_evaluated_df.index = verified_df[verified_df["string"].isin(sub_df["string"])].index

            evaluated_df.loc[verified_df[verified_df["string"].isin(sub_df["string"])].index] = temp_evaluated_df
        else:
            evaluated_df = evaluator.evaluate(verified_df, df, list(df.index), feature_names=best_features)

        loss = statistics.mean(evaluated_df["empty"].to_list())
        losses.append(evaluated_df["empty"].to_list())

        best_feature = select_best_feature(evaluated_df, best_features)

        if best_feature == "empty":
            sys.exit("Terminating the program because no additional features that lower perplexity can be found.")

        df[best_feature + "_property"] = verified_df[best_feature]
        best_features.append(best_feature)

        log_results(best_feature, evaluated_df, best_features, best_features_file, loss, output_dir)


def load_verified_df(df, input_dir):
    verified_df = pd.read_csv(os.path.join(input_dir, "verified_df.csv"))
    verified_df = pd.merge(df["string"], verified_df.drop_duplicates(subset=["string"]), on="string", how="left")
    verified_df = verified_df[list(verified_df.columns)[1:] + ["string"]]

    column_sums = verified_df.drop(columns=["string"]).sum(axis=0)
    threshold_sum = len(verified_df) * 0.05
    selected_by_sum = column_sums[column_sums >= threshold_sum].index.tolist() + ["string"]

    return verified_df.loc[:, selected_by_sum]

def select_best_feature(evaluated_df, best_features):
    remaining_features = [col for col in evaluated_df.columns if col not in best_features]
    feature_scores = evaluated_df[remaining_features].sum().sort_values()
    return feature_scores.index[0]

def log_results(best_feature, evaluated_df, best_features, best_features_file, loss, output_dir):
    print(f"Selected feature: {best_feature}")
    print(f"Top 5 features: {evaluated_df[evaluated_df.columns[:-1]].sum().sort_values().index[:5].tolist()}")

    with open(best_features_file, "w") as bf_file:
        bf_file.write("\n".join(best_features) + "\n")

    with open(os.path.join(output_dir, "losses.txt"), "a") as losses_file:
        losses_file.write(f"{loss}\n")
    
    wandb.log({"mean_loss": loss})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate features from a dataset.")
    parser.add_argument("experiment_name", nargs='?', type=str, help="Name of the experiment log into wandb", default= "modeling")
    parser.add_argument("input_dir", nargs='?', type=str, help="Directory to get the initial dataset and proposed features from", default = "data/generation")
    parser.add_argument("output_dir", nargs='?', type=str, help="Directory to save the features produced by the system", default= "data/featurization")
    parser.add_argument("dataset", nargs='?', type=str, help="Name of the dataset with the strings to be analyzed", default= "instruction_dataset.csv")
    parser.add_argument("num_iterations", nargs='?', type=int, help="Number of features to be produced", default= 30)
    parser.add_argument("batch_size", nargs='?', type=int, help="Batch size used for featurization", default= 2)
    args = parser.parse_args()

    main(args.experiment_name, args.input_dir, args.output_dir, args.dataset, args.num_iterations, args.batch_size)