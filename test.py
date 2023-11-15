import pandas as pd
import numpy as np

# Read input files
system_results = pd.read_csv("ttdssystemresults.csv")
qrels = pd.read_csv("qrels.csv")

# Initialize an empty list to store individual results
results_list = []

# Define functions for evaluation metrics


def precision_at_k(relevant_docs, k):
    # Calculate precision at cutoff k
    return sum(relevant_docs[:k]) / k


def recall_at_k(relevant_docs, k):
    # Calculate recall at cutoff k
    return sum(relevant_docs[:k]) / len(relevant_docs)


def r_precision(relevant_docs):
    # Calculate r-precision
    return sum(relevant_docs) / len(relevant_docs)


def average_precision(relevant_docs):
    # Calculate average precision
    precision_values = [
        precision_at_k(relevant_docs, k + 1) for k in range(len(relevant_docs))
    ]
    return np.mean(precision_values)


def discounted_cumulative_gain(relevant_docs):
    # Calculate discounted cumulative gain
    dcg = sum(
        [(2**rel - 1) / np.log2(rank + 2) for rank, rel in enumerate(relevant_docs)]
    )
    return dcg


def normalized_discounted_cumulative_gain(relevant_docs, k):
    # Calculate normalized discounted cumulative gain at cutoff k
    ideal_dcg = discounted_cumulative_gain(sorted(relevant_docs, reverse=True))
    dcg = discounted_cumulative_gain(relevant_docs[:k])
    return dcg / ideal_dcg


# Iterate through each system and query to calculate evaluation metrics
for system_number in range(1, 7):
    for query_number in range(1, 11):
        # Filter results for the current system and query
        system_query_results = system_results[
            (system_results["system_number"] == system_number)
            & (system_results["query_number"] == query_number)
        ]

        # Merge with qrels to get relevance information
        merged_data = pd.merge(
            system_query_results,
            qrels,
            left_on=["query_number", "doc_number"],
            right_on=["query_id", "doc_id"],
            how="left",
        )
        merged_data.fillna(0, inplace=True)

        # Sort by rank
        merged_data.sort_values(by="rank_of_doc", inplace=True)

        # Get relevant documents and calculate evaluation metrics
        relevant_docs = list(merged_data["relevance"])
        precision_10 = precision_at_k(relevant_docs, 10)
        recall_50 = recall_at_k(relevant_docs, 50)
        r_precision_value = r_precision(relevant_docs)
        average_precision_value = average_precision(relevant_docs)
        ndcg_10 = normalized_discounted_cumulative_gain(relevant_docs, 10)
        ndcg_20 = normalized_discounted_cumulative_gain(relevant_docs, 20)

        # Append results to the list
        results_list.append(
            {
                "system_number": system_number,
                "query_number": query_number,
                "P@10": precision_10,
                "R@50": recall_50,
                "r-precision": r_precision_value,
                "AP": average_precision_value,
                "nDCG@10": ndcg_10,
                "nDCG@20": ndcg_20,
            }
        )

# Calculate mean values for each system
mean_results = pd.DataFrame(results_list).groupby("system_number").mean().reset_index()
mean_results["query_number"] = "mean"

# Concatenate individual results and mean results using pd.concat
eval_results = pd.concat([pd.DataFrame(results_list), mean_results], ignore_index=True)

# Write results to ir_eval.csv
eval_results.to_csv("ir_eval.csv", index=False)
