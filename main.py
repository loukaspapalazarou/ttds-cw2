import argparse
import pandas as pd


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(description="Welcome to my TTDS coursework 2")

    parser.add_argument("module_name", help="The module you want to run")

    parser.add_argument(
        "--system_results_file",
        nargs="?",
        type=str,
        default="ttdssystemresults.csv",
        help="The file containing the retrieval results of a given IR system",
    )

    parser.add_argument(
        "--query_relevance_file",
        nargs="?",
        type=str,
        default="qrels.csv",
        help="A file that has the list of relevant documents for each of the queries",
    )

    parser.add_argument(
        "--eval_output_file",
        nargs="?",
        type=str,
        default="ir_eval.csv",
        help="The output file name of the eval module",
    )

    args = parser.parse_args()
    return args


def eval(
    system_results_file,
    query_relevance_file,
    eval_output_file,
    precision_cutoff=10,
    recall_cutoff=50,
):
    system_results = pd.read_csv(system_results_file)
    qrels = pd.read_csv(query_relevance_file)

    final_values = []

    for system_id in system_results["system_number"].unique().tolist():
        for query_id in system_results["query_number"].unique().tolist():
            # get relevant docs given a query
            relevant_docs = qrels.loc[qrels["query_id"] == query_id, "doc_id"].tolist()
            # get docids given a system and a query
            total_retrieved = system_results.loc[
                (system_results["system_number"] == system_id)
                & (system_results["query_number"] == query_id),
                "doc_number",
            ].tolist()

            # Calculate precision
            relevant_and_retrieved_precision = [
                doc
                for doc in total_retrieved[:precision_cutoff]
                if doc in relevant_docs
            ]
            precision = (
                len(relevant_and_retrieved_precision)
                / len(total_retrieved[:precision_cutoff])
                if len(total_retrieved[:precision_cutoff]) > 0
                else 0
            )

            # Calculate recall
            relevant_and_retrieved_recall = [
                doc for doc in total_retrieved[:recall_cutoff] if doc in relevant_docs
            ]
            recall = (
                len(relevant_and_retrieved_recall) / len(relevant_docs)
                if len(relevant_docs) > 0
                else 0
            )

            # Calculate R-precision
            r_precision_cutoff = len(relevant_docs)
            r_precision_relevant_and_retrieved_precision = [
                doc
                for doc in total_retrieved[:r_precision_cutoff]
                if doc in relevant_docs
            ]
            r_precision = (
                len(r_precision_relevant_and_retrieved_precision)
                / len(total_retrieved[:r_precision_cutoff])
                if len(total_retrieved[:r_precision_cutoff]) > 0
                else 0
            )

            # Append precision, recall, and R-precision values to the list
            final_values.append(
                {
                    "system_number": system_id,
                    "query_number": query_id,
                    "P@10": precision,
                    "R@50": recall,
                    "r-precision": r_precision,
                }
            )

        # Calculate mean values for each statistic after each system_id
        mean_values = {
            "system_number": system_id,
            "query_number": "mean",
            "P@10": pd.DataFrame(final_values)["P@10"].mean(),
            "R@50": pd.DataFrame(final_values)["R@50"].mean(),
            "r-precision": pd.DataFrame(final_values)["r-precision"].mean(),
        }

        final_values.append(mean_values)

    # Create a DataFrame from the precision, recall, and R-precision values
    final_values_df = pd.DataFrame(final_values)
    final_values_df = final_values_df.round(3)
    pd.set_option("display.max_rows", None)
    print(final_values_df)


if __name__ == "__main__":
    args = get_args()
    match args.module_name.lower():
        case "eval":
            eval(
                args.system_results_file,
                args.query_relevance_file,
                args.eval_output_file,
            )
        case "analyze":
            raise NotImplementedError
        case "classify":
            raise NotImplementedError
        case _:
            print("Module not supported.")
