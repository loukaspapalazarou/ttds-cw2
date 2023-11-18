import argparse
import math
import pandas as pd
import re
from nltk.stem import PorterStemmer
import numpy as np
from math import log2, pow
import gc
import csv
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary

STOPWORDS_FILENAME = "ttds_2023_english_stop_words.txt"
STEMMER = PorterStemmer()

STOPPING_ENABLED = True
STOPWORDS = []

if STOPPING_ENABLED:
    with open(file=STOPWORDS_FILENAME, mode="r") as stopwords_file:
        STOPWORDS = set(stopwords_file.read().splitlines())


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

    parser.add_argument(
        "--text_analysis_file",
        nargs="?",
        type=str,
        default="train_and_dev.tsv",
        help="The output file name of the eval module",
    )

    parser.add_argument(
        "--text_analysis_output_file",
        nargs="?",
        type=str,
        default="text_analysis.csv",
        help="The output file name of the eval module",
    )

    parser.add_argument(
        "--text_classification_file",
        nargs="?",
        type=str,
        default="train.txt",
        help="The output file name of the eval module",
    )

    args = parser.parse_args()
    return args


def eval(system_results_file, query_relevance_file, eval_output_file, verbose=False):
    system_results = pd.read_csv(system_results_file)
    qrels = pd.read_csv(query_relevance_file)

    precision_cutoff = 10
    recall_cutoff = 50
    nDCG_10 = 10
    nDCG_20 = 20
    nDCG_cutoff = max(nDCG_10, nDCG_20)

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

            # Calculate Average Precision
            precision_at_relevant = 0
            num_relevant_docs = len(relevant_docs)
            num_retrieved_docs = 0
            for i, doc in enumerate(total_retrieved):
                if doc in relevant_docs:
                    num_retrieved_docs += 1
                    precision_at_relevant += num_retrieved_docs / (i + 1)

            average_precision = (
                (precision_at_relevant / num_relevant_docs)
                if num_relevant_docs > 0
                else 0
            )

            # Calculate nDCG
            relevance_scores = (
                qrels[qrels["query_id"] == query_id]
                .set_index("doc_id")["relevance"]
                .to_dict()
            )

            ndcg_docs = total_retrieved[:nDCG_cutoff]

            g = []
            for doc in ndcg_docs:
                if doc in relevant_docs:
                    g.append(relevance_scores[doc])
                else:
                    g.append(0)

            dg = []
            for i, score in enumerate(g):
                if i == 0:
                    dg.append(score)
                elif score == 0:
                    dg.append(0)
                else:
                    dg.append(score / math.log2(i + 1))

            dcg = [dg[0]]
            for i in range(1, len(dg)):
                dcg.append(dcg[i - 1] + dg[i])

            ig = sorted(g, reverse=True)
            idg = []
            for i, score in enumerate(ig):
                if i == 0:
                    idg.append(score)
                elif score == 0:
                    idg.append(0)
                else:
                    idg.append(score / math.log2(i + 1))

            idcg = [idg[0]]
            for i in range(1, len(idg)):
                idcg.append(idcg[i - 1] + idg[i])

            ndcg = [dcg[i] / idcg[i] if idcg[i] != 0 else 0 for i in range(nDCG_cutoff)]

            # Append precision, recall, R-precision, AP, nDCG at 10, and nDCG at 20 values to the list
            final_values.append(
                {
                    "system_number": system_id,
                    "query_number": query_id,
                    "P@10": precision,
                    "R@50": recall,
                    "r-precision": r_precision,
                    "AP": average_precision,
                    "nDCG@10": ndcg[nDCG_10 - 1],
                    "nDCG@20": ndcg[nDCG_20 - 1],
                }
            )

        # Calculate mean values for each statistic after each system_id
        mean_values = {
            "system_number": system_id,
            "query_number": "mean",
            "P@10": pd.DataFrame(final_values)["P@10"].mean(),
            "R@50": pd.DataFrame(final_values)["R@50"].mean(),
            "r-precision": pd.DataFrame(final_values)["r-precision"].mean(),
            "AP": pd.DataFrame(final_values)["AP"].mean(),
            "nDCG@10": pd.DataFrame(final_values)["nDCG@10"].mean(),
            "nDCG@20": pd.DataFrame(final_values)["nDCG@20"].mean(),
        }

        final_values.append(mean_values)

    # Create a DataFrame from the precision, recall, R-precision, AP, nDCG at 10, and nDCG at 20 values
    final_values_df = pd.DataFrame(final_values)
    final_values_df = final_values_df.round(3)
    final_values_df.to_csv(eval_output_file, index=False)
    if verbose:
        pd.set_option("display.max_rows", None)
        print(final_values_df)


def text_to_terms(text):
    return [
        STEMMER.stem(word)
        for word in re.split(r"[^a-zA-Z0-9]+", text.lower())
        if word and word not in STOPWORDS
    ]


def calc_mutual_information(N_values):
    N11 = N_values[0]
    N10 = N_values[1]
    N01 = N_values[2]
    N00 = N_values[3]
    N = sum(N_values)
    N1_dot = N10 + N11
    dot_N1 = N01 + N11
    N0_dot = N00 + N01
    dot_N0 = N00 + N10

    try:
        t1 = (N11 / N) * log2((N * N11) / (N1_dot * dot_N1))
    except Exception:
        t1 = 0
    try:
        t2 = (N01 / N) * log2((N * N01) / (N0_dot * dot_N1))
    except Exception:
        t2 = 0
    try:
        t3 = (N10 / N) * log2((N * N10) / (N1_dot * dot_N0))
    except Exception:
        t3 = 0
    try:
        t4 = (N00 / N) * log2((N * N00) / (N0_dot * dot_N0))
    except:
        t4 = 0

    return t1 + t2 + t3 + t4


def calc_chi_squred(N_values):
    N11 = N_values[0]
    N10 = N_values[1]
    N01 = N_values[2]
    N00 = N_values[3]

    t1 = sum(N_values) * pow(N11 * N00 - N10 * N01, 2)
    t2 = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00)
    try:
        return t1 / t2
    except Exception:
        return 0


def analyze(text_analysis_file, text_analysis_output_file):
    corpora_df = pd.read_csv(text_analysis_file, sep="\t", names=["class", "terms"])
    corpora_df["terms"] = corpora_df["terms"].apply(text_to_terms)

    top_k = 10
    term_map = {}
    for i, i_row in corpora_df.iterrows():
        print(f"Processing rows {i+1}/{len(corpora_df)}")
        # Accessing the values in each row
        i_class = i_row["class"]
        for term in i_row["terms"]:
            N00, N01, N10, N11 = 0, 0, 0, 0
            for _, j_row in corpora_df.iterrows():
                if i_class == j_row["class"] and term in j_row["terms"]:
                    N11 += 1
                elif i_class != j_row["class"] and term in j_row["terms"]:
                    N10 += 1
                elif i_class == j_row["class"] and term not in j_row["terms"]:
                    N01 += 1
                else:
                    N00 += 1
            current_value = term_map.get((i_class, term), (0, 0, 0, 0))
            updated_value = (
                max(N11, current_value[0]),
                max(N10, current_value[1]),
                max(N01, current_value[2]),
                max(N00, current_value[3]),
            )
            term_map[(i_class, term)] = updated_value

    mutual_information = {}
    for key in term_map:
        classname = key[0]
        term = key[1]
        N_values = term_map[key]
        if classname not in mutual_information:
            mutual_information[classname] = [(term, calc_mutual_information(N_values))]
        else:
            mutual_information[classname].append(
                (term, calc_mutual_information(N_values))
            )
    # Chi Squared
    chi_squared = {}
    for key in term_map:
        classname = key[0]
        term = key[1]
        N_values = term_map[key]
        if classname not in chi_squared:
            chi_squared[classname] = [(term, calc_chi_squred(N_values))]
        else:
            chi_squared[classname].append((term, calc_chi_squred(N_values)))
    del term_map
    gc.collect()

    for key in mutual_information:
        mutual_information[key] = sorted(
            mutual_information[key], key=lambda x: x[1], reverse=True
        )[:top_k]

    for key in chi_squared:
        chi_squared[key] = sorted(chi_squared[key], key=lambda x: x[1], reverse=True)[
            :top_k
        ]

    with open(text_analysis_output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Mutual Information"])
        for classname in mutual_information:
            csvwriter.writerow([classname])
            for term in mutual_information[classname]:
                csvwriter.writerow([term[0], term[1]])

        csvwriter.writerow([])
        csvwriter.writerow(["Chi Squared"])
        for classname in chi_squared:
            csvwriter.writerow([classname])
            for term in chi_squared[classname]:
                csvwriter.writerow([term[0], term[1]])

    # Run LDA
    all_docs = corpora_df["terms"].tolist()
    all_docs_dictionary = Dictionary(all_docs)
    all_docs_corpus = [all_docs_dictionary.doc2bow(text) for text in all_docs]
    lda = LdaModel(all_docs_corpus, num_topics=20)

    for i in lda.print_topics():
        print(i)

    for i in all_docs_dictionary:
        print(i, all_docs_dictionary[i])


def classify(text_classification_file):
    raise NotImplementedError


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
            analyze(args.text_analysis_file, args.text_analysis_output_file)
        case "classify":
            raise NotImplementedError
        case _:
            print("Module not supported.")
