import os
import gc
import csv
import re
from math import log2, pow
import argparse
import pickle

import joblib
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from scipy.sparse import dok_matrix
from sklearn.svm import SVC

pd.options.mode.chained_assignment = None

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

    parser.add_argument(
        "module_name",
        type=str,
        help="The module you want to run. Options available: 'eval', 'analyze', 'classify'",
    )

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
        help="The input file of the analyze module",
    )

    parser.add_argument(
        "--text_analysis_output_file",
        nargs="?",
        type=str,
        default="text_analysis.csv",
        help="The output file of the analyze module",
    )

    parser.add_argument(
        "--text_classification_file",
        nargs="?",
        type=str,
        default="train.txt",
        help="The input file of the classify module",
    )

    parser.add_argument(
        "--text_classification_output_file",
        nargs="?",
        type=str,
        default="text_classification.csv",
        help="The output file of the classify module",
    )

    parser.add_argument(
        "--svm_model_file",
        nargs="?",
        type=str,
        default="svm_model.joblib",
        help="The filename that the SVM model will be saved in and loaded from.",
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
                    dg.append(score / log2(i + 1))

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
                    idg.append(score / log2(i + 1))

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


def get_highest_probability_topic(input_list):
    # Create a dictionary to store sums and counts for each id
    id_sum_count = {}

    # Iterate through the input list
    for tup in input_list:
        id, value = tup
        if id in id_sum_count:
            id_sum_count[id] += value
        else:
            id_sum_count[id] = value

    # Create a new list of tuples with id and average value
    result_list = [(id, id_sum_count[id] / len(input_list)) for id in id_sum_count]
    return sorted(result_list, key=lambda x: x[1], reverse=True)[0]


def top_topics_str_to_list(s, dict):
    matches = re.findall(r'"(\d+)"\s*\+\s*([\d.]+)', s)
    return [(dict[int(num)], float(score)) for num, score in matches]


def analyze(text_analysis_file, text_analysis_output_file, top_k=10, num_topics=20):
    print("Creating dataframe...")
    corpora_df = pd.read_csv(text_analysis_file, sep="\t", names=["class", "terms"])
    corpora_df["terms"] = corpora_df["terms"].apply(text_to_terms)

    term_map = {}
    for i, i_row in corpora_df.iterrows():
        print(f"Processing rows {i+1}/{len(corpora_df)}")
        i_class = i_row["class"]

        # Create a boolean mask for rows with the same class as i_row
        same_class_mask = corpora_df["class"] == i_class

        for term in i_row["terms"]:
            # Count occurrences for each combination of terms and classes using vectorized operations
            N11 = (
                (same_class_mask) & (corpora_df["terms"].apply(lambda x: term in x))
            ).sum()
            N10 = (
                (~same_class_mask) & (corpora_df["terms"].apply(lambda x: term in x))
            ).sum()
            N01 = (
                (same_class_mask) & (~corpora_df["terms"].apply(lambda x: term in x))
            ).sum()
            N00 = (
                (~same_class_mask) & (~corpora_df["terms"].apply(lambda x: term in x))
            ).sum()

            current_value = term_map.get((i_class, term), (0, 0, 0, 0))

            # Use numpy maximum function for element-wise maximum
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

    del mutual_information, chi_squared
    gc.collect()

    # Run LDA
    print("Creating LDA model...")
    all_docs = corpora_df["terms"].tolist()
    all_docs_dict = Dictionary(all_docs)
    all_docs_corpus = [all_docs_dict.doc2bow(terms) for terms in all_docs]
    lda_model = LdaModel(corpus=all_docs_corpus, num_topics=num_topics)

    print("Getting topics of each document...")
    corpora_df["transformed_doc"] = corpora_df["terms"].apply(
        lambda x: all_docs_dict.doc2bow(x)
    )
    corpora_df["topics"] = corpora_df["transformed_doc"].apply(
        lambda x: lda_model.get_document_topics(x)
    )

    print("Finding the average probability of each topic per corpus...")
    topics_df = corpora_df.groupby("class").agg({"topics": "sum"}).reset_index()
    topics_df["top_topic_probability"] = topics_df["topics"].apply(
        get_highest_probability_topic
    )
    topics_df = topics_df.drop("topics", axis=1)

    topics_df["topic_token"] = topics_df["top_topic_probability"].apply(
        lambda x: all_docs_dict[x[0]]
    )

    topics_df["top_10_tokens"] = topics_df["top_topic_probability"].apply(
        lambda x: top_topics_str_to_list(lda_model.print_topic(x[0]), all_docs_dict)
    )

    topics_df.to_csv(
        text_analysis_output_file.replace(".csv", "_topics.csv"), index=False
    )


def preprocess_tweet(text):
    return [
        word
        for word in re.split(r"[^a-zA-Z0-9]+", re.sub(r"http\S+", "", text).lower())
        if len(word) > 0
    ]


def tweets_df_to_model_input(df, term_dict):
    X = dok_matrix((len(df), len(term_dict)), dtype=np.int32)
    for i, bow_terms in enumerate(df["bow_terms"]):
        for term in bow_terms:
            X[i, term] += 1
    return X


def tweets_df_to_model_output(df):
    categories = df["sentiment"].unique()
    categories_dict = {}
    for i, c in enumerate(categories):
        categories_dict[c] = i
    y = df["sentiment"].apply(lambda x: categories_dict[x]).values
    return y


def model_output_to_sentiment(output, dict):
    sentiments = []
    for out in output:
        sentiments.append(list(dict.keys())[list(dict.values()).index(out)])
    return sentiments


def create_single_tweet_model_input(tweet, term_dict):
    df = {"tweet": [tweet]}
    df = pd.DataFrame(df)
    df["tokenized_tweet"] = df["tweet"].apply(preprocess_tweet)
    df["bow_terms"] = df["tokenized_tweet"].apply(
        lambda x: [term_dict[term] for term in x if term in term_dict]
    )
    X = dok_matrix((len(df), len(term_dict)), dtype=np.int32)
    for i, bow_terms in enumerate(df["bow_terms"]):
        for term in bow_terms:
            X[i, term] += 1
    return X


def classify(
    text_classification_file,
    text_classification_output_file,
    svm_model_file,
    random_seed=0,
    train_fraction=0.9,
    load_saved_model=True,
):
    tweets_df = pd.read_csv(text_classification_file, delimiter="\t")
    tweets_df = tweets_df.sample(frac=1, random_state=random_seed)

    train_size = int(len(tweets_df) * train_fraction)
    test_size = len(tweets_df) - train_size
    tweets_train_df = tweets_df.head(train_size)
    tweets_test_df = tweets_df.tail(test_size)
    tweets_train_df["tokenized_tweet"] = tweets_train_df["tweet"].apply(
        preprocess_tweet
    )
    tweets_test_df["tokenized_tweet"] = tweets_test_df["tweet"].apply(preprocess_tweet)

    # NON DETERMINISTIC
    train_terms = set()
    for tokens in tweets_train_df["tokenized_tweet"].tolist():
        train_terms.update(tokens)
    term_dict = {}
    for i, term in enumerate(train_terms):
        term_dict[term] = i
    ###

    ### DO THIS ###
    if load_saved_model and os.path.isfile(svm_model_file):
        model = joblib.load(svm_model_file)
        with open("saved_dictionary.pkl", "wb") as f:
            pickle.dump(term_dict, f)
    else:
        X = tweets_df_to_model_input(tweets_train_df, term_dict)
        y = tweets_df_to_model_output(tweets_train_df)
        model = SVC(C=1000)
        print("Training model...")
        model.fit(X, y)
        joblib.dump(model, svm_model_file)

    tweets_train_df["bow_terms"] = tweets_train_df["tokenized_tweet"].apply(
        lambda x: [term_dict[term] for term in x]
    )

    tweets_test_df["bow_terms"] = tweets_test_df["tokenized_tweet"].apply(
        lambda x: [term_dict[term] for term in x if term in term_dict]
    )

    classes = tweets_df["sentiment"].unique()
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i

    if load_saved_model and os.path.isfile(svm_model_file):
        model = joblib.load(svm_model_file)
    else:
        X = tweets_df_to_model_input(tweets_train_df, term_dict)
        y = tweets_df_to_model_output(tweets_train_df)
        model = SVC(C=1000)
        print("Training model...")
        model.fit(X, y)
        joblib.dump(model, svm_model_file)

    ### TESTING
    tweet = "love you"
    X = create_single_tweet_model_input(tweet, term_dict)
    y_pred = model.predict(X)
    print(y_pred)
    s = model_output_to_sentiment(y_pred, classes_dict)
    print(s)


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
            classify(
                args.text_classification_file,
                args.text_classification_output_file,
                args.svm_model_file,
            )
        case _:
            print("Module not supported.")
            exit(-1)
    print("Done")
