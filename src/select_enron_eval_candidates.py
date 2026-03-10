import os
import pandas as pd

# Input and output files for candidate selection.
cleanInputFilePath = os.path.join("data", "intermediate", "enron_messages_clean.csv")
candidateOutputFilePath = os.path.join("data", "intermediate", "enron_eval_candidates.csv")


def main():
    """
    Build a review pack of candidate real-world messages for manual labelling.

    The selection keeps a mix of:
    - high relevance rows
    - medium relevance rows
    - low relevance rows

    This helps avoid creating an evaluation set that only contains obvious positives.
    """
    if not os.path.exists(cleanInputFilePath):
        print("Clean input file not found:", cleanInputFilePath)
        return

    cleanDataFrame = pd.read_csv(cleanInputFilePath)

    # Prioritise likely event/action messages first.
    highRelevanceRows = cleanDataFrame[cleanDataFrame["relevance_score"] >= 4].head(50)

    # Keep some medium-signal rows as well.
    mediumRelevanceRows = cleanDataFrame[
        (cleanDataFrame["relevance_score"] >= 2) &
        (cleanDataFrame["relevance_score"] < 4)
    ].head(40)

    # Include some likely negatives so the evaluation set is not too easy.
    lowRelevanceRows = cleanDataFrame[cleanDataFrame["relevance_score"] < 2].head(30)

    candidateDataFrame = pd.concat(
        [highRelevanceRows, mediumRelevanceRows, lowRelevanceRows],
        ignore_index=True
    )

    candidateDataFrame = candidateDataFrame.drop_duplicates(subset=["message_id"]).copy()
    candidateDataFrame = candidateDataFrame.sort_values(
        by=["relevance_score", "message_id"],
        ascending=[False, True]
    )

    candidateDataFrame.to_csv(candidateOutputFilePath, index=False)

    print("Candidate selection complete.")
    print("Candidate rows saved:", len(candidateDataFrame))
    print("Candidate output file:", candidateOutputFilePath)


if __name__ == "__main__":
    main()