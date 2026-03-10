import os
import pandas as pd

# Input and output files for the Enron labelling template.
candidateInputFilePath = os.path.join("data", "intermediate", "enron_eval_candidates.csv")
templateOutputFilePath = os.path.join("data", "processed", "enron_label_template.csv")


def buildBalancedSample(candidateDataFrame):
    """
    Build a small balanced set of Enron rows for manual labelling.

    The goal is to avoid only taking the top relevance rows, which would make
    the evaluation set too easy and too heavily biased towards obvious positives.
    """
    highRelevanceRows = candidateDataFrame[candidateDataFrame["relevance_score"] >= 4].head(8)
    mediumRelevanceRows = candidateDataFrame[
        (candidateDataFrame["relevance_score"] >= 2) &
        (candidateDataFrame["relevance_score"] < 4)
    ].head(7)
    lowRelevanceRows = candidateDataFrame[candidateDataFrame["relevance_score"] < 2].head(5)

    selectedRows = pd.concat(
        [highRelevanceRows, mediumRelevanceRows, lowRelevanceRows],
        ignore_index=True
    )

    selectedRows = selectedRows.drop_duplicates(subset=["message_id"]).copy()

    return selectedRows.head(20)


def assignSplitValues(selectedDataFrame):
    """
    Assign split values for the manual labelling set.

    Use a smaller dev portion and a larger test portion so the final benchmark
    remains evaluation-focused.
    """
    splitValues = []

    for rowIndex in range(len(selectedDataFrame)):
        if rowIndex < 6:
            splitValues.append("dev")
        else:
            splitValues.append("test")

    selectedDataFrame["split"] = splitValues

    return selectedDataFrame


def buildTemplateDataFrame(selectedDataFrame):
    """
    Convert the selected Enron rows into the project's evaluation schema.

    Gold-label fields are left blank so they can be completed manually.
    """
    templateRows = []

    for rowIndex in range(len(selectedDataFrame)):
        currentRow = selectedDataFrame.iloc[rowIndex]

        templateRows.append({
            "message_id": currentRow["message_id"],
            "source_type": "enron",
            "split": currentRow["split"],
            "sent_at": currentRow["sent_at"],
            "subject": currentRow["subject"],
            "body": currentRow["body"],
            "gold_calendar_event_required": "",
            "gold_event_category": "",
            "gold_event_date": "",
            "gold_start_time": "",
            "gold_end_time": "",
            "gold_action_required": "",
            "gold_action_type": "",
            "gold_action_deadline": "",
            "gold_summary": "",
            "edge_case_tag": ""
        })

    templateDataFrame = pd.DataFrame(templateRows)
    return templateDataFrame


def main():
    """
    Build a 20-row Enron labelling template using a balanced candidate sample.
    """
    if not os.path.exists(candidateInputFilePath):
        print("Candidate input file not found:", candidateInputFilePath)
        return

    candidateDataFrame = pd.read_csv(candidateInputFilePath)

    selectedDataFrame = buildBalancedSample(candidateDataFrame)
    selectedDataFrame = assignSplitValues(selectedDataFrame)

    templateDataFrame = buildTemplateDataFrame(selectedDataFrame)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    templateDataFrame.to_csv(templateOutputFilePath, index=False)

    print("Enron label template created.")
    print("Rows selected for labelling:", len(templateDataFrame))
    print("Dev rows:", len(templateDataFrame[templateDataFrame["split"] == "dev"]))
    print("Test rows:", len(templateDataFrame[templateDataFrame["split"] == "test"]))
    print("Template output file:", templateOutputFilePath)


if __name__ == "__main__":
    main()