import os
import pandas as pd

# Input files.
goldFilePath = os.path.join("data", "processed", "eval_dataset.csv")
predictionFilePath = os.path.join("outputs", "baseline_predictions.csv")

# Output files.
failureCasesFilePath = os.path.join("outputs", "failure_cases.csv")
failureSummaryFilePath = os.path.join("outputs", "failure_summary.csv")


def loadCsvFile(filePath):
    """
    Load a CSV file if it exists.

    Returns:
        pandas.DataFrame | None:
            The loaded dataframe, or None if the file cannot be found.
    """
    if not os.path.exists(filePath):
        print("File not found:", filePath)
        return None

    return pd.read_csv(filePath)


def normaliseBoolean(value):
    """
    Convert boolean-like values into a consistent lowercase string.

    This keeps comparisons simple across gold labels and predictions.
    """
    if pd.isna(value):
        return "false"

    valueAsText = str(value).strip().lower()

    if valueAsText in ["true", "1", "yes"]:
        return "true"

    return "false"


def normaliseText(value):
    """
    Convert missing values to an empty string and strip whitespace.

    This avoids false mismatches caused by NaN values or extra spaces.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def prepareMergedData(goldDataFrame, predictionDataFrame):
    """
    Merge the gold labels and predictions using message_id.

    Only rows present in both files will be included in the analysis.
    """
    mergedDataFrame = goldDataFrame.merge(
        predictionDataFrame,
        on="message_id",
        how="inner"
    )

    return mergedDataFrame


def getFieldMappings():
    """
    Define which gold fields should be compared against which prediction fields.

    Each dictionary entry keeps the comparison logic in one place so it is
    easy to extend later when more providers or fields are added.
    """
    return [
        {
            "field_name": "calendar_event_required",
            "gold_column": "gold_calendar_event_required",
            "pred_column": "pred_calendar_event_required",
            "field_type": "boolean"
        },
        {
            "field_name": "event_category",
            "gold_column": "gold_event_category",
            "pred_column": "pred_event_category",
            "field_type": "text"
        },
        {
            "field_name": "event_date",
            "gold_column": "gold_event_date",
            "pred_column": "pred_event_date",
            "field_type": "text"
        },
        {
            "field_name": "start_time",
            "gold_column": "gold_start_time",
            "pred_column": "pred_start_time",
            "field_type": "text"
        },
        {
            "field_name": "end_time",
            "gold_column": "gold_end_time",
            "pred_column": "pred_end_time",
            "field_type": "text"
        },
        {
            "field_name": "action_required",
            "gold_column": "gold_action_required",
            "pred_column": "pred_action_required",
            "field_type": "boolean"
        },
        {
            "field_name": "action_type",
            "gold_column": "gold_action_type",
            "pred_column": "pred_action_type",
            "field_type": "text"
        },
        {
            "field_name": "action_deadline",
            "gold_column": "gold_action_deadline",
            "pred_column": "pred_action_deadline",
            "field_type": "text"
        },
        {
            "field_name": "summary",
            "gold_column": "gold_summary",
            "pred_column": "pred_summary",
            "field_type": "text"
        }
    ]


def getNormalisedValue(value, fieldType):
    """
    Apply the correct normalisation rule based on the field type.
    """
    if fieldType == "boolean":
        return normaliseBoolean(value)

    return normaliseText(value)


def buildFailureRows(mergedDataFrame):
    """
    Compare each prediction field against the gold label and record mismatches.

    Each failure row includes the message metadata so it is easier to inspect
    patterns later, such as whether certain edge cases fail more often.
    """
    failureRows = []
    fieldMappings = getFieldMappings()

    for rowIndex in range(len(mergedDataFrame)):
        currentRow = mergedDataFrame.iloc[rowIndex]

        for fieldMapping in fieldMappings:
            fieldName = fieldMapping["field_name"]
            goldColumn = fieldMapping["gold_column"]
            predColumn = fieldMapping["pred_column"]
            fieldType = fieldMapping["field_type"]

            goldValue = getNormalisedValue(currentRow[goldColumn], fieldType)
            predictedValue = getNormalisedValue(currentRow[predColumn], fieldType)

            if goldValue != predictedValue:
                failureRows.append({
                    "message_id": currentRow["message_id"],
                    "provider": currentRow.get("provider", "unknown"),
                    "source_type": currentRow["source_type"],
                    "split": currentRow["split"],
                    "edge_case_tag": currentRow["edge_case_tag"],
                    "field_name": fieldName,
                    "gold_value": goldValue,
                    "predicted_value": predictedValue,
                    "subject": normaliseText(currentRow["subject"]),
                    "body": normaliseText(currentRow["body"])
                })

    return failureRows


def buildFailureSummary(failureDataFrame):
    """
    Build a simple summary showing how many failures occurred per field.
    """
    if len(failureDataFrame) == 0:
        return pd.DataFrame(columns=["field_name", "failure_count"])

    summaryDataFrame = (
        failureDataFrame.groupby("field_name")
        .size()
        .reset_index(name="failure_count")
        .sort_values(by="failure_count", ascending=False)
    )

    return summaryDataFrame


def main():
    """
    Load gold labels and predictions, identify field-level failures,
    and save both detailed and summary outputs.
    """
    goldDataFrame = loadCsvFile(goldFilePath)
    predictionDataFrame = loadCsvFile(predictionFilePath)

    if goldDataFrame is None or predictionDataFrame is None:
        print("Failure analysis could not run because one or more files are missing.")
        return

    mergedDataFrame = prepareMergedData(goldDataFrame, predictionDataFrame)

    if len(mergedDataFrame) == 0:
        print("No matching rows found between gold labels and predictions.")
        return

    failureRows = buildFailureRows(mergedDataFrame)
    failureDataFrame = pd.DataFrame(failureRows)
    failureSummaryDataFrame = buildFailureSummary(failureDataFrame)

    os.makedirs("outputs", exist_ok=True)

    failureDataFrame.to_csv(failureCasesFilePath, index=False)
    failureSummaryDataFrame.to_csv(failureSummaryFilePath, index=False)

    print("Failure analysis complete.")
    print("Rows evaluated:", len(mergedDataFrame))
    print("Total field-level failures:", len(failureDataFrame))
    print("Failure cases saved to:", failureCasesFilePath)
    print("Failure summary saved to:", failureSummaryFilePath)


if __name__ == "__main__":
    main()