import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from schemas import EVENT_CATEGORIES, ACTION_TYPES

# Gold dataset and baseline prediction file paths.
goldFilePath = os.path.join("data", "processed", "eval_dataset.csv")
predictionFilePath = os.path.join("outputs", "baseline_predictions.csv")

# Output files for evaluation results.
summaryMetricsFilePath = os.path.join("outputs", "summary_metrics.csv")
fieldMetricsFilePath = os.path.join("outputs", "field_metrics.csv")


def loadCsvFile(filePath):
    """
    Load a CSV file if it exists.

    Returns:
        pandas.DataFrame | None:
            The loaded dataframe, or None if the file is missing.
    """
    if not os.path.exists(filePath):
        print("File not found:", filePath)
        return None

    return pd.read_csv(filePath)


def normaliseBoolean(value):
    """
    Convert boolean-like values into a consistent lowercase string.

    This keeps comparisons simple during evaluation.
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
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def prepareMergedData(goldDataFrame, predictionDataFrame):
    """
    Merge the gold dataset and prediction dataset on message_id.

    Only rows present in both files will be kept for evaluation.
    """
    mergedDataFrame = goldDataFrame.merge(
        predictionDataFrame,
        on="message_id",
        how="inner"
    )

    return mergedDataFrame


def calculateBinaryMetrics(goldValues, predictedValues, metricName):
    """
    Calculate precision, recall, and F1 for a binary field.

    Returns:
        dict:
            A dictionary containing the metric results.
    """
    precision, recall, f1Score, _ = precision_recall_fscore_support(
        goldValues,
        predictedValues,
        average="binary",
        pos_label="true",
        zero_division=0
    )

    return {
        "metric_name": metricName,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1Score, 4)
    }


def calculateMultiClassMetrics(goldValues, predictedValues, metricName):
    """
    Calculate macro precision, recall, and F1 for a multi-class field.
    """
    precision, recall, f1Score, _ = precision_recall_fscore_support(
        goldValues,
        predictedValues,
        average="macro",
        zero_division=0
    )

    return {
        "metric_name": metricName,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1Score, 4)
    }


def calculateAccuracyMetric(goldValues, predictedValues, metricName):
    """
    Calculate simple accuracy for exact-match fields such as dates and times.
    """
    accuracyValue = accuracy_score(goldValues, predictedValues)

    return {
        "metric_name": metricName,
        "accuracy": round(accuracyValue, 4)
    }


def buildFieldMetrics(mergedDataFrame):
    """
    Build a list of field-level evaluation metrics.

    This includes:
    - binary metrics for event/action required
    - macro metrics for category/type classification
    - exact-match accuracy for date and time fields
    """
    fieldMetrics = []

    # Normalise boolean fields before scoring them.
    goldEventRequired = mergedDataFrame["gold_calendar_event_required"].apply(normaliseBoolean)
    predEventRequired = mergedDataFrame["pred_calendar_event_required"].apply(normaliseBoolean)

    goldActionRequired = mergedDataFrame["gold_action_required"].apply(normaliseBoolean)
    predActionRequired = mergedDataFrame["pred_action_required"].apply(normaliseBoolean)

    fieldMetrics.append(
        calculateBinaryMetrics(
            goldEventRequired,
            predEventRequired,
            "calendar_event_required"
        )
    )

    fieldMetrics.append(
        calculateBinaryMetrics(
            goldActionRequired,
            predActionRequired,
            "action_required"
        )
    )

    # Use text normalisation for categorical fields.
    goldEventCategory = mergedDataFrame["gold_event_category"].apply(normaliseText)
    predEventCategory = mergedDataFrame["pred_event_category"].apply(normaliseText)

    goldActionType = mergedDataFrame["gold_action_type"].apply(normaliseText)
    predActionType = mergedDataFrame["pred_action_type"].apply(normaliseText)

    fieldMetrics.append(
        calculateMultiClassMetrics(
            goldEventCategory,
            predEventCategory,
            "event_category"
        )
    )

    fieldMetrics.append(
        calculateMultiClassMetrics(
            goldActionType,
            predActionType,
            "action_type"
        )
    )

    # Exact-match accuracy for extracted date and time fields.
    goldEventDate = mergedDataFrame["gold_event_date"].apply(normaliseText)
    predEventDate = mergedDataFrame["pred_event_date"].apply(normaliseText)

    goldStartTime = mergedDataFrame["gold_start_time"].apply(normaliseText)
    predStartTime = mergedDataFrame["pred_start_time"].apply(normaliseText)

    goldEndTime = mergedDataFrame["gold_end_time"].apply(normaliseText)
    predEndTime = mergedDataFrame["pred_end_time"].apply(normaliseText)

    goldActionDeadline = mergedDataFrame["gold_action_deadline"].apply(normaliseText)
    predActionDeadline = mergedDataFrame["pred_action_deadline"].apply(normaliseText)

    fieldMetrics.append(
        calculateAccuracyMetric(
            goldEventDate,
            predEventDate,
            "event_date_accuracy"
        )
    )

    fieldMetrics.append(
        calculateAccuracyMetric(
            goldStartTime,
            predStartTime,
            "start_time_accuracy"
        )
    )

    fieldMetrics.append(
        calculateAccuracyMetric(
            goldEndTime,
            predEndTime,
            "end_time_accuracy"
        )
    )

    fieldMetrics.append(
        calculateAccuracyMetric(
            goldActionDeadline,
            predActionDeadline,
            "action_deadline_accuracy"
        )
    )

    return fieldMetrics


def buildSummaryMetrics(mergedDataFrame, fieldMetrics):
    """
    Build a small summary table for the provider.

    This gives one row with the provider name, row count, average latency,
    and the most important headline metrics.
    """
    providerName = "unknown"

    if "provider" in mergedDataFrame.columns and len(mergedDataFrame) > 0:
        providerName = str(mergedDataFrame.iloc[0]["provider"]).strip()

    averageLatency = 0.0
    if "latency_ms" in mergedDataFrame.columns:
        averageLatency = round(pd.to_numeric(mergedDataFrame["latency_ms"], errors="coerce").fillna(0).mean(), 2)

    summaryRow = {
        "provider": providerName,
        "rows_evaluated": len(mergedDataFrame),
        "average_latency_ms": averageLatency,
        "calendar_event_required_f1": "",
        "action_required_f1": "",
        "event_category_macro_f1": "",
        "action_type_macro_f1": "",
        "event_date_accuracy": "",
        "action_deadline_accuracy": ""
    }

    for metricRow in fieldMetrics:
        metricName = metricRow["metric_name"]

        if metricName == "calendar_event_required":
            summaryRow["calendar_event_required_f1"] = metricRow["f1_score"]

        elif metricName == "action_required":
            summaryRow["action_required_f1"] = metricRow["f1_score"]

        elif metricName == "event_category":
            summaryRow["event_category_macro_f1"] = metricRow["f1_score"]

        elif metricName == "action_type":
            summaryRow["action_type_macro_f1"] = metricRow["f1_score"]

        elif metricName == "event_date_accuracy":
            summaryRow["event_date_accuracy"] = metricRow["accuracy"]

        elif metricName == "action_deadline_accuracy":
            summaryRow["action_deadline_accuracy"] = metricRow["accuracy"]

    return [summaryRow]


def main():
    """
    Load the gold labels and baseline predictions, evaluate them,
    and save the results as CSV files.
    """
    goldDataFrame = loadCsvFile(goldFilePath)
    predictionDataFrame = loadCsvFile(predictionFilePath)

    if goldDataFrame is None or predictionDataFrame is None:
        print("Evaluation could not run because one or more files are missing.")
        return

    mergedDataFrame = prepareMergedData(goldDataFrame, predictionDataFrame)

    if len(mergedDataFrame) == 0:
        print("No matching rows found between gold labels and predictions.")
        return

    fieldMetrics = buildFieldMetrics(mergedDataFrame)
    summaryMetrics = buildSummaryMetrics(mergedDataFrame, fieldMetrics)

    fieldMetricsDataFrame = pd.DataFrame(fieldMetrics)
    summaryMetricsDataFrame = pd.DataFrame(summaryMetrics)

    os.makedirs("outputs", exist_ok=True)
    fieldMetricsDataFrame.to_csv(fieldMetricsFilePath, index=False)
    summaryMetricsDataFrame.to_csv(summaryMetricsFilePath, index=False)

    print("Evaluation complete.")
    print("Rows evaluated:", len(mergedDataFrame))
    print("Field metrics saved to:", fieldMetricsFilePath)
    print("Summary metrics saved to:", summaryMetricsFilePath)


if __name__ == "__main__":
    main()