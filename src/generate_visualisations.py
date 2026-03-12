import os
import pandas as pd
import matplotlib.pyplot as plt


# Input files for the final baseline and Qwen results.
baselineSummaryFilePath = os.path.join("outputs", "summary_metrics.csv")
baselineFailureFilePath = os.path.join("outputs", "failure_summary.csv")

qwenSummaryFilePath = os.path.join("outputs", "qwen_summary_metrics.csv")
qwenFailureFilePath = os.path.join("outputs", "qwen_failure_summary.csv")

# Output folder and chart files.
chartOutputFolder = os.path.join("outputs", "charts")
metricChartFilePath = os.path.join(chartOutputFolder, "metric_comparison.png")
latencyChartFilePath = os.path.join(chartOutputFolder, "latency_comparison.png")
failureChartFilePath = os.path.join(chartOutputFolder, "failure_comparison.png")


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


def createMetricComparisonChart(baselineSummaryDataFrame, qwenSummaryDataFrame):
    """
    Create a grouped bar chart comparing the main evaluation metrics
    for the baseline and Qwen systems.
    """
    metricNames = [
        "calendar_event_required_f1",
        "action_required_f1",
        "event_category_macro_f1",
        "action_type_macro_f1",
        "event_date_accuracy",
        "action_deadline_accuracy"
    ]

    displayNames = [
        "Calendar Event F1",
        "Action Required F1",
        "Event Category F1",
        "Action Type F1",
        "Event Date Accuracy",
        "Action Deadline Accuracy"
    ]

    baselineValues = []
    qwenValues = []

    for metricName in metricNames:
        baselineValues.append(float(baselineSummaryDataFrame.iloc[0][metricName]))
        qwenValues.append(float(qwenSummaryDataFrame.iloc[0][metricName]))

    xPositions = list(range(len(metricNames)))
    barWidth = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(
        [position - barWidth / 2 for position in xPositions],
        baselineValues,
        width=barWidth,
        label="Baseline Rules"
    )
    plt.bar(
        [position + barWidth / 2 for position in xPositions],
        qwenValues,
        width=barWidth,
        label="Qwen 3 8B"
    )

    plt.xticks(xPositions, displayNames, rotation=20, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Baseline vs Qwen: Main Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metricChartFilePath, dpi=300)
    plt.close()


def createLatencyComparisonChart(baselineSummaryDataFrame, qwenSummaryDataFrame):
    """
    Create a bar chart comparing average latency between the two systems.
    """
    providerNames = ["Baseline Rules", "Qwen 3 8B"]
    latencyValues = [
        float(baselineSummaryDataFrame.iloc[0]["average_latency_ms"]),
        float(qwenSummaryDataFrame.iloc[0]["average_latency_ms"])
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(providerNames, latencyValues)
    plt.ylabel("Average Latency (ms)")
    plt.title("Baseline vs Qwen: Average Latency")
    plt.tight_layout()
    plt.savefig(latencyChartFilePath, dpi=300)
    plt.close()


def createFailureComparisonChart(baselineFailureDataFrame, qwenFailureDataFrame):
    """
    Create a grouped bar chart comparing field-level failure counts
    for the baseline and Qwen systems.
    """
    allFieldNames = [
        "calendar_event_required",
        "action_required",
        "event_category",
        "action_type",
        "event_date",
        "action_deadline",
        "start_time",
        "end_time"
    ]

    baselineFailureMap = {}
    qwenFailureMap = {}

    for rowIndex in range(len(baselineFailureDataFrame)):
        currentRow = baselineFailureDataFrame.iloc[rowIndex]
        baselineFailureMap[str(currentRow["field_name"])] = int(currentRow["failure_count"])

    for rowIndex in range(len(qwenFailureDataFrame)):
        currentRow = qwenFailureDataFrame.iloc[rowIndex]
        qwenFailureMap[str(currentRow["field_name"])] = int(currentRow["failure_count"])

    baselineValues = []
    qwenValues = []

    for fieldName in allFieldNames:
        baselineValues.append(baselineFailureMap.get(fieldName, 0))
        qwenValues.append(qwenFailureMap.get(fieldName, 0))

    xPositions = list(range(len(allFieldNames)))
    barWidth = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(
        [position - barWidth / 2 for position in xPositions],
        baselineValues,
        width=barWidth,
        label="Baseline Rules"
    )
    plt.bar(
        [position + barWidth / 2 for position in xPositions],
        qwenValues,
        width=barWidth,
        label="Qwen 3 8B"
    )

    plt.xticks(xPositions, allFieldNames, rotation=25, ha="right")
    plt.ylabel("Failure Count")
    plt.title("Baseline vs Qwen: Failure Count by Field")
    plt.legend()
    plt.tight_layout()
    plt.savefig(failureChartFilePath, dpi=300)
    plt.close()


def main():
    """
    Load the final evaluation outputs and generate three visualisations:
    1. Main metric comparison
    2. Latency comparison
    3. Failure comparison
    """
    baselineSummaryDataFrame = loadCsvFile(baselineSummaryFilePath)
    baselineFailureDataFrame = loadCsvFile(baselineFailureFilePath)

    qwenSummaryDataFrame = loadCsvFile(qwenSummaryFilePath)
    qwenFailureDataFrame = loadCsvFile(qwenFailureFilePath)

    if (
        baselineSummaryDataFrame is None or
        baselineFailureDataFrame is None or
        qwenSummaryDataFrame is None or
        qwenFailureDataFrame is None
    ):
        print("Visualisation generation could not continue because one or more files are missing.")
        return

    os.makedirs(chartOutputFolder, exist_ok=True)

    createMetricComparisonChart(baselineSummaryDataFrame, qwenSummaryDataFrame)
    createLatencyComparisonChart(baselineSummaryDataFrame, qwenSummaryDataFrame)
    createFailureComparisonChart(baselineFailureDataFrame, qwenFailureDataFrame)

    print("Visualisations generated successfully.")
    print("Metric comparison chart:", metricChartFilePath)
    print("Latency comparison chart:", latencyChartFilePath)
    print("Failure comparison chart:", failureChartFilePath)


if __name__ == "__main__":
    main()