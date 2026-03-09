import os
import pandas as pd
from datetime import datetime
from schemas import EVENT_CATEGORIES, ACTION_TYPES, EDGE_CASE_TAGS

# File paths
inputFilePath = os.path.join("data", "processed", "eval_dataset.csv")
devOutputFilePath = os.path.join("data", "processed", "dev_dataset.csv")
testOutputFilePath = os.path.join("data", "processed", "test_dataset.csv")

# Required dataset columns
requiredColumns = [
    "message_id",
    "source_type",
    "split",
    "sent_at",
    "subject",
    "body",
    "gold_calendar_event_required",
    "gold_event_category",
    "gold_event_date",
    "gold_start_time",
    "gold_end_time",
    "gold_action_required",
    "gold_action_type",
    "gold_action_deadline",
    "gold_summary",
    "edge_case_tag"
]


def isValidDate(dateValue):
    """
    Check if a value matches YYYY-MM-DD format.
    Blank values are allowed.
    """
    if pd.isna(dateValue) or str(dateValue).strip() == "":
        return True

    try:
        datetime.strptime(str(dateValue), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def isValidTime(timeValue):
    """
    Check if a value matches HH:MM 24-hour format.
    Blank values are allowed.
    """
    if pd.isna(timeValue) or str(timeValue).strip() == "":
        return True

    try:
        datetime.strptime(str(timeValue), "%H:%M")
        return True
    except ValueError:
        return False


def isValidBoolean(booleanValue):
    """
    Check if a value is true or false.
    """
    if pd.isna(booleanValue):
        return False

    valueAsText = str(booleanValue).strip().lower()
    return valueAsText == "true" or valueAsText == "false"


def validateColumns(dataFrame):
    """
    Ensure all required columns exist in the dataset.
    """
    missingColumns = []

    for columnName in requiredColumns:
        if columnName not in dataFrame.columns:
            missingColumns.append(columnName)

    if len(missingColumns) > 0:
        print("Missing columns found:")
        for columnName in missingColumns:
            print("-", columnName)
        return False

    return True


def validateRows(dataFrame):
    """
    Validate each row in the dataset and print all errors found.
    """
    foundErrors = False

    for rowIndex in range(len(dataFrame)):
        rowNumber = rowIndex + 2  # +2 because CSV header is row 1
        currentRow = dataFrame.iloc[rowIndex]

        # Validate boolean fields
        if not isValidBoolean(currentRow["gold_calendar_event_required"]):
            print("Row", rowNumber, ": invalid gold_calendar_event_required value")
            foundErrors = True

        if not isValidBoolean(currentRow["gold_action_required"]):
            print("Row", rowNumber, ": invalid gold_action_required value")
            foundErrors = True

        # Validate event category
        eventCategory = str(currentRow["gold_event_category"]).strip()
        if eventCategory not in EVENT_CATEGORIES:
            print("Row", rowNumber, ": invalid gold_event_category value ->", eventCategory)
            foundErrors = True

        # Validate action type
        actionType = str(currentRow["gold_action_type"]).strip()
        if actionType not in ACTION_TYPES:
            print("Row", rowNumber, ": invalid gold_action_type value ->", actionType)
            foundErrors = True

        # Validate edge case tag
        edgeCaseTag = str(currentRow["edge_case_tag"]).strip()
        if edgeCaseTag not in EDGE_CASE_TAGS:
            print("Row", rowNumber, ": invalid edge_case_tag value ->", edgeCaseTag)
            foundErrors = True

        # Validate date fields
        if not isValidDate(currentRow["gold_event_date"]):
            print("Row", rowNumber, ": invalid gold_event_date format")
            foundErrors = True

        if not isValidDate(currentRow["gold_action_deadline"]):
            print("Row", rowNumber, ": invalid gold_action_deadline format")
            foundErrors = True

        # Validate time fields
        if not isValidTime(currentRow["gold_start_time"]):
            print("Row", rowNumber, ": invalid gold_start_time format")
            foundErrors = True

        if not isValidTime(currentRow["gold_end_time"]):
            print("Row", rowNumber, ": invalid gold_end_time format")
            foundErrors = True

        # Validate event logic
        eventRequired = str(currentRow["gold_calendar_event_required"]).strip().lower()
        if eventRequired == "false":
            if str(currentRow["gold_event_category"]).strip() != "none":
                print("Row", rowNumber, ": event is false but event category is not none")
                foundErrors = True

        # Validate action logic
        actionRequired = str(currentRow["gold_action_required"]).strip().lower()
        if actionRequired == "false":
            if str(currentRow["gold_action_type"]).strip() != "none":
                print("Row", rowNumber, ": action is false but action type is not none")
                foundErrors = True

    return not foundErrors


def splitDataset(dataFrame):
    """
    Split the full dataset into dev and test datasets using the split column.
    """
    devDataFrame = dataFrame[dataFrame["split"] == "dev"].copy()
    testDataFrame = dataFrame[dataFrame["split"] == "test"].copy()

    devDataFrame.to_csv(devOutputFilePath, index=False)
    testDataFrame.to_csv(testOutputFilePath, index=False)

    print("Dev dataset saved to:", devOutputFilePath)
    print("Test dataset saved to:", testOutputFilePath)
    print("Dev rows:", len(devDataFrame))
    print("Test rows:", len(testDataFrame))


def main():
    """
    Main function for loading, validating, and splitting the dataset.
    """
    if not os.path.exists(inputFilePath):
        print("Input dataset file not found:", inputFilePath)
        return

    dataFrame = pd.read_csv(inputFilePath)

    print("Dataset loaded successfully.")
    print("Total rows:", len(dataFrame))
    print("Total columns:", len(dataFrame.columns))

    columnsAreValid = validateColumns(dataFrame)
    if not columnsAreValid:
        print("Column validation failed.")
        return

    rowsAreValid = validateRows(dataFrame)
    if not rowsAreValid:
        print("Row validation failed. Fix the errors above before continuing.")
        return

    print("Dataset validation passed.")
    splitDataset(dataFrame)


if __name__ == "__main__":
    main()