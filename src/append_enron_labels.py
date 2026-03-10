import os
import pandas as pd

# Existing benchmark file and newly labelled Enron rows.
existingEvalFilePath = os.path.join("data", "processed", "eval_dataset.csv")
enronLabelFilePath = os.path.join("data", "processed", "enron_label_template_labeled.csv")
outputFilePath = os.path.join("data", "processed", "eval_dataset.csv")


def loadCsvFile(filePath):
    """
    Load a CSV file if it exists.
    """
    if not os.path.exists(filePath):
        print("File not found:", filePath)
        return None

    return pd.read_csv(filePath)


def main():
    """
    Append the labelled Enron rows to the existing evaluation dataset.

    A duplicate check is applied on message_id so the benchmark stays clean.
    """
    existingEvalDataFrame = loadCsvFile(existingEvalFilePath)
    enronLabelDataFrame = loadCsvFile(enronLabelFilePath)

    if existingEvalDataFrame is None or enronLabelDataFrame is None:
        print("Append process could not continue because one or more files are missing.")
        return

    originalExistingCount = len(existingEvalDataFrame)
    originalEnronCount = len(enronLabelDataFrame)

    combinedDataFrame = pd.concat(
        [existingEvalDataFrame, enronLabelDataFrame],
        ignore_index=True
    )

    beforeDuplicateRemovalCount = len(combinedDataFrame)
    combinedDataFrame = combinedDataFrame.drop_duplicates(subset=["message_id"]).copy()
    duplicatesRemoved = beforeDuplicateRemovalCount - len(combinedDataFrame)

    combinedDataFrame.to_csv(outputFilePath, index=False)

    print("Enron rows appended successfully.")
    print("Existing rows before append:", originalExistingCount)
    print("Labelled Enron rows loaded:", originalEnronCount)
    print("Duplicates removed:", duplicatesRemoved)
    print("Final benchmark size:", len(combinedDataFrame))
    print("Updated file:", outputFilePath)


if __name__ == "__main__":
    main()