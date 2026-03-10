import os
import re
import hashlib
import pandas as pd

# Input and output files for the real-world cleaning stage.
rawInputFilePath = os.path.join("data", "intermediate", "enron_messages_raw.csv")
cleanOutputFilePath = os.path.join("data", "intermediate", "enron_messages_clean.csv")
cleaningReportFilePath = os.path.join("outputs", "enron_cleaning_report.csv")


def normaliseWhitespace(textValue):
    """
    Collapse repeated whitespace into single spaces and trim the result.

    This makes text easier to read and reduces small formatting differences
    before duplicate detection and later review.
    """
    if pd.isna(textValue):
        return ""

    textValue = str(textValue)
    textValue = re.sub(r"\s+", " ", textValue)
    return textValue.strip()


def removeQuotedAndForwardedLines(bodyText):
    """
    Remove common quoted or forwarded header lines from the message body.

    This keeps the cleaned dataset focused on the main message content rather
    than long email chains and repeated metadata.
    """
    if bodyText == "":
        return ""

    cleanedLines = []
    lineList = str(bodyText).splitlines()

    for lineText in lineList:
        strippedLine = lineText.strip()

        if strippedLine.startswith(">"):
            continue

        if strippedLine.lower().startswith("from:"):
            continue

        if strippedLine.lower().startswith("sent:"):
            continue

        if strippedLine.lower().startswith("to:"):
            continue

        if strippedLine.lower().startswith("subject:"):
            continue

        cleanedLines.append(strippedLine)

    cleanedBody = " ".join(cleanedLines)
    cleanedBody = re.sub(r"\s+", " ", cleanedBody)

    return cleanedBody.strip()


def buildContentHash(subjectText, bodyText):
    """
    Build a stable hash from the cleaned subject and body.

    This is used to remove duplicate messages without relying on file names.
    """
    combinedText = (subjectText + "||" + bodyText).lower().strip()
    return hashlib.md5(combinedText.encode("utf-8")).hexdigest()


def isUsableCleanRow(subjectText, bodyText):
    """
    Apply final usability checks after cleaning.

    These rules are intentionally practical rather than overly strict.
    The goal is to keep enough real-world variety while removing rows that
    would not be useful for manual labelling later.
    """
    bodyLength = len(bodyText)

    if bodyLength < 60:
        return False

    if bodyLength > 3000:
        return False

    if subjectText == "" and bodyText == "":
        return False

    return True


def main():
    """
    Load raw extracted Enron messages, clean them, remove duplicates,
    and save a cleaned real-world dataset plus a simple cleaning report.
    """
    if not os.path.exists(rawInputFilePath):
        print("Raw input file not found:", rawInputFilePath)
        return

    rawDataFrame = pd.read_csv(rawInputFilePath)

    originalRowCount = len(rawDataFrame)

    # Standardise the main text fields used later for review and labelling.
    rawDataFrame["subject"] = rawDataFrame["subject_raw"].apply(normaliseWhitespace)
    rawDataFrame["body"] = rawDataFrame["body_raw"].apply(removeQuotedAndForwardedLines)
    rawDataFrame["body"] = rawDataFrame["body"].apply(normaliseWhitespace)

    # Keep a simplified sent_at field for downstream use.
    rawDataFrame["sent_at"] = rawDataFrame["sent_at_parsed"].fillna("").astype(str).str.strip()

    # Build a duplicate key from the cleaned subject and body.
    rawDataFrame["content_hash"] = rawDataFrame.apply(
        lambda currentRow: buildContentHash(currentRow["subject"], currentRow["body"]),
        axis=1
    )

    beforeDuplicateRemovalCount = len(rawDataFrame)
    cleanDataFrame = rawDataFrame.drop_duplicates(subset=["content_hash"]).copy()
    duplicateRowsRemoved = beforeDuplicateRemovalCount - len(cleanDataFrame)

    # Mark which rows are worth keeping for later evaluation review.
    cleanDataFrame["is_usable_for_eval"] = cleanDataFrame.apply(
        lambda currentRow: isUsableCleanRow(currentRow["subject"], currentRow["body"]),
        axis=1
    )

    usableDataFrame = cleanDataFrame[cleanDataFrame["is_usable_for_eval"] == True].copy()

    # Keep only the columns that matter for the next stage.
    usableDataFrame = usableDataFrame[
        [
            "message_id",
            "source_file_path",
            "from_address",
            "to_address",
            "sent_at",
            "subject",
            "body",
            "relevance_score",
            "is_usable_for_eval"
        ]
    ].copy()

    os.makedirs("outputs", exist_ok=True)
    usableDataFrame.to_csv(cleanOutputFilePath, index=False)

    reportRows = [
        {
            "metric_name": "raw_rows_loaded",
            "metric_value": originalRowCount
        },
        {
            "metric_name": "duplicate_rows_removed",
            "metric_value": duplicateRowsRemoved
        },
        {
            "metric_name": "clean_rows_saved",
            "metric_value": len(usableDataFrame)
        }
    ]

    reportDataFrame = pd.DataFrame(reportRows)
    reportDataFrame.to_csv(cleaningReportFilePath, index=False)

    print("Real-world cleaning complete.")
    print("Raw rows loaded:", originalRowCount)
    print("Duplicate rows removed:", duplicateRowsRemoved)
    print("Clean rows saved:", len(usableDataFrame))
    print("Clean output file:", cleanOutputFilePath)
    print("Cleaning report file:", cleaningReportFilePath)


if __name__ == "__main__":
    main()