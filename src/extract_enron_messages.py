import os
import re
import pandas as pd

# Likely location of the extracted Enron maildir dataset.
rawMaildirRootFolder = os.path.join("data", "raw", "enron_maildir", "maildir")

# Output file for the parsed raw messages.
rawOutputFilePath = os.path.join("data", "intermediate", "enron_messages_raw.csv")

# Target number of usable messages to collect.
targetMessageCount = 1000


def ensureOutputFolderExists():
    """
    Create the intermediate output folder if it does not already exist.
    """
    os.makedirs(os.path.join("data", "intermediate"), exist_ok=True)


def normaliseWhitespace(textValue):
    """
    Collapse repeated whitespace into single spaces and trim the result.
    """
    if textValue is None:
        return ""

    textValue = str(textValue)
    textValue = re.sub(r"\s+", " ", textValue)
    return textValue.strip()


def readRawFile(filePath):
    """
    Read a raw Enron email file as text.

    The Enron dataset is messy, so this tries UTF-8 first and falls back
    to latin-1 with replacement if needed.

    A Windows-safe absolute path is used because many Enron files have names
    ending with a trailing dot.
    """
    safeFilePath = getSafeFilePath(filePath)

    try:
        with open(safeFilePath, "r", encoding="utf-8", errors="replace") as rawFile:
            return rawFile.read()
    except Exception:
        with open(safeFilePath, "r", encoding="latin-1", errors="replace") as rawFile:
            return rawFile.read()

def extractHeaderValue(rawText, headerName):
    """
    Extract a simple single-line header value from the raw email text.
    """
    pattern = r"^" + re.escape(headerName) + r":\s*(.*)$"
    matchObject = re.search(pattern, rawText, flags=re.MULTILINE | re.IGNORECASE)

    if matchObject is None:
        return ""

    return normaliseWhitespace(matchObject.group(1))


def extractBody(rawText):
    """
    Extract the body by splitting headers and body on the first blank line.
    """
    splitParts = re.split(r"\r?\n\r?\n", rawText, maxsplit=1)

    if len(splitParts) < 2:
        return ""

    bodyText = splitParts[1]
    return normaliseWhitespace(bodyText)


def calculateRelevanceScore(subjectText, bodyText):
    """
    Assign a simple heuristic score to help identify messages that may be
    useful for event/action extraction later.
    """
    combinedText = (subjectText + " " + bodyText).lower()

    keywordList = [
        "meeting", "call", "conference", "appointment", "schedule", "scheduled",
        "rescheduled", "cancelled", "canceled", "deadline", "due", "submit",
        "confirm", "reply", "attend", "payment", "invoice", "event", "tomorrow",
        "monday", "tuesday", "wednesday", "thursday", "friday"
    ]

    scoreValue = 0

    for keyword in keywordList:
        if keyword in combinedText:
            scoreValue = scoreValue + 1

    if re.search(r"\b\d{1,2}:\d{2}\b", combinedText):
        scoreValue = scoreValue + 2

    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", combinedText):
        scoreValue = scoreValue + 2

    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", combinedText):
        scoreValue = scoreValue + 2

    return scoreValue


def isUsableMessage(subjectText, bodyText):
    """
    Apply light filtering so the export stays practical to review.

    These rules are intentionally loose because we want a large messy sample
    first, not a highly filtered one.
    """
    if subjectText == "" and bodyText == "":
        return False

    if len(bodyText) < 20:
        return False

    if len(bodyText) > 12000:
        return False

    return True


def parseEmailFile(filePath):
    """
    Parse one raw Enron mail file into a simple structured row.
    """
    rawText = readRawFile(filePath)

    subjectText = extractHeaderValue(rawText, "Subject")
    fromAddress = extractHeaderValue(rawText, "From")
    toAddress = extractHeaderValue(rawText, "To")
    dateText = extractHeaderValue(rawText, "Date")
    bodyText = extractBody(rawText)

    relevanceScore = calculateRelevanceScore(subjectText, bodyText)

    return {
        "source_file_path": filePath,
        "from_address": fromAddress,
        "to_address": toAddress,
        "sent_at_raw": dateText,
        "sent_at_parsed": dateText,
        "subject_raw": subjectText,
        "body_raw": bodyText,
        "relevance_score": relevanceScore
    }

def getSafeFilePath(filePath):
    """
    Convert a relative file path into a Windows-safe absolute path
    without losing trailing dots in filenames.

    os.path.abspath() normalises Windows paths and strips trailing dots,
    so this builds the absolute path manually instead.
    """
    if os.path.isabs(filePath):
        absolutePath = filePath
    else:
        currentWorkingFolder = os.getcwd().rstrip("\\/")
        cleanedRelativePath = str(filePath).replace("/", "\\")
        absolutePath = currentWorkingFolder + "\\" + cleanedRelativePath

    if os.name == "nt":
        if not absolutePath.startswith("\\\\?\\"):
            absolutePath = "\\\\?\\" + absolutePath

    return absolutePath


def main():
    """
    Walk through the Enron maildir folders, parse usable messages,
    and save them into a raw CSV file.
    """
    ensureOutputFolderExists()

    if not os.path.exists(rawMaildirRootFolder):
        print("Raw Enron maildir folder not found:", rawMaildirRootFolder)
        return

    print("Using dataset root:", rawMaildirRootFolder)

    parsedRows = []
    fileCounter = 0
    parseErrorCount = 0
    filteredOutCount = 0
    printedErrorCount = 0

    for rootFolder, _, fileNames in os.walk(rawMaildirRootFolder):
        for fileName in fileNames:
            fileCounter = fileCounter + 1
            filePath = os.path.join(rootFolder, fileName)

            try:
                parsedRow = parseEmailFile(filePath)

                if isUsableMessage(parsedRow["subject_raw"], parsedRow["body_raw"]):
                    parsedRow["message_id"] = "enron_raw_" + str(len(parsedRows) + 1).zfill(4)
                    parsedRows.append(parsedRow)
                else:
                    filteredOutCount = filteredOutCount + 1

                if len(parsedRows) >= targetMessageCount:
                    break

            except Exception as error:
                parseErrorCount = parseErrorCount + 1

                # Print the first few real errors so we can see what is happening.
                if printedErrorCount < 5:
                    print("Parse error on file:", filePath)
                    print("Error type:", type(error).__name__)
                    print("Error message:", str(error))
                    print("-" * 60)
                    printedErrorCount = printedErrorCount + 1

                continue

        if len(parsedRows) >= targetMessageCount:
            break

    print("Files scanned:", fileCounter)
    print("Usable messages found:", len(parsedRows))
    print("Messages filtered out:", filteredOutCount)
    print("Parse errors:", parseErrorCount)

    if len(parsedRows) == 0:
        print("No usable messages were extracted.")
        return

    rawDataFrame = pd.DataFrame(parsedRows)
    rawDataFrame.to_csv(rawOutputFilePath, index=False)

    print("Raw Enron extraction complete.")
    print("Output file:", rawOutputFilePath)


if __name__ == "__main__":
    main()