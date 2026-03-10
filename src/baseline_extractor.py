import os
import re
import time
import pandas as pd
from datetime import datetime
from email.utils import parsedate_to_datetime
from dateparser.search import search_dates
from schemas import EVENT_CATEGORIES, ACTION_TYPES


# Input dataset and output prediction file paths.
inputFilePath = os.path.join("data", "processed", "eval_dataset.csv")
outputFilePath = os.path.join("outputs", "baseline_predictions.csv")


def loadDataset():
    """
    Load the labelled evaluation dataset from CSV.

    Returns:
        pandas.DataFrame | None:
            The dataset if the file exists, otherwise None.
    """
    if not os.path.exists(inputFilePath):
        print("Dataset file not found:", inputFilePath)
        return None

    dataFrame = pd.read_csv(inputFilePath)
    return dataFrame


def cleanText(textValue):
    """
    Normalise a text value for simple rule-based matching.

    Missing values are converted to an empty string so the rest of the
    extraction logic can run safely without extra checks.
    """
    if pd.isna(textValue):
        return ""

    return str(textValue).strip().lower()


def getCombinedText(subjectText, bodyText):
    """
    Combine the subject and body into one lowercase string.

    This keeps the rule-based matching simple because all keyword checks
    can be done against one text block.
    """
    subjectValue = cleanText(subjectText)
    bodyValue = cleanText(bodyText)

    return subjectValue + " " + bodyValue


def getSentAtDateTime(sentAtValue):
    """
    Convert the sent_at value into a datetime object.

    The project uses two timestamp styles:
    - benchmark rows in YYYY-MM-DD HH:MM:SS format
    - Enron rows in email header date format

    This function supports both so the extractor can run on the mixed dataset.
    """
    sentAtText = str(sentAtValue).strip()

    if sentAtText == "" or sentAtText.lower() == "nan":
        # Fall back to a fixed date if no usable timestamp is available.
        return datetime(2000, 1, 1, 0, 0, 0)

    # First try the standard benchmark format used by the synthetic rows.
    try:
        return datetime.strptime(sentAtText, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    # Then try the email-style date format used by the Enron rows.
    try:
        parsedDateTime = parsedate_to_datetime(sentAtText)

        # Remove timezone information so it works cleanly with the rest
        # of the pipeline and the dateparser settings.
        if parsedDateTime.tzinfo is not None:
            parsedDateTime = parsedDateTime.replace(tzinfo=None)

        return parsedDateTime
    except Exception:
        pass

    # Final fallback so one bad timestamp does not stop the whole run.
    return datetime(2000, 1, 1, 0, 0, 0)


def detectEventCategory(fullText):
    """
    Detect the main event category using keyword rules.

    The order of checks matters. More specific categories are checked first
    so they are not overwritten by broader keywords later on.
    """
    if "cancelled" in fullText or "canceled" in fullText or "postponed" in fullText:
        return "cancellation_change"

    if "payment due" in fullText or "fee due" in fullText or "deposit due" in fullText:
        return "payment_deadline"

    if "trip" in fullText or "visit" in fullText or "museum" in fullText:
        return "trip"

    if "parents evening" in fullText or "meeting" in fullText or "appointment" in fullText:
        return "meeting_admin"

    if "club" in fullText or "training" in fullText or "practice" in fullText:
        return "club_activity"

    if "reminder" in fullText or "please ensure" in fullText or "please remember" in fullText:
        return "reminder_other"

    return "none"


def detectActionType(fullText):
    """
    Detect whether the message contains an action and classify that action.

    Returns:
        tuple[bool, str]:
            A boolean showing whether an action is required and the
            corresponding action type label.
    """
    if "pay" in fullText or "payment" in fullText or "fee" in fullText or "deposit" in fullText:
        return True, "pay"

    if "reply" in fullText or "confirm" in fullText or "let us know" in fullText or "rsvp" in fullText:
        return True, "reply_confirm"

    if "submit" in fullText or "complete form" in fullText or "return slip" in fullText:
        return True, "submit_form"

    if "bring" in fullText or "wear" in fullText or "pack" in fullText:
        return True, "bring_item"

    if "attend" in fullText:
        return True, "attend"

    return False, "none"


def extractTimes(fullText):
    """
    Extract up to two time values from the message text.

    This baseline version only looks for explicit HH:MM patterns.
    If two times are found, they are treated as start and end time.
    """
    timeMatches = re.findall(r"\b\d{1,2}:\d{2}\b", fullText)

    startTime = ""
    endTime = ""

    if len(timeMatches) >= 1:
        startTime = standardiseTime(timeMatches[0])

    if len(timeMatches) >= 2:
        endTime = standardiseTime(timeMatches[1])

    return startTime, endTime


def standardiseTime(timeText):
    """
    Convert a recognised time string into HH:MM format.

    Returns an empty string if the value cannot be parsed cleanly.
    """
    try:
        parsedTime = datetime.strptime(timeText, "%H:%M")
        return parsedTime.strftime("%H:%M")
    except ValueError:
        return ""


def findAllDates(fullText, sentAtDateTime):
    """
    Find possible date mentions in the message using dateparser.

    Relative expressions are resolved against the sent_at value so that
    phrases like 'tomorrow' are converted into a real date.
    """
    foundDates = search_dates(
        fullText,
        settings={
            "RELATIVE_BASE": sentAtDateTime,
            "PREFER_DATES_FROM": "future",
            "DATE_ORDER": "DMY"
        }
    )

    if foundDates is None:
        return []

    uniqueDateList = []

    for foundItem in foundDates:
        foundText = foundItem[0]
        foundDateTime = foundItem[1]
        formattedDate = foundDateTime.strftime("%Y-%m-%d")

        # Avoid keeping duplicate text-date pairs.
        alreadyExists = False
        for existingItem in uniqueDateList:
            if existingItem["text"] == foundText and existingItem["date"] == formattedDate:
                alreadyExists = True
                break

        if not alreadyExists:
            uniqueDateList.append({
                "text": foundText,
                "date": formattedDate
            })

    return uniqueDateList


def getDeadlineKeywords():
    """
    Return words that often indicate a deadline rather than the main event date.
    """
    return [
        "due",
        "by",
        "reply",
        "confirm",
        "deadline",
        "submit",
        "return"
    ]


def chooseActionDeadline(fullText, foundDates, actionRequired, actionType, eventDate):
    """
    Choose the most likely action deadline from the extracted dates.

    The baseline looks for a date that appears near deadline-related wording.
    If the action is something like bringing an item or attending, the event
    date itself can act as the deadline.
    """
    if not actionRequired:
        return ""

    deadlineKeywords = getDeadlineKeywords()

    for dateItem in foundDates:
        loweredDateText = str(dateItem["text"]).lower()

        for keyword in deadlineKeywords:
            if keyword in fullText and loweredDateText in fullText:
                textPosition = fullText.find(loweredDateText)
                searchStart = max(0, textPosition - 20)
                nearbyText = fullText[searchStart:textPosition + len(loweredDateText)]

                if keyword in nearbyText:
                    return dateItem["date"]

    if actionType == "bring_item" or actionType == "attend":
        if eventDate != "":
            return eventDate

    return ""


def chooseEventDate(foundDates, actionDeadline):
    """
    Choose the most likely main event date.

    If multiple dates are found, prefer a date that is different from the
    action deadline so the event date and deadline are not confused.
    """
    if len(foundDates) == 0:
        return ""

    if len(foundDates) == 1:
        return foundDates[0]["date"]

    for dateItem in foundDates:
        if dateItem["date"] != actionDeadline:
            return dateItem["date"]

    return foundDates[0]["date"]


def buildSummary(eventCategory, eventDate, startTime, endTime, actionRequired, actionType, actionDeadline):
    """
    Build a short text summary from the extracted structured fields.
    """
    summaryParts = []

    if eventCategory != "none" and eventDate != "":
        categoryText = eventCategory.replace("_", " ")
        summaryParts.append(categoryText + " on " + eventDate)

    if startTime != "" and endTime != "":
        summaryParts.append("from " + startTime + " to " + endTime)

    if actionRequired and actionType != "none":
        actionText = actionType.replace("_", " ")
        if actionDeadline != "":
            summaryParts.append(actionText + " by " + actionDeadline)
        else:
            summaryParts.append(actionText + " required")

    if len(summaryParts) == 0:
        return "No event or action required."

    return "; ".join(summaryParts) + "."


def extractFromRow(currentRow):
    """
    Apply the rule-based baseline to one dataset row and return a prediction row.
    """
    startTimeStamp = time.perf_counter()

    sentAtDateTime = getSentAtDateTime(currentRow["sent_at"])
    fullText = getCombinedText(currentRow["subject"], currentRow["body"])

    eventCategory = detectEventCategory(fullText)
    actionRequired, actionType = detectActionType(fullText)
    startTimeValue, endTimeValue = extractTimes(fullText)
    foundDates = findAllDates(fullText, sentAtDateTime)

    eventDate = ""
    actionDeadline = ""

    if len(foundDates) > 0:
        # First try to identify whether one of the dates is a deadline.
        actionDeadline = chooseActionDeadline(
            fullText,
            foundDates,
            actionRequired,
            actionType,
            eventDate
        )

        # Then choose the main event date from the remaining date options.
        eventDate = chooseEventDate(foundDates, actionDeadline)

        # For simple reminder-style actions, the event date itself can be used
        # as the effective deadline if no separate deadline was found.
        if actionDeadline == "" and (actionType == "bring_item" or actionType == "attend"):
            actionDeadline = eventDate

    calendarEventRequired = False
    if eventCategory != "none":
        calendarEventRequired = True

    # If there is only an action and no actual event, keep the calendar flag false.
    if eventCategory == "none" and actionRequired:
        calendarEventRequired = False

    # Keep the output logically consistent with the schema.
    if not calendarEventRequired:
        eventCategory = "none"
        eventDate = ""
        startTimeValue = ""
        endTimeValue = ""

    if not actionRequired:
        actionType = "none"
        actionDeadline = ""

    summaryText = buildSummary(
        eventCategory,
        eventDate,
        startTimeValue,
        endTimeValue,
        actionRequired,
        actionType,
        actionDeadline
    )

    endTimeStamp = time.perf_counter()
    latencyMilliseconds = round((endTimeStamp - startTimeStamp) * 1000, 2)

    resultRow = {
        "message_id": currentRow["message_id"],
        "provider": "baseline_rules",
        "pred_calendar_event_required": str(calendarEventRequired).lower(),
        "pred_event_category": eventCategory,
        "pred_event_date": eventDate,
        "pred_start_time": startTimeValue,
        "pred_end_time": endTimeValue,
        "pred_action_required": str(actionRequired).lower(),
        "pred_action_type": actionType,
        "pred_action_deadline": actionDeadline,
        "pred_summary": summaryText,
        "latency_ms": latencyMilliseconds
    }

    return resultRow


def validatePredictionValues(predictionRow):
    """
    Ensure the predicted categorical values stay inside the allowed schema.

    This is a safety check so that downstream evaluation does not break if a
    rule accidentally produces an invalid label.
    """
    if predictionRow["pred_event_category"] not in EVENT_CATEGORIES:
        predictionRow["pred_event_category"] = "none"

    if predictionRow["pred_action_type"] not in ACTION_TYPES:
        predictionRow["pred_action_type"] = "none"

    return predictionRow


def main():
    """
    Run the baseline extractor across the full dataset and save predictions.
    """
    dataFrame = loadDataset()

    if dataFrame is None:
        return

    predictionRows = []

    for rowIndex in range(len(dataFrame)):
        currentRow = dataFrame.iloc[rowIndex]
        predictionRow = extractFromRow(currentRow)
        predictionRow = validatePredictionValues(predictionRow)
        predictionRows.append(predictionRow)

    predictionDataFrame = pd.DataFrame(predictionRows)

    os.makedirs("outputs", exist_ok=True)
    predictionDataFrame.to_csv(outputFilePath, index=False)

    print("Baseline extraction complete.")
    print("Predictions saved to:", outputFilePath)
    print("Total predictions:", len(predictionDataFrame))


if __name__ == "__main__":
    main()