import json
import os
import time
import pandas as pd
from typing import Literal
from ollama import chat
from pydantic import BaseModel, ConfigDict, ValidationError


# Model and file paths for the first LLM benchmark.
modelName = "qwen3:8b"
inputFilePath = os.path.join("data", "processed", "eval_dataset.csv")
outputFilePath = os.path.join("outputs", "qwen_predictions.csv")


class ExtractionResult(BaseModel):
    """
    Structured output schema for the extractor.

    This matches the fields used by the baseline so both providers can be
    evaluated with the same pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    calendar_event_required: bool
    event_category: Literal[
        "none",
        "meeting_admin",
        "club_activity",
        "trip",
        "payment_deadline",
        "cancellation_change",
        "reminder_other"
    ]
    event_date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    action_required: bool
    action_type: Literal[
        "none",
        "attend",
        "pay",
        "reply_confirm",
        "bring_item",
        "submit_form"
    ]
    action_deadline: str | None = None
    summary: str


def loadDataset():
    """
    Load the evaluation dataset from CSV.

    Returns:
        pandas.DataFrame | None:
            The dataset if the file exists, otherwise None.
    """
    if not os.path.exists(inputFilePath):
        print("Dataset file not found:", inputFilePath)
        return None

    return pd.read_csv(inputFilePath)


def normaliseOptionalText(value):
    """
    Convert blank or missing values into an empty string for CSV output.
    """
    if value is None:
        return ""

    valueAsText = str(value).strip()
    if valueAsText == "":
        return ""

    return valueAsText


def getResponseContent(responseObject):
    """
    Read the model text from the Ollama response.

    The SDK examples use response.message.content, but this helper also
    supports dict-style access so the script stays robust.
    """
    messageObject = getattr(responseObject, "message", None)

    if messageObject is not None and hasattr(messageObject, "content"):
        return messageObject.content

    if isinstance(responseObject, dict):
        if "message" in responseObject and "content" in responseObject["message"]:
            return responseObject["message"]["content"]

    raise ValueError("Could not read message content from Ollama response.")


def buildSystemPrompt():
    """
    Build the system instructions for the extractor.

    The goal is to keep the model focused on deterministic extraction rather
    than open-ended chat behaviour.
    """
    return (
        "You extract structured calendar and action information from school, club, "
        "and admin messages. Follow the schema exactly. "
        "Return JSON only. Do not include explanations."
    )


def buildUserPrompt(currentRow):
    """
    Build the user prompt for one dataset row.

    The schema is included in the prompt as extra grounding so the model sees
    both the enforced JSON schema and the field rules in plain text.
    """
    schemaText = json.dumps(ExtractionResult.model_json_schema(), indent=2)

    promptText = f"""
Extract the primary calendar event and primary action from this message.

Rules:
- Use sent_at as the reference point for relative dates like "tomorrow".
- If there is no calendar event, set calendar_event_required to false, event_category to "none", and event date/time fields to null.
- If there is no action, set action_required to false, action_type to "none", and action_deadline to null.
- Dates must use YYYY-MM-DD format.
- Times must use HH:MM in 24-hour format.
- Summary must be one sentence and no more than 20 words.
- Extract only one primary event and one primary action.

Allowed event_category values:
- none
- meeting_admin
- club_activity
- trip
- payment_deadline
- cancellation_change
- reminder_other

Allowed action_type values:
- none
- attend
- pay
- reply_confirm
- bring_item
- submit_form

JSON schema:
{schemaText}

Message metadata:
sent_at: {currentRow["sent_at"]}
subject: {currentRow["subject"]}
body: {currentRow["body"]}
""".strip()

    return promptText


def buildFallbackPrediction(currentRow, errorMessage, latencyMilliseconds):
    """
    Build a safe fallback prediction if the model response cannot be parsed.

    This keeps the batch running even if one row fails.
    """
    return {
        "message_id": currentRow["message_id"],
        "provider": modelName.replace(":", "_"),
        "pred_calendar_event_required": "false",
        "pred_event_category": "none",
        "pred_event_date": "",
        "pred_start_time": "",
        "pred_end_time": "",
        "pred_action_required": "false",
        "pred_action_type": "none",
        "pred_action_deadline": "",
        "pred_summary": "LLM extraction failed.",
        "latency_ms": latencyMilliseconds,
        "parse_status": "failed",
        "error_message": errorMessage
    }


def extractFromRow(currentRow):
    """
    Run one dataset row through the LLM extractor and return a prediction row.
    """
    startTimeStamp = time.perf_counter()

    try:
        responseObject = chat(
            model=modelName,
            messages=[
                {
                    "role": "system",
                    "content": buildSystemPrompt()
                },
                {
                    "role": "user",
                    "content": buildUserPrompt(currentRow)
                }
            ],
            format=ExtractionResult.model_json_schema(),
            options={
                "temperature": 0
            }
        )

        responseContent = getResponseContent(responseObject)
        validatedOutput = ExtractionResult.model_validate_json(responseContent)

        endTimeStamp = time.perf_counter()
        latencyMilliseconds = round((endTimeStamp - startTimeStamp) * 1000, 2)

        return {
            "message_id": currentRow["message_id"],
            "provider": modelName.replace(":", "_"),
            "pred_calendar_event_required": str(validatedOutput.calendar_event_required).lower(),
            "pred_event_category": validatedOutput.event_category,
            "pred_event_date": normaliseOptionalText(validatedOutput.event_date),
            "pred_start_time": normaliseOptionalText(validatedOutput.start_time),
            "pred_end_time": normaliseOptionalText(validatedOutput.end_time),
            "pred_action_required": str(validatedOutput.action_required).lower(),
            "pred_action_type": validatedOutput.action_type,
            "pred_action_deadline": normaliseOptionalText(validatedOutput.action_deadline),
            "pred_summary": normaliseOptionalText(validatedOutput.summary),
            "latency_ms": latencyMilliseconds,
            "parse_status": "success",
            "error_message": ""
        }

    except ValidationError as validationError:
        endTimeStamp = time.perf_counter()
        latencyMilliseconds = round((endTimeStamp - startTimeStamp) * 1000, 2)

        return buildFallbackPrediction(
            currentRow,
            "Validation error: " + str(validationError),
            latencyMilliseconds
        )

    except Exception as error:
        endTimeStamp = time.perf_counter()
        latencyMilliseconds = round((endTimeStamp - startTimeStamp) * 1000, 2)

        return buildFallbackPrediction(
            currentRow,
            str(error),
            latencyMilliseconds
        )


def main():
    """
    Run the LLM extractor across the full dataset and save predictions.
    """
    dataFrame = loadDataset()

    if dataFrame is None:
        return

    predictionRows = []

    for rowIndex in range(len(dataFrame)):
        currentRow = dataFrame.iloc[rowIndex]
        print("Running row", rowIndex + 1, "of", len(dataFrame), "-", currentRow["message_id"])

        predictionRow = extractFromRow(currentRow)
        predictionRows.append(predictionRow)

    predictionDataFrame = pd.DataFrame(predictionRows)

    os.makedirs("outputs", exist_ok=True)
    predictionDataFrame.to_csv(outputFilePath, index=False)

    print("LLM extraction complete.")
    print("Predictions saved to:", outputFilePath)
    print("Total predictions:", len(predictionDataFrame))


if __name__ == "__main__":
    main()