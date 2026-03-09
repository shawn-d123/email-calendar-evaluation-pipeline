# Label Guide

This document defines the gold labels used in the evaluation dataset.

## General Rules

- Each row represents one message only.
- Extract only one primary calendar event.
- Extract only one primary action.
- If there is no event, event fields must be null.
- If there is no action, action fields must be null and action_type must be none.
- Dates must use YYYY-MM-DD.
- Times must use HH:MM in 24-hour format.
- Missing values must be left blank in the CSV and treated as null later.
- Summary must be one sentence with a maximum of 20 words.

## Fields

### gold_calendar_event_required
- true
- false

### gold_event_category
Allowed values:
- none
- meeting_admin
- club_activity
- trip
- payment_deadline
- cancellation_change
- reminder_other

### gold_event_date
Primary event date in YYYY-MM-DD format.

### gold_start_time
Primary event start time in HH:MM format.

### gold_end_time
Primary event end time in HH:MM format.

### gold_action_required
- true
- false

### gold_action_type
Allowed values:
- none
- attend
- pay
- reply_confirm
- bring_item
- submit_form

### gold_action_deadline
Deadline for the action in YYYY-MM-DD format.

### gold_summary
Short one-sentence summary of the message.

## Edge Case Tags

Allowed values:
- explicit_date
- relative_date
- time_range
- missing_time
- deadline_only
- cancellation
- ambiguous_wording
- long_message
- multiple_actions
- no_event