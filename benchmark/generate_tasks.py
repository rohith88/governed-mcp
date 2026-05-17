"""
Generate benchmark/tasks.json from registry/real_tools.json.

One task per tool (up to 500). Each task has:
  - instruction: business-scenario description (NO service names)
  - correct_tool: exact tool name
  - attributes: semantic attribute tag(s)
  - service: source service
  - difficulty: easy / medium / hard
  - distractor_cluster: [3 distractor tool names]

Instructions deliberately omit service names so the model cannot trivially
name-match the instruction to a tool. The client is assumed to already know
which attribute to pass (?attribute=payments); the benchmark tests whether
the LLM picks the RIGHT tool from the filtered set.

Difficulty:
  easy   → distractors from different attributes (cross-domain)
  medium → distractors from same attribute, different service
  hard   → distractors from same service (same-service near-miss)

Usage:
    python benchmark/generate_tasks.py
    python benchmark/generate_tasks.py --registry registry/real_tools.json --output benchmark/tasks.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

random.seed(42)

# ── Noun translation: strip service-specific jargon → business language ────────

NOUN_TRANSLATE: dict[str, str] = {
    "payment intent":       "payment transaction",
    "payment intents":      "payment transactions",
    "pull request":         "code review request",
    "pull requests":        "code review requests",
    "direct message":       "private message",
    "direct messages":      "private messages",
    "checkout session":     "checkout session",
    "workflow run":         "pipeline run",
    "workflow runs":        "pipeline runs",
    "cohort":               "user segment",
    "cohorts":              "user segments",
    "funnel":               "conversion funnel",
    "funnels":              "conversion funnels",
    "funnel analysis":      "conversion analysis",
    "retention analysis":   "retention report",
    "event stats":          "event metrics",
    "segmentation":         "user segmentation report",
    "presigned url":        "temporary access link",
    "bucket policy":        "storage access policy",
    "lifecycle policy":     "storage lifecycle rule",
    "public access block":  "public access settings",
    "object metadata":      "file metadata",
    "object versions":      "file version history",
    "billing plan":         "subscription plan",
    "billing plans":        "subscription plans",
    "user profile":         "user profile",
    "user roles":           "assigned roles",
    "api":                  "analytics API",
    "api key":              "API credential",
    "api keys":             "API credentials",
    "open stats":           "email open metrics",
    "click stats":          "email click metrics",
    "whatsapp messages":    "WhatsApp messages",
    "satisfaction ratings":  "satisfaction scores",
    "satisfaction rating":   "satisfaction score",
    "balance transactions": "balance history",
    "from starred":         "favourites",
    "to starred":           "favourites",
    "group identify":       "group traits",
}


def _translate_noun(noun: str) -> str:
    return NOUN_TRANSLATE.get(noun, noun)


# ── Business-scenario templates keyed by (attribute, verb) ─────────────────────
# {noun} is filled with the business-translated noun from the tool name.
# Templates never mention service names (Stripe, GitHub, Slack, etc.).

SCENARIO_TEMPLATES: dict[tuple[str, str], list[str]] = {

    # ── payments ──────────────────────────────────────────────────────────────
    ("payments", "create"): [
        "A customer is ready to pay — set up a new {noun} to begin the transaction.",
        "Finance needs a {noun} created for this account.",
        "The order is confirmed; create a {noun} to collect the funds.",
        "Initiate a {noun} as part of the checkout workflow.",
    ],
    ("payments", "get"): [
        "A support agent needs to look up the {noun} to resolve a billing dispute.",
        "Retrieve the {noun} details to verify the transaction status.",
        "The customer is asking about their {noun} — fetch the record.",
        "Pull up the {noun} before issuing the refund.",
    ],
    ("payments", "list"): [
        "The finance team needs a list of {noun} for the monthly reconciliation.",
        "Generate a report of all {noun} for the current billing period.",
        "Pull all {noun} for the auditor's review.",
        "Show every {noun} associated with this account.",
    ],
    ("payments", "update"): [
        "The customer updated their billing details — reflect the change in the {noun}.",
        "Modify the {noun} to apply the latest pricing adjustment.",
        "Correct the {noun} record with the right information.",
    ],
    ("payments", "cancel"): [
        "The customer changed their mind — cancel the pending {noun} immediately.",
        "The order was flagged as fraudulent; cancel the {noun} before it settles.",
        "Stop the {noun} before it is processed.",
    ],
    ("payments", "capture"): [
        "The goods have shipped — capture the {noun} to collect the funds.",
        "The authorization window is closing; capture the {noun} now.",
        "Finalize the transaction by capturing the {noun}.",
    ],
    ("payments", "confirm"): [
        "The customer completed 3-D Secure authentication — confirm the {noun} to proceed.",
        "Confirm the {noun} after verifying all payment details.",
    ],
    ("payments", "void"): [
        "The invoice was issued in error — void the {noun} before the customer sees it.",
        "Invalidate the {noun} as it is no longer payable.",
    ],
    ("payments", "finalize"): [
        "The billing cycle is closing — finalize the {noun} and send it to the customer.",
        "Lock the {noun} to prepare it for collection.",
    ],
    ("payments", "pay"): [
        "Settle the outstanding {noun} on behalf of the customer.",
        "Process payment for the {noun} that is now due.",
    ],
    ("payments", "activate"): [
        "The customer upgraded their plan — activate the {noun} so service begins.",
        "Enable the {noun} so the customer can start using premium features.",
    ],
    ("payments", "deactivate"): [
        "The customer's trial has ended — deactivate the {noun}.",
        "Temporarily deactivate the {noun} pending an account review.",
    ],
    ("payments", "pause"): [
        "The customer is taking a break — pause the {noun} until they return.",
        "Put the {noun} on hold at the customer's request.",
    ],
    ("payments", "resume"): [
        "The customer is back — resume their paused {noun}.",
        "Re-activate the {noun} that was previously paused.",
    ],
    ("payments", "suspend"): [
        "Repeated payment failures — suspend the {noun} for non-payment.",
        "Suspend the {noun} due to a compliance hold.",
    ],
    ("payments", "archive"): [
        "The {noun} is discontinued — archive it to remove it from the active catalogue.",
        "Move the retired {noun} to the archive.",
    ],
    ("payments", "delete"): [
        "Remove the test {noun} created during QA.",
        "Permanently delete the {noun} that was created by mistake.",
    ],
    ("payments", "expire"): [
        "Force-expire the {noun} so the customer must start a fresh session.",
        "The session timed out — expire the {noun}.",
    ],
    ("payments", "send"): [
        "Email the {noun} to the customer so they can review and pay.",
        "Dispatch the {noun} to the client for settlement.",
    ],
    ("payments", "authorize"): [
        "Authorize the {noun} before capturing the funds.",
        "Pre-authorize the {noun} to reserve the amount.",
    ],

    # ── developer ─────────────────────────────────────────────────────────────
    ("developer", "create"): [
        "A developer needs a new {noun} to track the upcoming work.",
        "Create a {noun} as part of the sprint planning process.",
        "The team needs a {noun} to get the feature started.",
        "Kick off the work by creating a {noun} in the project tracker.",
    ],
    ("developer", "get"): [
        "Fetch the {noun} to review its current state before the stand-up.",
        "A team member needs the {noun} details to continue their work.",
        "Retrieve the {noun} to prepare the release notes.",
        "Pull up the {noun} to investigate the reported regression.",
    ],
    ("developer", "list"): [
        "The engineering manager wants to see all {noun} for this sprint.",
        "List the {noun} to triage and prioritize incoming work.",
        "Show all open {noun} so the team can plan capacity.",
        "Browse the {noun} to find the one causing the build failure.",
    ],
    ("developer", "delete"): [
        "The {noun} is stale and no longer needed — remove it.",
        "Clean up the project by deleting the {noun} from a previous sprint.",
        "Remove the {noun} that was accidentally created.",
    ],
    ("developer", "update"): [
        "Requirements changed — update the {noun} to reflect the new scope.",
        "Edit the {noun} with findings from the code review.",
        "Revise the {noun} after the team discussion.",
    ],
    ("developer", "add"): [
        "Add a {noun} to help categorize this item for the team.",
        "Attach a {noun} to provide additional context for the reviewers.",
    ],
    ("developer", "assign"): [
        "Assign the {noun} to the engineer who will fix it.",
        "The {noun} needs an owner — assign it to a team member.",
    ],
    ("developer", "close"): [
        "The fix has been deployed — close the {noun}.",
        "Mark the {noun} as resolved after confirming the fix works.",
    ],
    ("developer", "merge"): [
        "The {noun} has been approved — merge it into the main branch.",
        "Combine the {noun} after all CI checks pass.",
    ],
    ("developer", "protect"): [
        "Prevent accidental force-pushes by enabling protection on the {noun}.",
        "Add branch protection rules to the {noun} before the release.",
    ],
    ("developer", "fork"): [
        "Create an independent copy of the {noun} to experiment without affecting the original.",
        "Fork the {noun} to start a derivative project.",
    ],
    ("developer", "search"): [
        "Search the {noun} for the error pattern reported by the customer.",
        "Find a {noun} related to this feature area to avoid duplicating work.",
    ],
    ("developer", "trigger"): [
        "Manually trigger the {noun} to verify the hotfix deployment.",
        "Kick off the {noun} after the emergency patch is merged.",
    ],
    ("developer", "cancel"): [
        "The build is running on an outdated branch — cancel the {noun}.",
        "Stop the {noun} that was triggered by mistake.",
    ],
    ("developer", "complete"): [
        "All stories are done — mark the {noun} as complete.",
        "Close the development cycle by completing the {noun}.",
    ],
    ("developer", "start"): [
        "Begin the next development cycle by starting the {noun}.",
        "Start the {noun} so the team can log their work against it.",
    ],
    ("developer", "archive"): [
        "Archive the old {noun} to keep the project history clean.",
        "Move the deprecated {noun} to the archive.",
    ],
    ("developer", "release"): [
        "Ship the code by publishing the {noun} to production.",
        "Cut the {noun} for the scheduled deployment window.",
    ],
    ("developer", "compare"): [
        "Compare the {noun} to pinpoint what changed before the regression.",
        "Show the diff between {noun} to review the delta.",
    ],
    ("developer", "submit"): [
        "Submit the {noun} after completing the code inspection.",
        "Post the {noun} so the author can see the reviewer's feedback.",
    ],
    ("developer", "transition"): [
        "Move the {noun} to the next stage in the workflow.",
        "Advance the {noun} from 'In Progress' to 'In Review'.",
    ],
    ("developer", "transfer"): [
        "The project is changing ownership — transfer the {noun} to the new team.",
        "Move the {noun} to a different organization.",
    ],
    ("developer", "remove"): [
        "Revoke access by removing the {noun} from the project.",
        "The contractor's contract ended — remove the {noun}.",
    ],

    # ── messaging ─────────────────────────────────────────────────────────────
    ("messaging", "send"): [
        "Alert the customer by sending a {noun} about their account update.",
        "The campaign is ready — send the {noun} to the contact list.",
        "Deliver the {noun} with the user's access credentials.",
        "Notify the team member by sending a {noun}.",
    ],
    ("messaging", "post"): [
        "Post a {noun} to the team channel with the deployment update.",
        "Share a {noun} announcing the release to the channel.",
    ],
    ("messaging", "create"): [
        "Set up a new {noun} for the project team to collaborate.",
        "Provision a {noun} to organize communications for this initiative.",
        "Create a {noun} for the new support queue.",
    ],
    ("messaging", "delete"): [
        "Remove the {noun} that was sent with incorrect information.",
        "Delete the {noun} to clean up the conversation history.",
    ],
    ("messaging", "get"): [
        "Retrieve the {noun} to check its delivery and open status.",
        "Look up the {noun} for the audit trail.",
    ],
    ("messaging", "list"): [
        "Show all {noun} to review recent communications.",
        "List the {noun} to find the one sent to the customer.",
    ],
    ("messaging", "invite"): [
        "Invite the new team member to the {noun} so they can join discussions.",
        "Add a stakeholder to the {noun} for visibility.",
    ],
    ("messaging", "archive"): [
        "The project wrapped up — archive the {noun} to declutter the workspace.",
        "Move the inactive {noun} to the archive.",
    ],
    ("messaging", "add"): [
        "Add a {noun} to acknowledge the key message in the thread.",
        "React with a {noun} to signal the team that work is complete.",
    ],
    ("messaging", "remove"): [
        "Remove the {noun} that was added by mistake.",
        "Clear the {noun} from the message to keep the channel clean.",
    ],
    ("messaging", "pin"): [
        "Pin the important {noun} so the team can find it easily.",
        "Keep the {noun} visible at the top of the channel.",
    ],
    ("messaging", "unpin"): [
        "Unpin the {noun} now that the information is out of date.",
        "Remove the pin from the {noun} to declutter the channel header.",
    ],
    ("messaging", "rename"): [
        "Rename the {noun} to better reflect the project's current scope.",
        "Update the {noun} name after the team rebranded the initiative.",
    ],
    ("messaging", "schedule"): [
        "Schedule the {noun} to go out at the start of business tomorrow.",
        "Queue the {noun} for delivery during peak engagement hours.",
    ],
    ("messaging", "share"): [
        "Share the {noun} with the team so everyone has access.",
        "Post the {noun} in the channel for the team to review.",
    ],
    ("messaging", "upload"): [
        "Upload the {noun} so it can be shared in the conversation.",
        "Attach the {noun} to the channel for the team.",
    ],
    ("messaging", "make"): [
        "Initiate a {noun} with the customer for a live support session.",
        "Place a {noun} to follow up on the open case.",
    ],
    ("messaging", "cancel"): [
        "Cancel the {noun} that was dialled to the wrong number.",
        "Stop the {noun} before it connects.",
    ],
    ("messaging", "modify"): [
        "Adjust the {noun} to route it to the correct support agent.",
        "Update the {noun} parameters mid-session.",
    ],
    ("messaging", "buy"): [
        "Purchase a new {noun} for the marketing campaign.",
        "Acquire a {noun} for the new regional support line.",
    ],
    ("messaging", "release"): [
        "Release the {noun} that is no longer in use.",
        "Return the {noun} to free up the resource.",
    ],
    ("messaging", "open"): [
        "Open a {noun} with the team member to discuss a sensitive issue privately.",
        "Start a {noun} with the customer for a confidential conversation.",
    ],
    ("messaging", "set"): [
        "Update the {noun} to show the team member is out of office.",
        "Set the {noun} to reflect the user's current availability.",
    ],
    ("messaging", "pin"): [
        "Pin the important {noun} so the team can find it easily.",
        "Keep the {noun} at the top of the channel for visibility.",
    ],

    # ── crm ───────────────────────────────────────────────────────────────────
    ("crm", "create"): [
        "A sales rep just finished a call — log the new {noun} in the system.",
        "Create a {noun} to start tracking this customer relationship.",
        "Add a {noun} from the latest marketing campaign lead.",
    ],
    ("crm", "get"): [
        "Look up the {noun} to prepare for the upcoming customer call.",
        "Pull the {noun} details before the sales meeting.",
        "Retrieve the {noun} to review the account history.",
    ],
    ("crm", "list"): [
        "Show all {noun} so the sales team can prioritize outreach.",
        "Pull the list of {noun} for the quarterly business review.",
        "List all active {noun} for the account manager's pipeline.",
    ],
    ("crm", "update"): [
        "Update the {noun} with the new information gathered on the call.",
        "Edit the {noun} to reflect the latest status after the meeting.",
        "Revise the {noun} following the customer's feedback.",
    ],
    ("crm", "delete"): [
        "Remove the duplicate {noun} that was created in error.",
        "Delete the {noun} for the churned account.",
    ],
    ("crm", "search"): [
        "Search for the {noun} associated with this email address.",
        "Find the {noun} for the customer who called the support line.",
    ],
    ("crm", "add"): [
        "Add a {noun} to document the outcome of the support call.",
        "Log a {noun} after the team discussion about this account.",
    ],
    ("crm", "assign"): [
        "Assign the {noun} to the account executive for follow-up.",
        "Route the {noun} to the appropriate support agent.",
    ],
    ("crm", "close"): [
        "The negotiation is complete — close the {noun} as won.",
        "Mark the {noun} as resolved and close it out.",
    ],
    ("crm", "complete"): [
        "Mark the {noun} as done after the follow-up call.",
        "Complete the {noun} to advance the deal to the next stage.",
    ],
    ("crm", "merge"): [
        "The customer has duplicate records — merge the {noun} into one.",
        "Consolidate the {noun} to create a single source of truth.",
    ],
    ("crm", "solve"): [
        "The issue has been fixed — mark the {noun} as solved.",
        "Close the loop by resolving the {noun} after the patch.",
    ],
    ("crm", "suspend"): [
        "Suspicious activity detected — suspend the {noun} pending review.",
        "Put the {noun} on hold while the compliance team investigates.",
    ],
    ("crm", "log"): [
        "Log the {noun} to keep a record of the customer interaction.",
        "Document the {noun} in the CRM after the meeting.",
    ],
    ("crm", "apply"): [
        "Apply the {noun} to automate handling of this request category.",
        "Use the {noun} to streamline the repetitive support workflow.",
    ],
    ("crm", "remove"): [
        "Remove the {noun} who has left the team.",
        "Take the {noun} out of the group as they no longer need access.",
    ],

    # ── analytics ─────────────────────────────────────────────────────────────
    ("analytics", "track"): [
        "A user completed a key action — record the {noun} for analysis.",
        "Log the {noun} to capture how users interact with the feature.",
        "Capture the {noun} as part of the product funnel measurement.",
    ],
    ("analytics", "identify"): [
        "Send updated profile traits for the {noun} to the analytics platform.",
        "Associate the {noun} attributes with the user's analytics record.",
    ],
    ("analytics", "create"): [
        "Build a new {noun} to monitor the key product metric this quarter.",
        "Set up a {noun} to track performance against the OKR.",
        "The product team needs a new {noun} for the A/B test analysis.",
    ],
    ("analytics", "get"): [
        "Pull the {noun} to include in the weekly performance review.",
        "Fetch the {noun} results to share with the product team.",
        "Retrieve the {noun} for the board presentation.",
    ],
    ("analytics", "list"): [
        "Show all {noun} the team has set up so far.",
        "List existing {noun} to find the one for this campaign.",
    ],
    ("analytics", "delete"): [
        "Remove the {noun} for a user who submitted a data deletion request.",
        "Delete the {noun} as part of GDPR right-to-erasure compliance.",
    ],
    ("analytics", "update"): [
        "Update the {noun} with the revised cohort definition.",
        "Edit the {noun} to include the new event properties.",
    ],
    ("analytics", "export"): [
        "Export the {noun} to share with the data science team for modelling.",
        "Download the {noun} for offline analysis in the BI tool.",
    ],
    ("analytics", "import"): [
        "Import historical {noun} to backfill the analytics platform.",
        "Load {noun} from the data warehouse into the analytics system.",
    ],
    ("analytics", "query"): [
        "Run a query against the {noun} to answer the product team's question.",
        "Query the {noun} to generate the custom performance report.",
    ],
    ("analytics", "group"): [
        "Group the {noun} data to analyse behaviour by account type.",
        "Segment {noun} to understand usage patterns across plans.",
    ],

    # ── storage ───────────────────────────────────────────────────────────────
    ("storage", "create"): [
        "Create a new {noun} to organize the project's assets.",
        "Set up a {noun} to store the files for this initiative.",
        "Provision a {noun} for the team to collaborate on.",
    ],
    ("storage", "get"): [
        "Retrieve the {noun} to share it with a stakeholder.",
        "Fetch the {noun} to review its contents before the meeting.",
        "Pull the {noun} for the compliance audit trail.",
    ],
    ("storage", "list"): [
        "List all {noun} in the project workspace.",
        "Browse the {noun} to locate the latest version of the report.",
        "Show the {noun} to find the document the client requested.",
    ],
    ("storage", "delete"): [
        "Remove the {noun} that is no longer needed to free up space.",
        "Permanently delete the sensitive {noun} after it has been archived.",
        "Delete the {noun} left over from the cancelled project.",
    ],
    ("storage", "update"): [
        "Update the {noun} with the revised content from the team.",
        "Edit the {noun} to reflect the latest approved changes.",
    ],
    ("storage", "upload"): [
        "Upload the {noun} so it is available to the whole team.",
        "Transfer the {noun} to cloud storage for backup.",
    ],
    ("storage", "download"): [
        "Download the {noun} for offline review before the client meeting.",
        "Export the {noun} to the local machine for processing.",
    ],
    ("storage", "copy"): [
        "Copy the {noun} to create a backup before making edits.",
        "Duplicate the {noun} to reuse it in another project.",
    ],
    ("storage", "move"): [
        "Move the {noun} to the archive folder now that the project is closed.",
        "Relocate the {noun} to the correct team folder.",
    ],
    ("storage", "rename"): [
        "Rename the {noun} to conform to the new naming convention.",
        "Update the {noun} name so it is easier for the team to find.",
    ],
    ("storage", "share"): [
        "Share the {noun} with the external partner for their review.",
        "Grant the client access to the {noun} for sign-off.",
    ],
    ("storage", "export"): [
        "Export the {noun} in the format the client requested.",
        "Download the {noun} to send to the stakeholder offline.",
    ],
    ("storage", "put"): [
        "Put the {noun} into cloud storage so the deployment pipeline can access it.",
        "Upload the {noun} to the designated storage location.",
    ],
    ("storage", "add"): [
        "Add a {noun} to the document before sending it for signature.",
        "Include the {noun} in the workflow to complete the process.",
    ],
    ("storage", "remove"): [
        "Remove the {noun} from the document.",
        "Revoke the {noun} so the former employee loses access.",
    ],
    ("storage", "append"): [
        "Append new content to the {noun} without overwriting existing text.",
        "Add additional {noun} to the bottom of the page.",
    ],
    ("storage", "archive"): [
        "Archive the {noun} once the project is complete.",
        "Move the old {noun} to the archive to declutter the workspace.",
    ],
    ("storage", "bulk"): [
        "Process a batch of {noun} in a single operation to save time.",
        "Handle multiple {noun} updates at once.",
    ],
    ("storage", "enable"): [
        "Enable {noun} on the storage bucket to meet the compliance requirement.",
        "Turn on {noun} as required by the security policy.",
    ],
    ("storage", "empty"): [
        "Empty the {noun} to reclaim storage space before deleting it.",
        "Clear all items from the {noun} as part of the cleanup.",
    ],
    ("storage", "send"): [
        "Send the {noun} to the signatories so they can complete it.",
        "Dispatch the {noun} to the recipients for review and approval.",
    ],
    ("storage", "set"): [
        "Configure the {noun} on the storage bucket for the new environment.",
        "Apply the {noun} settings to meet the data governance requirement.",
    ],
    ("storage", "query"): [
        "Query the {noun} to retrieve the specific records the team needs.",
        "Run a query against the {noun} to generate the report.",
    ],
    ("storage", "search"): [
        "Search the {noun} for the document the auditor requested.",
        "Find the {noun} matching the search criteria.",
    ],
    ("storage", "reply"): [
        "Reply to the {noun} with the team's feedback.",
        "Respond to the {noun} to continue the document review thread.",
    ],

    # ── identity ──────────────────────────────────────────────────────────────
    ("identity", "create"): [
        "Provision a {noun} for the new employee joining the team today.",
        "Set up a {noun} so the contractor can access the required systems.",
        "Create a {noun} as part of the onboarding workflow.",
    ],
    ("identity", "get"): [
        "Look up the {noun} to verify the user's current access level.",
        "Retrieve the {noun} details for the quarterly security audit.",
        "Fetch the {noun} to investigate the failed login attempt.",
    ],
    ("identity", "list"): [
        "List all {noun} for the access review — flag any over-privileged accounts.",
        "Show all {noun} to audit who currently has access.",
        "Pull the {noun} to check for any stale accounts.",
    ],
    ("identity", "update"): [
        "The user changed roles — update the {noun} to reflect their new permissions.",
        "Edit the {noun} after the department restructure.",
    ],
    ("identity", "delete"): [
        "The employee has left the company — delete the {noun} to revoke access.",
        "Remove the {noun} as part of the offboarding checklist.",
    ],
    ("identity", "assign"): [
        "Assign the appropriate {noun} to the new team member based on their job function.",
        "Grant the {noun} to the user so they can access the required resources.",
    ],
    ("identity", "remove"): [
        "Remove the {noun} from the user who moved to a different team.",
        "Revoke the {noun} now that the project has ended.",
    ],
    ("identity", "block"): [
        "Suspicious login activity detected — block the {noun} immediately.",
        "Temporarily restrict the {noun} pending a security investigation.",
    ],
    ("identity", "unblock"): [
        "The investigation cleared the user — unblock the {noun} to restore access.",
        "Unblock the {noun} now that the false-positive alert has been resolved.",
    ],
    ("identity", "search"): [
        "Search for the {noun} associated with this email address.",
        "Find the {noun} to reset their access credentials.",
    ],
    ("identity", "send"): [
        "Trigger a {noun} so the user can regain access to their account.",
        "Send a {noun} to the user who forgot their login credentials.",
    ],
}

# Fallback templates for (attribute, verb) pairs not explicitly listed
_FALLBACK: list[str] = [
    "Handle the {noun} operation as required.",
    "Perform the {verb} action on the {noun}.",
    "Execute the {verb} request for the {noun}.",
]

ATTR_PREFIX = {
    "payments":  "PAY",
    "developer": "DEV",
    "messaging": "MSG",
    "crm":       "CRM",
    "identity":  "IDN",
    "analytics": "ANL",
    "storage":   "STR",
}


# ── Difficulty & distractor logic (unchanged) ─────────────────────────────────

_HARD_VERBS = {
    "delete", "void", "cancel", "expire", "block", "suspend", "revoke",
    "purge", "drop", "destroy", "disable", "deactivate",
}
_MEDIUM_VERBS = {
    "create", "update", "send", "trigger", "merge", "transition",
    "capture", "confirm", "finalize", "pay", "refund", "close",
    "activate", "transfer", "assign", "share", "invite", "add",
}


def _difficulty(tool: dict) -> str:
    action = tool["name"][len(tool["service"]) + 1:]
    verb = action.split("_")[0]
    if verb in _HARD_VERBS:
        return "hard"
    elif verb in _MEDIUM_VERBS:
        return "medium"
    return "easy"


def _distractors(
    tool: dict,
    all_tools: list[dict],
    tools_by_service: dict[str, list[dict]],
    tools_by_attr: dict[str, list[dict]],
    n: int = 3,
) -> list[str]:
    service = tool["service"]
    attr = tool["attributes"][0]
    name = tool["name"]

    same_svc  = [t["name"] for t in tools_by_service[service] if t["name"] != name]
    same_attr = [t["name"] for t in tools_by_attr[attr]
                 if t["service"] != service and t["name"] != name]
    cross_attr = [t["name"] for t in all_tools
                  if t["attributes"][0] != attr and t["name"] != name]

    chosen: list[str] = []
    if same_svc:
        chosen.append(random.choice(same_svc))
    if same_attr:
        chosen.append(random.choice(same_attr))
    remaining = n - len(chosen)
    if cross_attr and remaining > 0:
        chosen += random.sample(cross_attr, min(remaining, len(cross_attr)))
    return chosen[:n]


# ── Instruction builder ───────────────────────────────────────────────────────

def _instruction(tool: dict) -> str:
    name    = tool["name"]
    service = tool["service"]
    attr    = tool["attributes"][0]

    action = name[len(service) + 1:]
    parts  = action.split("_")
    verb   = parts[0]
    raw_noun = " ".join(parts[1:]) if len(parts) > 1 else "resource"
    noun   = _translate_noun(raw_noun)

    key = (attr, verb)
    templates = SCENARIO_TEMPLATES.get(key)

    if templates:
        tmpl = random.choice(templates)
        return tmpl.format(noun=noun, verb=verb)
    else:
        # Fallback: generic business phrasing without service name
        tmpl = random.choice(_FALLBACK)
        return tmpl.format(noun=noun, verb=verb)


# ── Main generator ────────────────────────────────────────────────────────────

def generate(registry_path: str, max_tasks: int = 500) -> list[dict]:
    with open(registry_path) as f:
        all_tools: list[dict] = json.load(f)

    tools_by_service: dict[str, list[dict]] = {}
    tools_by_attr:    dict[str, list[dict]] = {}
    for t in all_tools:
        tools_by_service.setdefault(t["service"], []).append(t)
        for a in t["attributes"]:
            tools_by_attr.setdefault(a, []).append(t)

    shuffled = all_tools[:]
    random.shuffle(shuffled)
    selected = shuffled[:max_tasks]

    attr_counters: dict[str, int] = {}
    tasks = []
    for tool in selected:
        attr   = tool["attributes"][0]
        prefix = ATTR_PREFIX.get(attr, attr[:3].upper())
        attr_counters[attr] = attr_counters.get(attr, 0) + 1
        task_id = f"{prefix}_{attr_counters[attr]:03d}"

        tasks.append({
            "task_id":          task_id,
            "instruction":      _instruction(tool),
            "correct_tool":     tool["name"],
            "service":          tool["service"],
            "attributes":       tool["attributes"],
            "difficulty":       _difficulty(tool),
            "distractor_cluster": _distractors(
                tool, all_tools, tools_by_service, tools_by_attr
            ),
        })

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry",  default="registry/real_tools.json")
    parser.add_argument("--output",    default="benchmark/tasks.json")
    parser.add_argument("--max-tasks", type=int, default=500)
    args = parser.parse_args()

    tasks = generate(args.registry, args.max_tasks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(tasks, indent=2))

    from collections import Counter
    diff = Counter(t["difficulty"]       for t in tasks)
    attr = Counter(t["attributes"][0]    for t in tasks)
    svc  = Counter(t["service"]          for t in tasks)
    print(f"Generated {len(tasks)} tasks → {args.output}")
    print(f"Difficulty: {dict(diff)}")
    print(f"By attribute: {dict(attr)}")
    print(f"By service (top 5): {dict(svc.most_common(5))}")


if __name__ == "__main__":
    main()
