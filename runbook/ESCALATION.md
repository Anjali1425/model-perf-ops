# Escalation Runbook

## Thresholds
- Faithfulness score below 0.85 on more than 10% of sampled traces
- p95 latency above 3 seconds
- Cost per session above $0.05

## When a threshold breaks
1. Check Langfuse dashboard: is this one bad trace or a trend over the last 20+ requests?
2. If a trend: check what changed recently (prompt edit, doc update, model version)
3. If nothing changed: check for a retrieval failure (wrong docs surfaced)
4. Write the fix, however small
5. Run it through the Promptfoo gate before redeploying
6. Log the incident in `reviews/`, even a single sentence, so the pattern is visible later
