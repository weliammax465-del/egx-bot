# EGX Track Recommendations

This skill processes the daily recommendations JSON file from the EGX Bot GitHub repository
and saves it to the RecommendationHistory entity for performance tracking.

## What it does

1. Clones (or pulls) the EGX Bot GitHub repo
2. Finds the latest `data/recommendations_YYYY-MM-DD.json` file
3. Saves new recommendations to the `RecommendationHistory` entity (via `create_entity_records`)
4. Evaluates past recommendations that are due for 7-day or 30-day check (using `current_prices` from the JSON)
5. Updates evaluated records with PnL, result, and status

## When to run

This skill is called by a scheduled automation at 9:30 AM Cairo time (30 minutes after the bot's daily run).

## Inputs

- `json_date` (optional): Specific date to process (YYYY-MM-DD). If omitted, uses today's date.

## Evaluation criteria

- Buy: pnl = (current - recommended) / recommended * 100
- Sell: pnl = (recommended - current) / recommended * 100
- pnl > 3% = win, pnl < -3% = loss, between = neutral
