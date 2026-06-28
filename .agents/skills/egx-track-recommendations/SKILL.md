# EGX Track Recommendations

This skill monitors the EGX Bot's daily GitHub Actions run and processes recommendations data.

## What it does

1. **Checks GitHub Actions workflow status** for today (did the daily report run succeed?)
2. **If succeeded**: Clones the repo, reads `data/recommendations_YYYY-MM-DD.json`, and outputs it for entity tracking
3. **If failed or no run**: Sends an immediate Telegram alert to the user, then triggers `workflow_dispatch` to retry
4. **Waits 20 minutes**, checks the retry result:
   - If retry succeeded: sends a "✅ retry successful" Telegram message and processes the JSON
   - If retry also failed: sends a "❌ manual intervention needed" Telegram alert with details

## When it runs

Scheduled automation at **9:15 AM Cairo time** (7:15 AM UTC) — 15 minutes after the GitHub Actions cron (9:00 AM Cairo).

## Environment variables required

- `GITHUB_TOKEN` — GitHub PAT with `repo` and `actions` scope
- `TELEGRAM_BOT_TOKEN` — Telegram bot token for sending alerts
- `TELEGRAM_CHAT_ID` — User's Telegram chat ID (default: 7534010234)

## Telegram alerts sent

| Scenario | Alert message |
|----------|---------------|
| Workflow failed/missing | ⚠️ تنبيه EGX Bot — [date] — تقرير اليوم فشل... جاري إعادة التشغيل |
| Retry triggered but failed | ❌ تنبيه عاجل EGX Bot — محتاج تدخل يدوي |
| Retry succeeded | ✅ EGX Bot — تمت إعادة التشغيل بنجاح |
| Cannot trigger retry | ❌ تنبيه عاجل — فشل تشغيل إعادة المحاولة |

## JSON processing

When the workflow succeeds, the script outputs the recommendations JSON between markers:
```
=== JSON_DATA_START ===
{...json data...}
=== JSON_DATA_END ===
```
The agent reads this output and saves recommendations to the `RecommendationHistory` entity, then evaluates past recommendations due for 7-day or 30-day checks.
