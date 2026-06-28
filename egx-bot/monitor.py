#!/usr/bin/env python3
"""
monitor.py — EGX Bot Monitoring & Alerting System

Runs at 9:15 AM Cairo time (via Base44 automation).
Checks if today's GitHub Actions workflow succeeded.

Flow:
  1. Check GitHub Actions workflow run status for today
  2. If succeeded: clone repo, read recommendations JSON, output for entity tracking
  3. If failed or no run: send Telegram alert, trigger workflow_dispatch retry
  4. Wait 20 min, check retry result. If still failed: send "manual intervention" alert

Environment variables:
  GITHUB_TOKEN       — GitHub PAT with repo+actions scope
  TELEGRAM_BOT_TOKEN — Telegram bot token for sending alerts
  TELEGRAM_CHAT_ID   — Telegram chat ID (default: 7534010234)

Usage:
  python3 monitor.py              # Full monitoring run
  python3 monitor.py --test-alert # Send a test Telegram alert
  python3 monitor.py --check-only # Check status only, no actions
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

# ─── Configuration ───────────────────────────────────────────────────────────

REPO = "weliammax465-del/egx-bot"
WORKFLOW_FILE = "daily.yml"
WORKFLOW_NAME = "EGX Daily Technical Analysis Report"

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7534010234")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

CAIRO_TZ = timezone(timedelta(hours=2))
CLONE_DIR = "/tmp/egx-bot-monitor"

RETRY_WAIT_SECONDS = 1200  # 20 minutes
RUNNING_WAIT_SECONDS = 300  # 5 minutes


# ─── Helpers ─────────────────────────────────────────────────────────────────

def cairo_now() -> datetime:
    """Current time in Cairo."""
    return datetime.now(timezone.utc).astimezone(CAIRO_TZ)


def cairo_today_str() -> str:
    """Today's date in Cairo as YYYY-MM-DD."""
    return cairo_now().strftime("%Y-%m-%d")


def log(msg: str) -> None:
    """Print with timestamp."""
    ts = cairo_now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ─── Telegram ────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Send a message to Telegram. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN:
        log("❌ TELEGRAM_BOT_TOKEN not set — cannot send alert")
        return False
    if not TELEGRAM_CHAT_ID:
        log("❌ TELEGRAM_CHAT_ID not set — cannot send alert")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=15,
        )
        if resp.status_code == 200:
            log(f"✅ Telegram alert sent ({len(message)} chars)")
            return True
        else:
            log(f"❌ Telegram error {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        log(f"❌ Telegram exception: {e}")
        return False


# ─── GitHub API ──────────────────────────────────────────────────────────────

def _gh_headers() -> dict:
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def check_workflow_status() -> tuple[str, str, str]:
    """
    Check if today's GitHub Actions workflow run succeeded.
    Returns: (status, run_url, detail)
      status: 'success' | 'failure' | 'running' | 'no_run' | 'error'
    """
    today = cairo_today_str()
    headers = _gh_headers()

    url = (
        f"https://api.github.com/repos/{REPO}/actions/runs"
        f"?created={today}&per_page=10"
    )
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return ("error", "", f"GitHub API {resp.status_code}: {resp.text[:100]}")

        runs = resp.json().get("workflow_runs", [])
        if not runs:
            return ("no_run", "", "No workflow runs found for today")

        # Filter for the daily report workflow
        daily_runs = [
            r for r in runs
            if r.get("name") == WORKFLOW_NAME
            or r.get("path", "").endswith(WORKFLOW_FILE)
        ]
        if not daily_runs:
            return ("no_run", "", "No daily report runs found today")

        latest = daily_runs[0]
        gh_status = latest.get("status", "unknown")
        conclusion = latest.get("conclusion")
        run_url = latest.get("html_url", "")

        if gh_status in ("in_progress", "queued", "waiting"):
            return ("running", run_url, f"Workflow is {gh_status}")

        if conclusion == "success":
            return ("success", run_url, "Workflow completed successfully")
        elif conclusion == "failure":
            return ("failure", run_url, "Workflow failed")
        elif conclusion == "cancelled":
            return ("failure", run_url, "Workflow was cancelled")
        else:
            return ("unknown", run_url, f"status={gh_status} conclusion={conclusion}")

    except Exception as e:
        return ("error", "", f"Exception: {e}")


def trigger_workflow() -> bool:
    """Trigger workflow_dispatch for the daily report. Returns True on success."""
    headers = _gh_headers()
    url = (
        f"https://api.github.com/repos/{REPO}/actions/workflows/"
        f"{WORKFLOW_FILE}/dispatches"
    )
    try:
        resp = requests.post(
            url, headers=headers, json={"ref": "main"}, timeout=15
        )
        if resp.status_code == 204:
            log("✅ workflow_dispatch triggered")
            return True
        else:
            log(f"❌ workflow_dispatch failed: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        log(f"❌ workflow_dispatch exception: {e}")
        return False


# ─── JSON Processing ─────────────────────────────────────────────────────────

def clone_and_read_json() -> dict | None:
    """Clone repo and read today's recommendations JSON. Returns data or None."""
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             f"https://github.com/{REPO}.git", CLONE_DIR],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        log(f"❌ Clone failed: {e}")
        return None

    today = cairo_today_str()
    json_path = os.path.join(
        CLONE_DIR, "data", f"recommendations_{today}.json"
    )

    if not os.path.exists(json_path):
        log(f"⚠️ No JSON file for {today}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    recs = data.get("recommendations", [])
    log(f"✅ Found JSON: {len(recs)} recommendations, "
        f"{len(data.get('current_prices', {}))} prices")
    return data


# ─── Main Monitoring Logic ───────────────────────────────────────────────────

def run_monitor() -> None:
    """Full monitoring run — check, alert, retry, re-check."""
    today = cairo_today_str()
    log(f"=== EGX Monitor — {today} {cairo_now().strftime('%H:%M')} Cairo ===")

    # Step 1: Check workflow status
    status, run_url, detail = check_workflow_status()
    log(f"Workflow: {status} — {detail}")

    # If still running, wait and re-check
    if status == "running":
        log(f"Workflow still running. Waiting {RUNNING_WAIT_SECONDS // 60} min...")
        time.sleep(RUNNING_WAIT_SECONDS)
        status, run_url, detail = check_workflow_status()
        log(f"After wait: {status} — {detail}")

    # Step 2: If succeeded, process JSON
    if status == "success":
        log("✅ Workflow succeeded — processing recommendations JSON")
        data = clone_and_read_json()
        if data:
            # Output JSON for the agent (me) to save to entities
            print("\n=== JSON_DATA_START ===")
            print(json.dumps(data, ensure_ascii=False))
            print("=== JSON_DATA_END ===")
        else:
            log("⚠️ Workflow succeeded but no JSON file found — may need manual check")
        return

    # Step 3: Failed / no run — send alert + trigger retry
    alert_msg = (
        f"⚠️ تنبيه EGX Bot — {today}\n\n"
        f"تقرير اليوم فشل أو لم يتم تشغيله.\n"
        f"السبب: {detail}\n"
    )
    if run_url:
        alert_msg += f"التفاصيل: {run_url}\n"
    alert_msg += "\nجاري إعادة التشغيل تلقائياً..."

    send_telegram(alert_msg)

    # Trigger retry
    triggered = trigger_workflow()

    if not triggered:
        urgent_msg = (
            f"❌ تنبيه عاجل EGX Bot — {today}\n\n"
            f"فشل التقرير وفشل تشغيل إعادة المحاولة.\n"
            f"محتاج تدخل يدوي:\n"
            f"• تحقق من GitHub Token\n"
            f"• تحقق من حالة الـ repository\n"
            f"• تحقق من Telegram Bot Token\n\n"
            f"GitHub Actions:\n"
            f"https://github.com/{REPO}/actions"
        )
        send_telegram(urgent_msg)
        return

    # Step 4: Wait for retry to complete, then check again
    log(f"Retry triggered. Waiting {RETRY_WAIT_SECONDS // 60} min for completion...")
    time.sleep(RETRY_WAIT_SECONDS)

    status2, run_url2, detail2 = check_workflow_status()
    log(f"Retry result: {status2} — {detail2}")

    if status2 == "success":
        success_msg = (
            f"✅ EGX Bot — {today}\n\n"
            f"تمت إعادة التشغيل بنجاح!\n"
            f"التقرير وصل على Telegram."
        )
        send_telegram(success_msg)

        # Process JSON from the successful retry
        data = clone_and_read_json()
        if data:
            print("\n=== JSON_DATA_START ===")
            print(json.dumps(data, ensure_ascii=False))
            print("=== JSON_DATA_END ===")
        return

    # Retry also failed — send final alert
    if status2 == "running":
        # Still running after 20 min — give it more time
        log("Retry still running. Waiting 10 more minutes...")
        time.sleep(600)
        status2, run_url2, detail2 = check_workflow_status()
        log(f"Final check: {status2} — {detail2}")

    if status2 == "success":
        send_telegram(
            f"✅ EGX Bot — {today}\n\n"
            f"تمت إعادة التشغيل بنجاح (بعد انتظار إضافي)!\n"
            f"التقرير وصل على Telegram."
        )
        data = clone_and_read_json()
        if data:
            print("\n=== JSON_DATA_START ===")
            print(json.dumps(data, ensure_ascii=False))
            print("=== JSON_DATA_END ===")
        return

    # Final failure
    final_msg = (
        f"❌ تنبيه عاجل EGX Bot — {today}\n\n"
        f"فشل التقرير وفشلت إعادة التشغيل أيضاً.\n"
        f"السبب النهائي: {detail2}\n"
    )
    if run_url2:
        final_msg += f"التفاصيل: {run_url2}\n"
    final_msg += (
        f"\n⚠️ محتاج تدخل يدوي.\n"
        f"تحقق من GitHub Actions:\n"
        f"https://github.com/{REPO}/actions"
    )
    send_telegram(final_msg)
    log("❌ All attempts failed — final alert sent")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    if "--test-alert" in sys.argv:
        # Send a test alert to verify Telegram delivery
        today = cairo_today_str()
        msg = (
            f"🧪 اختبار تنبيه EGX Bot — {today}\n\n"
            f"ده رسالة اختبار للتأكد إن نظام التنبيه شغال.\n"
            f"لو وصلتك دي الرسالة، يبقى الـ monitoring system شغال ✅"
        )
        ok = send_telegram(msg)
        sys.exit(0 if ok else 1)

    if "--check-only" in sys.argv:
        # Check status only — no actions
        status, run_url, detail = check_workflow_status()
        print(f"Status: {status}")
        print(f"Detail: {detail}")
        if run_url:
            print(f"URL: {run_url}")
        sys.exit(0)

    # Full monitoring run
    if not GITHUB_TOKEN:
        log("❌ GITHUB_TOKEN not set")
        sys.exit(1)
    if not TELEGRAM_BOT_TOKEN:
        log("❌ TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    run_monitor()


if __name__ == "__main__":
    main()
