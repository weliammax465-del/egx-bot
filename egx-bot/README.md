# 🇪🇬 EGX Daily Market Bot

A free, open-source Telegram bot that sends a daily Arabic summary of the Egyptian Stock Exchange (EGX 30) using:

- 🐍 **Python** — clean, minimal code
- 🤖 **Telegram Bot API** — free delivery
- 🧠 **Google Gemini API** (free tier) — Arabic AI summaries
- ⏰ **GitHub Actions** — free scheduled daily runs
- 📊 **Investing.com** — public EGX market data

> ⚠️ This bot is for **informational purposes only**. It does not provide financial advice or guarantee profits.

---

## 📁 Project Structure

```
egx-bot/
├── bot.py               # Main bot + scheduled sender
├── fetch_egx.py         # EGX market data scraper
├── ai_report.py         # Gemini AI Arabic report generator
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .github/
    └── workflows/
        └── daily.yml    # GitHub Actions schedule
```

---

## 🚀 Setup Guide (Mobile-Friendly)

### Step 1 — Create Your Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Follow the prompts — choose a name (e.g. `EGX Daily`) and a username (e.g. `EgxDailyBot`)
4. Copy the **bot token** (looks like `123456789:ABCdef...`) — you'll need it later

### Step 2 — Get Your Chat ID

1. Start a chat with your new bot (search its username, press Start)
2. Send any message to it
3. Open this URL in your browser (replace `YOUR_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
4. Find `"chat": {"id": XXXXXXX}` — that number is your **Chat ID**
5. Save it

> 💡 To send to a **channel** instead: add the bot as an admin to the channel, then use the channel username prefixed with `-100` as the Chat ID (e.g. `-1001234567890`).

### Step 3 — Get a Free Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key — it starts with `AIza...`

> 🆓 The free tier includes **1,500 requests/day** with `gemini-1.5-flash` — more than enough for a daily bot.

### Step 4 — Fork the Repository

1. Go to this repository on GitHub
2. Click **Fork** (top right) to copy it to your account

### Step 5 — Add GitHub Actions Secrets

1. In your forked repo, go to **Settings → Secrets and variables → Actions**
2. Click **"New repository secret"** and add each of the following:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat ID from Step 2 |
| `GEMINI_API_KEY` | Your Gemini key from Step 3 |

### Step 6 — Enable GitHub Actions

1. Go to the **Actions** tab in your forked repo
2. If prompted, click **"I understand my workflows, enable them"**
3. That's it — the bot will now run automatically every day ✅

---

## ⏰ Schedule

The bot runs at **10:30 AM Cairo time** on **Sunday through Thursday** (Egyptian trading days).

The cron expression in `daily.yml`:
```yaml
- cron: "30 8 * * 0-4"   # 08:30 UTC = 10:30 Cairo (UTC+2)
```

> If you're in summer time (UTC+3), change to `"30 7 * * 0-4"`.

---

## 🧪 Manual Trigger

To send a report immediately (without waiting for the schedule):

1. Go to **Actions → EGX Daily Market Report**
2. Click **"Run workflow"**
3. Click the green **"Run workflow"** button

---

## 💬 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/report` | Fetch and show today's report on demand |

To run the bot in interactive (polling) mode locally:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/egx-bot.git
cd egx-bot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TELEGRAM_BOT_TOKEN=your_token_here
export GEMINI_API_KEY=your_key_here

# Run the bot
python bot.py
```

---

## 🔒 Privacy & Security

- **No sensitive data is stored** anywhere in the code or repository
- All secrets are stored exclusively in GitHub Actions Secrets (encrypted)
- The bot never logs user messages or personal data
- Market data is fetched from public sources only

---

## 📦 Dependencies

| Package | Purpose | License |
|---|---|---|
| `python-telegram-bot` | Telegram Bot API wrapper | LGPL-3.0 |
| `google-generativeai` | Gemini AI SDK | Apache-2.0 |
| `requests` | HTTP requests | Apache-2.0 |
| `beautifulsoup4` | HTML scraping | MIT |
| `pytz` | Timezone support | MIT |
| `lxml` | Fast HTML parser | BSD |

---

## ⚠️ Disclaimer

This bot provides **market information for educational and informational purposes only**.

- It does **not** provide personalized financial advice
- It does **not** guarantee profits or predict market movements
- Always consult a licensed financial advisor before making investment decisions
- Past market performance does not guarantee future results

---

## 🛠️ Troubleshooting

**Bot not sending messages?**
- Check that all 3 secrets are correctly set in GitHub Actions Secrets
- Run the workflow manually from the Actions tab to see logs

**Gemini API error?**
- Verify your API key is correct at [aistudio.google.com](https://aistudio.google.com)
- Check you haven't exceeded the free tier limit (1,500 req/day)

**Data shows N/A?**
- Investing.com may have changed their HTML structure
- Open an issue and the scraper will be updated

---

Made with ❤️ for Egyptian investors.
