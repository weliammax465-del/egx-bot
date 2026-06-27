# 🇪🇬 EGX Daily Market Bot

A free, open-source Telegram bot that sends a daily Arabic summary of the Egyptian Stock Exchange (EGX 30) using:

- 🐍 **Python** — clean, minimal, production-minded code
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
├── fetch_egx.py         # EGX market data scraper (with retry + fallback)
├── ai_report.py         # Gemini AI Arabic report generator
├── requirements.txt     # Python dependencies
├── .gitignore           # Python gitignore
├── LICENSE              # MIT License
├── README.md            # This file
├── tests/
│   └── test_egx_bot.py  # Unit tests (34 tests, all mocked — no API keys needed)
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

> 💡 To send to a **channel** instead: add the bot as an admin to the channel, then use the channel ID prefixed with `-100` (e.g. `-1001234567890`).

### Step 3 — Get a Free Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key — it starts with `AIza...`

> 🆓 The free tier includes **1,500 requests/day** with `gemini-1.5-flash` — more than enough for a daily bot.

### Step 4 — Create a Public GitHub Repository

1. Open [github.com](https://github.com) on your phone browser
2. Tap **+** → **New repository**
3. Name it `egx-bot`
4. Set it to **Public** (required for free GitHub Actions)
5. Check **"Add a README"**
6. Tap **Create repository**

### Step 5 — Upload the Bot Files

1. Download the `egx-bot.zip` file and extract it on your phone
2. In your GitHub repo, tap **Add file → Upload files**
3. Upload all files from the extracted folder:
   - `bot.py`
   - `fetch_egx.py`
   - `ai_report.py`
   - `requirements.txt`
   - `.gitignore`
   - `LICENSE`
   - `README.md`
4. **Important:** Also upload the `.github` folder (create it if needed):
   - Tap **Add file → Create new file**
   - Type `.github/workflows/daily.yml` as the filename
   - Paste the content from the file
   - Commit
5. Upload the `tests/` folder the same way:
   - Tap **Add file → Create new file**
   - Type `tests/test_egx_bot.py` as the filename
   - Paste the content
   - Commit

### Step 6 — Add GitHub Actions Secrets

1. In your repo, go to **Settings → Secrets and variables → Actions**
2. Tap **"New repository secret"** and add each of the following:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat ID from Step 2 |
| `GEMINI_API_KEY` | Your Gemini key from Step 3 |

### Step 7 — Enable & Test GitHub Actions

1. Go to the **Actions** tab in your repo
2. If prompted, tap **"I understand my workflows, enable them"**
3. Select **EGX Daily Market Report**
4. Tap **"Run workflow"** → **"Run workflow"**
5. Watch the run — if all secrets are correct, you'll receive a message in Telegram! ✅

---

## ⏰ Schedule

The bot runs at **10:30 AM Cairo time** on **Sunday through Thursday** (EGX trading days).

Egypt uses **UTC+2 year-round** (DST was cancelled in 2024), so the cron is:
```yaml
- cron: "30 8 * * 0-4"   # 08:30 UTC = 10:30 Cairo
```

The bot also automatically skips Fridays and Saturdays (Egyptian weekend) even if the cron fires.

---

## 🧪 Tests

The project includes 34 unit tests covering:
- Market data parsing and error handling
- Markdown escaping for Telegram
- Arabic date formatting
- Gemini API retry and fallback logic
- Trading day detection
- Environment variable validation
- Telegram message length limits

```bash
# Run tests locally
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

All tests use mocks — no real API keys or network access required.

---

## 💬 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/report` | Fetch and show today's report on demand |

To run the bot in interactive (polling) mode locally:

```bash
# Clone your repo
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
- `.gitignore` prevents accidental commits of `.env` files
- GitHub Actions workflow uses minimal `permissions: contents: read`

---

## 🛡️ Production Features

- **Retry logic** — network requests retry 2x with exponential backoff
- **Graceful degradation** — if data source fails, bot sends a notification instead of crashing
- **Markdown escaping** — stock names with special characters don't break Telegram formatting
- **Message length safety** — messages are truncated to fit Telegram's 4096 char limit
- **Gemini safety settings** — configured to allow financial discussion without blocking
- **API timeout** — Gemini requests time out after 30 seconds
- **Non-trading day skip** — saves API quota on weekends
- **Error isolation** — errors in one component don't crash the whole flow

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
- Verify your Chat ID is correct (positive number for private chat, negative for channel)

**Gemini API error?**
- Verify your API key is correct at [aistudio.google.com](https://aistudio.google.com)
- Check you haven't exceeded the free tier limit (1,500 req/day)
- The bot will fall back to raw data if Gemini fails — check logs for details

**Data shows N/A?**
- Investing.com may have changed their HTML structure
- The bot will send a "market closed" notification instead of crashing
- Open an issue and the scraper will be updated

**GitHub Actions not running?**
- Make sure the repository is **Public** (free Actions for public repos)
- Check the `.github/workflows/daily.yml` file exists and is valid YAML
- Manually trigger from the Actions tab to see error logs

---

Made with ❤️ for Egyptian investors.
