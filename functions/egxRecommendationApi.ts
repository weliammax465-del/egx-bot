/**
 * egxRecommendationApi.ts
 * -----------------------
 * Backend function for recommendation tracking and performance evaluation.
 *
 * Actions:
 *   - save:     Save today's recommendations (called by bot after daily report)
 *   - evaluate: Evaluate past recommendations with current prices (called by bot)
 *   - stats:    Return win rate, avg profit/loss, total recommendations (for dashboard)
 *   - history:  Return paginated recommendation history (for dashboard)
 *
 * Evaluation criteria:
 *   Buy:  pnl = (current - recommended) / recommended * 100
 *   Sell: pnl = (recommended - current) / recommended * 100
 *   pnl > 3% = win, pnl < -3% = loss, between = neutral
 */

import { createClientFromRequest } from "npm:@base44/sdk@0.8.31";

const WIN_THRESHOLD = 3.0;
const LOSS_THRESHOLD = -3.0;

Deno.serve(async (req: Request) => {
  const base44 = createClientFromRequest(req);

  try {
    const body = await req.json();
    const action = body.action;

    switch (action) {
      case "save":
        return await saveRecommendations(base44, body);
      case "evaluate":
        return await evaluateRecommendations(base44, body);
      case "stats":
        return await getStats(base44, body);
      case "history":
        return await getHistory(base44, body);
      default:
        return json({ error: `Unknown action: ${action}` }, 400);
    }
  } catch (error) {
    const msg = error?.message || (typeof error === "string" ? error : JSON.stringify(error));
    return json({ error: msg }, 500);
  }
});

function json(data: any, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// ─── Save Recommendations ─────────────────────────────────────────────────

async function saveRecommendations(base44: any, body: any) {
  const { recommendations, report_date, report_id } = body;

  if (!recommendations || !Array.isArray(recommendations) || recommendations.length === 0) {
    return json({ error: "recommendations array is required" }, 400);
  }
  if (!report_date) {
    return json({ error: "report_date is required" }, 400);
  }

  const svc = base44.asServiceRole;

  // Check duplicates
  const existing = await svc.entities.RecommendationHistory.list({
    filter: { recommendation_date: report_date },
    limit: 1,
  });

  if (existing && existing.length > 0) {
    return json({ message: "Recommendations for this date already exist", count: existing.length });
  }

  // Create records one by one (create may not accept arrays)
  let savedCount = 0;
  const errors: string[] = [];

  for (const rec of recommendations) {
    try {
      await svc.entities.RecommendationHistory.create({
        recommendation_date: report_date,
        ticker: rec.ticker,
        score_at_recommendation: rec.score,
        price_at_recommendation: rec.price,
        recommendation_type: rec.type,
        status: "pending",
        report_id: report_id || "",
      });
      savedCount++;
    } catch (e: any) {
      errors.push(`${rec.ticker}: ${e?.message || "unknown"}`);
    }
  }

  return json({
    success: true,
    saved: savedCount,
    errors: errors.length > 0 ? errors.slice(0, 5) : undefined,
    date: report_date,
  });
}

// ─── Evaluate Past Recommendations ─────────────────────────────────────────

async function evaluateRecommendations(base44: any, body: any) {
  const { current_prices, today } = body;

  if (!current_prices || typeof current_prices !== "object") {
    return json({ error: "current_prices object is required" }, 400);
  }
  if (!today) {
    return json({ error: "today date is required" }, 400);
  }

  const svc = base44.asServiceRole;
  const todayDate = new Date(today);
  const evaluated: string[] = [];
  let eval7d = 0;
  let eval30d = 0;

  // Fetch all records and filter client-side (SDK doesn't support 'in' operator)
  const allRecs = await svc.entities.RecommendationHistory.list({ limit: 500 });
  const pending = allRecs.filter((r: any) => r.status === "pending" || r.status === "evaluated_7d");

  for (const rec of pending) {
    const recDate = new Date(rec.recommendation_date);
    const daysSince = Math.floor((todayDate.getTime() - recDate.getTime()) / (1000 * 60 * 60 * 24));
    const currentPrice = current_prices[rec.ticker];
    if (currentPrice === undefined || currentPrice === null) continue;

    const updateData: any = {};

    if (daysSince >= 7 && rec.result_7d == null) {
      const pnl = calcPnl(rec.recommendation_type, rec.price_at_recommendation, currentPrice);
      updateData.price_7d = currentPrice;
      updateData.pnl_pct_7d = Math.round(pnl * 100) / 100;
      updateData.result_7d = classify(pnl);
      updateData.eval_date_7d = today;
      updateData.status = "evaluated_7d";
      eval7d++;
      evaluated.push(`${rec.ticker}(7d:${updateData.result_7d})`);
    }

    if (daysSince >= 30 && rec.result_30d == null) {
      const pnl = calcPnl(rec.recommendation_type, rec.price_at_recommendation, currentPrice);
      updateData.price_30d = currentPrice;
      updateData.pnl_pct_30d = Math.round(pnl * 100) / 100;
      updateData.result_30d = classify(pnl);
      updateData.eval_date_30d = today;
      updateData.status = "final";
      eval30d++;
      evaluated.push(`${rec.ticker}(30d:${updateData.result_30d})`);
    }

    if (Object.keys(updateData).length > 0) {
      try {
        await svc.entities.RecommendationHistory.update(rec.id, updateData);
      } catch (e: any) {
        // log but continue
      }
    }
  }

  return json({
    success: true,
    evaluated_7d: eval7d,
    evaluated_30d: eval30d,
    total_evaluated: evaluated.length,
    details: evaluated.slice(0, 20),
  });
}

// ─── Performance Stats ─────────────────────────────────────────────────────

async function getStats(base44: any, _body: any) {
  const svc = base44.asServiceRole;
  const allRecs = await svc.entities.RecommendationHistory.list({ limit: 500 });
  const evaluated = allRecs.filter((r: any) => r.status === "evaluated_7d" || r.status === "final");

  const now = new Date();
  const d30 = new Date(now.getTime() - 30 * 86400000);
  const d90 = new Date(now.getTime() - 90 * 86400000);

  const last30 = evaluated.filter((r: any) => new Date(r.recommendation_date) >= d30);
  const last90 = evaluated.filter((r: any) => new Date(r.recommendation_date) >= d90);

  const byType: any = {};
  for (const rec of evaluated) {
    const t = rec.recommendation_type;
    if (!byType[t]) byType[t] = { total: 0, wins: 0, losses: 0, neutral: 0, pnl_sum: 0 };
    byType[t].total++;
    const result = rec.result_30d || rec.result_7d;
    const pnl = rec.pnl_pct_30d ?? rec.pnl_pct_7d;
    if (result === "win") byType[t].wins++;
    else if (result === "loss") byType[t].losses++;
    else if (result === "neutral") byType[t].neutral++;
    if (pnl != null) byType[t].pnl_sum += pnl;
  }
  for (const t of Object.keys(byType)) {
    const ev = byType[t].wins + byType[t].losses + byType[t].neutral;
    byType[t].avg_pnl = ev > 0 ? Math.round((byType[t].pnl_sum / ev) * 100) / 100 : 0;
    delete byType[t].pnl_sum;
  }

  return json({
    total_recommendations: evaluated.length,
    last_30d: winStats(last30, "30d"),
    last_90d: winStats(last90, "30d"),
    last_7d: winStats(last30, "7d"),
    by_type: byType,
  });
}

// ─── History ───────────────────────────────────────────────────────────────

async function getHistory(base44: any, body: any) {
  const svc = base44.asServiceRole;
  const limit = body.limit || 50;
  const skip = body.skip || 0;
  const filter: any = {};
  if (body.ticker) filter.ticker = body.ticker;
  if (body.recommendation_type) filter.recommendation_type = body.recommendation_type;

  const records = await svc.entities.RecommendationHistory.list({
    filter: Object.keys(filter).length > 0 ? filter : undefined,
    limit,
    skip,
    sort: "-recommendation_date",
  });

  return json({ records, count: records.length });
}

// ─── Helpers ───────────────────────────────────────────────────────────────

function calcPnl(type: string, recommended: number, current: number): number {
  if (recommended <= 0) return 0;
  if (type === "Sell") return ((recommended - current) / recommended) * 100;
  return ((current - recommended) / recommended) * 100;
}

function classify(pnl: number): "win" | "loss" | "neutral" {
  if (pnl > WIN_THRESHOLD) return "win";
  if (pnl < LOSS_THRESHOLD) return "loss";
  return "neutral";
}

function winStats(recs: any[], period: string): any {
  const rf = `result_${period}`;
  const pf = `pnl_pct_${period}`;
  const evaluated = recs.filter((r) => r[rf] != null);
  const wins = evaluated.filter((r) => r[rf] === "win");
  const losses = evaluated.filter((r) => r[rf] === "loss");
  const neutral = evaluated.filter((r) => r[rf] === "neutral");

  return {
    total: recs.length,
    evaluated: evaluated.length,
    wins: wins.length,
    losses: losses.length,
    neutral: neutral.length,
    win_rate: evaluated.length > 0 ? Math.round((wins.length / evaluated.length) * 10000) / 100 : 0,
    avg_profit_pct: wins.length > 0 ? Math.round((wins.reduce((s, r) => s + (r[pf] || 0), 0) / wins.length) * 100) / 100 : 0,
    avg_loss_pct: losses.length > 0 ? Math.round((losses.reduce((s, r) => s + (r[pf] || 0), 0) / losses.length) * 100) / 100 : 0,
  };
}
