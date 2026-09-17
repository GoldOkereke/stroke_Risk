import type { AlertTier } from "../../types";
import { ALERT_TIER_COLORS } from "../../utils/constants";
import "./AlertPanel.css";

type AlertPanelProps = {
  alertTier?: AlertTier;
  recommendation: string;
  contributingFactors?: Record<string, number>;
};

const tierIcon = (tier: AlertTier) => {
  if (tier === "CRITICAL") return "⚠";
  if (tier === "ADVISORY") return "!";
  return "✓";
};

const tierLabel = (tier: AlertTier) => {
  if (tier === "CRITICAL") return "Critical";
  if (tier === "ADVISORY") return "Advisory";
  return "Normal";
};

const factorLabel = (key: string) =>
  key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export default function AlertPanel({
  alertTier,
  recommendation,
  contributingFactors,
}: AlertPanelProps) {
  const safeTier: AlertTier = alertTier ?? "NORMAL";
  const safeFactors = contributingFactors ?? {};
  const backgroundColor = ALERT_TIER_COLORS[safeTier];

  return (
    <section className={`alert-panel ${safeTier.toLowerCase()}`}>
      <header className="alert-header" style={{ borderColor: backgroundColor }}>
        <span className="alert-icon" aria-hidden="true">
          {tierIcon(safeTier)}
        </span>
        <div>
          <p className="alert-title">{tierLabel(safeTier)} Alert</p>
          <p className="alert-subtitle">Overall risk status</p>
        </div>
      </header>

      <div className="alert-body" style={{ backgroundColor }}>
        <p className="alert-recommendation">{recommendation}</p>
        <ul className="alert-factors">
          {Object.entries(safeFactors).map(([key, value]) => (
            <li key={key}>
              {factorLabel(key)}: {(value * 100).toFixed(1)}%
            </li>
          ))}
        </ul>
      </div>

      <div className="alert-actions">
        <button type="button" className="alert-button">
          Emergency
        </button>
        <button type="button" className="alert-button secondary">
          Find Doctor
        </button>
        <button type="button" className="alert-button ghost">
          Export
        </button>
      </div>
    </section>
  );
}
