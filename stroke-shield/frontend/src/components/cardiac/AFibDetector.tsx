import { useMemo } from "react";

import "./AFibDetector.css";

type AFibDetectorProps = {
  afibProbability: number;
  prediction: "AFIB" | "NORMAL";
  confidence: number;
  signalQuality: number;
  history?: number[];
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

export default function AFibDetector({
  afibProbability,
  prediction,
  confidence,
  signalQuality,
  history = [],
}: AFibDetectorProps) {
  const prob = clamp(afibProbability, 0, 1);
  const quality = clamp(signalQuality, 0, 1);

  const ringOffset = useMemo(() => 251 - 251 * prob, [prob]);

  const comparison = useMemo(() => {
    if (!history.length) return "No history yet";
    const last = history[history.length - 1];
    const delta = prob - last;
    const sign = delta >= 0 ? "+" : "-";
    return `Change vs last: ${sign}${Math.abs(delta * 100).toFixed(1)}%`;
  }, [history, prob]);

  return (
    <div className="afib-detector">
      <div className="afib-ring">
        <svg viewBox="0 0 100 100">
          <circle className="ring-track" cx="50" cy="50" r="40" />
          <circle
            className="ring-progress"
            cx="50"
            cy="50"
            r="40"
            style={{ strokeDashoffset: ringOffset }}
          />
        </svg>
        <div className="afib-value">
          <span className="afib-label">
            {prediction === "AFIB" ? "Afib Detected" : "Normal"}
          </span>
          <span className="afib-score">{formatPercent(prob)}</span>
        </div>
      </div>

      <div className="afib-metrics">
        <div className="metric">
          <span>Confidence</span>
          <strong>{formatPercent(confidence)}</strong>
        </div>
        <div className="metric">
          <span>Signal Quality</span>
          <strong>{formatPercent(quality)}</strong>
        </div>
      </div>

      <div className="afib-history">
        <p>{comparison}</p>
        {history.length ? (
          <div className="history-bars">
            {history.slice(-8).map((value, index) => (
              <span
                key={`${value}-${index}`}
                style={{ height: `${clamp(value, 0, 1) * 100}%` }}
              />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
