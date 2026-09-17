import { useMemo } from "react";
import { motion } from "framer-motion";

import type { AlertTier } from "../../types";
import "./RiskGauge.css";

type RiskGaugeProps = {
  value: number;
  alertTier?: AlertTier;
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const describeColor = (value: number) => {
  if (value >= 75) return "var(--alert-critical)";
  if (value >= 35) return "var(--alert-advisory)";
  return "var(--alert-normal)";
};

export default function RiskGauge({ value, alertTier = "NORMAL" }: RiskGaugeProps) {
  const normalized = clamp(value, 0, 100);
  const arcLength = 260;
  const progress = (normalized / 100) * arcLength;

  const needleRotation = useMemo(() => -130 + (normalized / 100) * 260, [normalized]);
  const strokeColor = describeColor(normalized);

  return (
    <div className={`risk-gauge ${alertTier === "CRITICAL" ? "pulse" : ""}`}>
      <svg viewBox="0 0 260 160" className="risk-gauge-svg" role="img" aria-label="Risk gauge">
        <path
          className="risk-gauge-track"
          d="M20 140 A110 110 0 0 1 240 140"
        />
        <motion.path
          className="risk-gauge-fill"
          d="M20 140 A110 110 0 0 1 240 140"
          style={{ stroke: strokeColor, strokeDasharray: arcLength, strokeDashoffset: arcLength - progress }}
          initial={{ strokeDashoffset: arcLength }}
          animate={{ strokeDashoffset: arcLength - progress }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
        <motion.line
          className="risk-gauge-needle"
          x1="130"
          y1="140"
          x2="130"
          y2="40"
          style={{ stroke: strokeColor }}
          animate={{ rotate: needleRotation }}
          transform-origin="130px 140px"
          transition={{ type: "spring", stiffness: 120, damping: 14 }}
        />
        <circle className="risk-gauge-center" cx="130" cy="140" r="6" />
      </svg>
      <div className="risk-gauge-value">
        <span className="risk-gauge-number">{Math.round(normalized)}</span>
        <span className="risk-gauge-label">Risk Score</span>
      </div>
    </div>
  );
}
