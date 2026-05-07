export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const ALERT_TIER_COLORS = {
  NORMAL: "#00c896",
  ADVISORY: "#f5a623",
  CRITICAL: "#ff3b5c",
} as const;

export const CHART_COLORS = [
  "#0f4c5c",
  "#3d5a80",
  "#ee6c4d",
  "#98c1d9",
  "#e0fbfc",
];

export const DEFAULT_THRESHOLDS = {
  AFIB_DECISION: 0.5,
  UNCERTAIN_LOW: 0.35,
  UNCERTAIN_HIGH: 0.65,
  ADVISORY_RISK: 40,
  CRITICAL_RISK: 70,
} as const;
