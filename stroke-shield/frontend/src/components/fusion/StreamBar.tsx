import { memo } from "react";

import "./StreamBar.css";

type StreamBarProps = {
  name: string;
  score: number;
  layout?: "stacked" | "inline";
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const scoreToClass = (score: number) => {
  if (score >= 0.75) return "bar-fill critical";
  if (score >= 0.35) return "bar-fill advisory";
  return "bar-fill normal";
};

function StreamBar({ name, score, layout = "stacked" }: StreamBarProps) {
  const normalized = clamp(score, 0, 1);
  const fillClass = scoreToClass(normalized);
  const percent = Math.round(normalized * 100);

  return (
    <div className={`stream-bar ${layout}`}>
      <div className="stream-bar-labels">
        <span className="stream-name">{name}</span>
        <span className="stream-score">{percent}%</span>
      </div>
      <div className="stream-bar-track">
        <div className={fillClass} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

export default memo(StreamBar);
