import { useMemo, useState } from "react";
import {
  Line,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import "./SignalVisualizer.css";

type SignalVisualizerProps = {
  raw: number[];
  filtered?: number[];
};

type Point = {
  index: number;
  raw: number;
  filtered?: number;
};

const buildData = (raw: number[], filtered?: number[]) =>
  raw.map((value, index) => ({
    index,
    raw: value,
    filtered: filtered?.[index],
  }));

export default function SignalVisualizer({ raw, filtered }: SignalVisualizerProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState(0);
  const [showFiltered, setShowFiltered] = useState(Boolean(filtered));

  const data = useMemo<Point[]>(() => buildData(raw, filtered), [raw, filtered]);

  const windowSize = Math.max(50, Math.floor(300 / zoom));
  const start = Math.min(Math.max(0, pan), Math.max(0, data.length - windowSize));
  const windowed = data.slice(start, start + windowSize);

  return (
    <div className="signal-visualizer">
      <div className="signal-controls">
        <label>
          Zoom
          <input
            type="range"
            min={1}
            max={5}
            step={0.25}
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
          />
        </label>
        <label>
          Pan
          <input
            type="range"
            min={0}
            max={Math.max(0, data.length - windowSize)}
            value={pan}
            onChange={(event) => setPan(Number(event.target.value))}
          />
        </label>
        {filtered ? (
          <label className="toggle">
            <input
              type="checkbox"
              checked={showFiltered}
              onChange={(event) => setShowFiltered(event.target.checked)}
            />
            Show filtered
          </label>
        ) : null}
      </div>

      <div className="signal-chart">
        <div className="scan-line" />
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={windowed}>
            <CartesianGrid stroke="rgba(0,0,0,0.08)" strokeDasharray="3 3" />
            <XAxis dataKey="index" tick={false} />
            <YAxis tick={false} domain={["auto", "auto"]} />
            <Tooltip formatter={(value) => (typeof value === "number" ? value.toFixed(3) : "")} />
            <Line type="monotone" dataKey="raw" stroke="#0f4c5c" dot={false} />
            {filtered && showFiltered ? (
              <Line type="monotone" dataKey="filtered" stroke="#ff3b5c" dot={false} />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
