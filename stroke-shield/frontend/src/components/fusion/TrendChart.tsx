import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

import "./TrendChart.css";

type TrendPoint = {
  time: string;
  riskScore: number;
};

type TrendChartProps = {
  data: TrendPoint[];
};

const formatData = (data: TrendPoint[]) => {
  const trimmed = data.slice(-10);
  return trimmed.map((point, index) => ({
    index: index + 1,
    ...point,
  }));
};

export default function TrendChart({ data }: TrendChartProps) {
  const chartData = useMemo(() => formatData(data), [data]);

  return (
    <div className="trend-chart">
      <div className="trend-chart-header">
        <h3>Risk Trend</h3>
        <p>Last 10 tests</p>
      </div>
      <div className="trend-chart-body">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="var(--alert-normal)" />
                <stop offset="50%" stopColor="var(--alert-advisory)" />
                <stop offset="100%" stopColor="var(--alert-critical)" />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(0, 0, 0, 0.08)" vertical={false} />
            <XAxis dataKey="time" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(value) => {
                const numeric = typeof value === "number" ? value : Number(value);
                return [isNaN(numeric) ? "" : `${numeric.toFixed(1)}`, "Risk"];
              }}
              labelFormatter={(label) => `Test ${label}`}
            />
            <Line
              type="monotone"
              dataKey="riskScore"
              stroke="url(#riskGradient)"
              strokeWidth={3}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              fillOpacity={1}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
