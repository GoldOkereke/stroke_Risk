import { useEffect, useMemo, useState } from "react";

import StreamBar from "../components/fusion/StreamBar";
import SignalVisualizer from "../components/cardiac/SignalVisualizer";
import { startDemo } from "../api/demoApi";

const narratives = [
  "Baseline signals captured. Monitoring begins.",
  "Cardiac rhythm analysis running.",
  "Facial asymmetry and speech signals assessed.",
  "Fusion engine combining risk streams.",
  "Final risk score generated.",
];

const buildWaveform = (tick: number) =>
  Array.from({ length: 300 }, (_, i) => Math.sin((i + tick) / 10) * 0.6);

export default function DemoPage() {
  const [stage, setStage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!isPlaying) return;
    const interval = window.setInterval(() => {
      setStage((prev) => (prev + 1) % 5);
      setTick((prev) => prev + 1);
    }, 3000);
    return () => window.clearInterval(interval);
  }, [isPlaying]);

  const waveform = useMemo(() => buildWaveform(tick), [tick]);

  const streamValues = useMemo(() => {
    const base = stage / 4;
    return {
      cardiac: Math.min(1, base + 0.1),
      facial: Math.min(1, base + 0.05),
      voice: Math.min(1, base + 0.08),
      neuro: Math.min(1, base + 0.04),
    };
  }, [stage]);

  return (
    <div className="page demo-page">
      <div className="demo-header">
        <button type="button" onClick={() => startDemo()}>
          Simulate Pre-TIA
        </button>
        <div className="demo-stage">Stage {stage}</div>
        <label className="demo-toggle">
          Autoplay
          <input type="checkbox" checked={isPlaying} onChange={(e) => setIsPlaying(e.target.checked)} />
        </label>
      </div>

      <SignalVisualizer raw={waveform} />

      <div className="demo-streams">
        <StreamBar name="Cardiac CNN" score={streamValues.cardiac} layout="inline" />
        <StreamBar name="Facial" score={streamValues.facial} layout="inline" />
        <StreamBar name="Voice" score={streamValues.voice} layout="inline" />
        <StreamBar name="Neuro Tests" score={streamValues.neuro} layout="inline" />
      </div>

      <div className="demo-narrative">
        <p>{narratives[stage]}</p>
      </div>

      <div className="demo-controls">
        <button type="button" onClick={() => setStage((prev) => Math.max(0, prev - 1))}>
          Previous
        </button>
        <button type="button" onClick={() => setStage((prev) => (prev + 1) % 5)}>
          Next
        </button>
      </div>
    </div>
  );
}
