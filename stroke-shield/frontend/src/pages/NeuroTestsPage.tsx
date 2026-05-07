import { useEffect, useState } from "react";

import CognitiveTest from "../components/neuro/CognitiveTest";
import ReactionTimeTest from "../components/neuro/ReactionTimeTest";
import TapSpeedTest from "../components/neuro/TapSpeedTest";
import { getCombinedScore, sendCognitiveTest, sendReactionTime, sendTapSpeed } from "../api/neuroTestsApi";

import "./NeuroTestsPage.css";

export default function NeuroTestsPage() {
  const [reactionTimes, setReactionTimes] = useState<number[]>([]);
  const [maxSpan, setMaxSpan] = useState(0);
  const [tapRate, setTapRate] = useState(0);
  const [combinedScore, setCombinedScore] = useState(0);
  const [status, setStatus] = useState("Run the tests to sync with the backend.");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const refreshCombinedScore = async () => {
    try {
      const result = await getCombinedScore();
      setCombinedScore(typeof result?.score === "number" ? result.score : 0);
      setStatus("Backend combined score updated.");
    } catch {
      setStatus("Backend combined score unavailable.");
    }
  };

  useEffect(() => {
    void refreshCombinedScore();
  }, []);

  const submitReactionTime = async (results: number[]) => {
    setReactionTimes(results);
    setIsSubmitting(true);
    setStatus("Saving reaction-time results...");

    try {
      await sendReactionTime(results);
      await refreshCombinedScore();
    } catch {
      setStatus("Unable to save reaction-time results.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const submitCognitiveTest = async (span: number) => {
    setMaxSpan(span);
    setIsSubmitting(true);
    setStatus("Saving cognitive test...");

    try {
      await sendCognitiveTest({ max_span: span });
      await refreshCombinedScore();
    } catch {
      setStatus("Unable to save cognitive test.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const submitTapSpeed = async (taps: number[], rate: number) => {
    setTapRate(rate);
    setIsSubmitting(true);
    setStatus("Saving tap-speed test...");

    try {
      await sendTapSpeed(taps);
      await refreshCombinedScore();
    } catch {
      setStatus("Unable to save tap-speed test.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRunAll = async () => {
    setReactionTimes([]);
    setMaxSpan(0);
    setTapRate(0);
    setStatus("Tests reset. Run each card again to submit fresh data.");
    await refreshCombinedScore();
  };

  return (
    <div className="page neuro-tests-page">
      <div className="tests-grid">
        <ReactionTimeTest onComplete={submitReactionTime} />
        <CognitiveTest onComplete={submitCognitiveTest} />
        <TapSpeedTest onComplete={submitTapSpeed} />
      </div>

      <div className="combined-score">
        <h3>Combined Score</h3>
        <p>{(combinedScore * 100).toFixed(1)}%</p>
        <div className="combined-meta">
          <span>{status}</span>
          {isSubmitting ? <strong>Syncing...</strong> : null}
        </div>
        <div className="combined-buttons">
          <button type="button" onClick={() => void handleRunAll()}>
            Run All
          </button>
        </div>
      </div>

      <div className="neuro-summary">
        <div>
          <span>Reaction trials</span>
          <strong>{reactionTimes.length}</strong>
        </div>
        <div>
          <span>Max span</span>
          <strong>{maxSpan}</strong>
        </div>
        <div>
          <span>Tap rate</span>
          <strong>{tapRate ? `${tapRate.toFixed(2)} taps/s` : "0.00 taps/s"}</strong>
        </div>
      </div>
    </div>
  );
}
