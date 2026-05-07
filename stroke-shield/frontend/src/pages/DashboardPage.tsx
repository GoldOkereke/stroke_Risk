import { useEffect, useMemo, useState } from "react";

import AlertPanel from "../components/fusion/AlertPanel";
import RiskGauge from "../components/fusion/RiskGauge";
import StreamBar from "../components/fusion/StreamBar";
import TrendChart from "../components/fusion/TrendChart";
import { getHistory, getFusionScore } from "../api/fusionApi";
import type { FusionResult } from "../types";

const emptyResult: FusionResult = {
  riskScore: 0,
  alertTier: "NORMAL",
  confidence: 0,
  contributingFactors: {},
  recommendation: "No fusion data yet.",
};

export default function DashboardPage() {
  const [fusion, setFusion] = useState<FusionResult>(emptyResult);
  const [history, setHistory] = useState<number[]>([]);

  const trendData = useMemo(
    () =>
      history.slice(-10).map((value, index) => ({
        time: `T${index + 1}`,
        riskScore: value,
      })),
    [history],
  );

  const fusionFactors = fusion.contributingFactors ?? {};

  useEffect(() => {
    let isMounted = true;

    const loadFusion = async () => {
      try {
        const historyData = await getHistory();
        if (!isMounted) return;
        if (historyData.length) {
          setFusion(historyData[historyData.length - 1]);
          setHistory(historyData.map((item) => item.riskScore));
          return;
        }

        const score = await getFusionScore({});
        if (!isMounted) return;
        setFusion(score);
        setHistory((prev) => [...prev, score.riskScore]);
      } catch {
        // Keep existing state if API is unavailable.
      }
    };

    loadFusion();
    const interval = window.setInterval(loadFusion, 30000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  return (
    <div className="page dashboard-page">
      <AlertPanel
        alertTier={fusion.alertTier ?? "NORMAL"}
        recommendation={fusion.recommendation}
        contributingFactors={fusion.contributingFactors ?? {}}
      />

      <div className="dashboard-grid">
        <RiskGauge value={fusion.riskScore} alertTier={fusion.alertTier} />
        <TrendChart data={trendData} />
      </div>

      <div className="dashboard-streams">
        <StreamBar name="Cardiac CNN" score={fusionFactors.cardiacCnn ?? 0} />
        <StreamBar name="Cardiac IF" score={fusionFactors.cardiacIf ?? 0} />
        <StreamBar name="Facial" score={fusionFactors.facial ?? 0} />
        <StreamBar name="Dysarthria" score={fusionFactors.dysarthria ?? 0} />
        <StreamBar name="Parkinsons" score={fusionFactors.parkinsons ?? 0} />
        <StreamBar name="Neuro Tests" score={fusionFactors.neuroTests ?? 0} />
      </div>
    </div>
  );
}
