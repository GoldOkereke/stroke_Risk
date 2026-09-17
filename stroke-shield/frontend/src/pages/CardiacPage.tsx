import { useEffect, useMemo, useState } from "react";

import AFibDetector from "../components/cardiac/AFibDetector";
import PPGUploader from "../components/cardiac/PPGUploader";
import SignalVisualizer from "../components/cardiac/SignalVisualizer";
import { getRecords, loadMITBIH, simulatePPG } from "../api/signalApi";
import type { CardiacResult } from "../types";

const emptyResult: CardiacResult = {
  afibProbability: 0,
  prediction: "NORMAL",
  confidence: 0,
  signalQuality: 0,
};

const buildWaveform = (length: number) =>
  Array.from({ length }, (_, i) => Math.sin(i / 8) * 0.5 + Math.random() * 0.1);

export default function CardiacPage() {
  const [tab, setTab] = useState<"simulate" | "mitbih" | "upload">("simulate");
  const [records, setRecords] = useState<string[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<string>("");
  const [result, setResult] = useState<CardiacResult>(emptyResult);
  const [waveform, setWaveform] = useState<number[]>(buildWaveform(400));
  const [history, setHistory] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (tab !== "mitbih") return;
    getRecords()
      .then((data) => {
        setRecords(data);
        if (data.length) setSelectedRecord(data[0]);
      })
      .catch(() => setError("Unable to fetch record list."));
  }, [tab]);

  const handleSimulate = async (scenario: string) => {
    setError(null);
    try {
      const data = await simulatePPG(scenario);
      setResult(data);
      setWaveform(buildWaveform(400));
      setHistory((prev) => [...prev, data.afibProbability]);
    } catch {
      setError("Simulation failed.");
    }
  };

  const handleLoadRecord = async () => {
    if (!selectedRecord) return;
    setError(null);
    try {
      const data = await loadMITBIH(selectedRecord);
      setResult(data);
      setWaveform(buildWaveform(400));
      setHistory((prev) => [...prev, data.afibProbability]);
    } catch {
      setError("Record load failed.");
    }
  };

  const activeContent = useMemo(() => {
    if (tab === "simulate") {
      return (
        <div className="cardiac-actions">
          <button type="button" onClick={() => handleSimulate("normal")}>
            Simulate Normal
          </button>
          <button type="button" onClick={() => handleSimulate("afib_only")}>
            Simulate AFib
          </button>
          <button type="button" onClick={() => handleSimulate("pre_tia")}>
            Simulate Pre-TIA
          </button>
        </div>
      );
    }
    if (tab === "mitbih") {
      return (
        <div className="cardiac-actions">
          <select value={selectedRecord} onChange={(e) => setSelectedRecord(e.target.value)}>
            {records.map((record) => (
              <option key={record} value={record}>
                {record}
              </option>
            ))}
          </select>
          <button type="button" onClick={handleLoadRecord}>
            Load Record
          </button>
        </div>
      );
    }
    return (
      <div className="cardiac-upload">
        <PPGUploader
          onParsed={(signal) => setWaveform(signal.samples)}
          onUploaded={() => setHistory((prev) => [...prev, result.afibProbability])}
        />
      </div>
    );
  }, [tab, records, selectedRecord]);

  return (
    <div className="page cardiac-page">
      <div className="tabs">
        <button type="button" className={tab === "simulate" ? "active" : ""} onClick={() => setTab("simulate")}>
          Simulate
        </button>
        <button type="button" className={tab === "mitbih" ? "active" : ""} onClick={() => setTab("mitbih")}>
          MIT-BIH
        </button>
        <button type="button" className={tab === "upload" ? "active" : ""} onClick={() => setTab("upload")}>
          Upload CSV
        </button>
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      {activeContent}

      <SignalVisualizer raw={waveform} />
      <AFibDetector
        afibProbability={result.afibProbability}
        prediction={result.prediction}
        confidence={result.confidence}
        signalQuality={result.signalQuality}
        history={history}
      />
    </div>
  );
}
