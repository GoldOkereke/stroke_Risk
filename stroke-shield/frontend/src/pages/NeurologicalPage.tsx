import { useState } from "react";

import FaceAsymmetryMap from "../components/neurological/FaceAsymmetryMap";
import FaceTest from "../components/neurological/FaceTest";
import VoiceTest from "../components/neurological/VoiceTest";
import BaselineSetup from "../components/neurological/BaselineSetup";

export default function NeurologicalPage() {
  const [tab, setTab] = useState<"face" | "voice">("face");
  const [baselineComplete, setBaselineComplete] = useState(false);
  const [faceScore, setFaceScore] = useState(0.2);
  const [voiceScore, setVoiceScore] = useState(0.25);
  const [faceSample, setFaceSample] = useState<File | null>(null);
  const [voiceSample, setVoiceSample] = useState<File | null>(null);

  return (
    <div className="page neuro-page">
      {!baselineComplete ? (
        <BaselineSetup
          faceSample={faceSample}
          voiceSample={voiceSample}
          onComplete={() => setBaselineComplete(true)}
        />
      ) : null}

      <div className="tabs">
        <button type="button" className={tab === "face" ? "active" : ""} onClick={() => setTab("face")}>
          Face
        </button>
        <button type="button" className={tab === "voice" ? "active" : ""} onClick={() => setTab("voice")}>
          Voice
        </button>
      </div>

      {tab === "face" ? (
        <div className="neuro-section">
          <FaceTest
            onCapture={(imageFile) => {
              setFaceSample(imageFile);
              setFaceScore((prev) => Math.min(1, prev + 0.05));
            }}
          />
          <FaceAsymmetryMap leftScore={faceScore} rightScore={1 - faceScore} mouthDroop={faceScore} />
        </div>
      ) : null}

      {tab === "voice" ? (
        <div className="neuro-section">
          <VoiceTest
            onAnalyze={(audioFile) => {
              setVoiceSample(audioFile);
              setVoiceScore((prev) => Math.min(1, prev + 0.05));
            }}
          />
          <div className="result-card">
            <h4>Voice Result</h4>
            <p>Score: {(voiceScore * 100).toFixed(1)}%</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
