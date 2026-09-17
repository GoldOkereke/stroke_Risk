import { useEffect, useMemo, useState } from "react";
import { VoiceVisualizer, useVoiceVisualizer } from "react-voice-visualizer-react19";

import "./VoiceTest.css";

type VoiceTestProps = {
  prompt?: string;
  onAnalyze?: (audioFile: File) => void;
};

const averageLevel = (data: Uint8Array | null) => {
  if (!data || data.length === 0) return 0;
  const sum = data.reduce((acc, v) => acc + v, 0);
  return sum / data.length / 255;
};

export default function VoiceTest({ prompt, onAnalyze }: VoiceTestProps) {
  const controls = useVoiceVisualizer();
  const [status, setStatus] = useState("Ready to record.");

  const level = useMemo(() => averageLevel(controls.audioData), [controls.audioData]);

  useEffect(() => {
    if (controls.error) {
      setStatus(controls.error.message);
    }
  }, [controls.error]);

  useEffect(() => {
    if (!controls.recordedBlob) {
      return;
    }

    const file = new File([controls.recordedBlob], "voice-recording.webm", {
      type: controls.recordedBlob.type || "audio/webm",
    });
    onAnalyze?.(file);
    setStatus("Voice sample captured.");
  }, [controls.recordedBlob, onAnalyze]);

  const handleRecordToggle = () => {
    if (controls.isRecordingInProgress) {
      controls.stopRecording();
      setStatus("Stopping recording...");
      return;
    }

    setStatus("Requesting microphone access...");
    controls.startRecording();
  };

  return (
    <section className="voice-test">
      {prompt ? <p className="voice-prompt">{prompt}</p> : null}

      <VoiceVisualizer
        controls={controls}
        height={180}
        backgroundColor="#ffffff"
        mainBarColor="var(--alert-normal)"
        secondaryBarColor="var(--alert-advisory)"
        isControlPanelShown={false}
        isDownloadAudioButtonShown={false}
        isDefaultUIShown={false}
        onlyRecording={false}
      />

      <div className="voice-status" aria-live="polite">
        {status}
        {controls.isProcessingStartRecording ? " Waiting for microphone permission..." : null}
      </div>

      <div className="voice-controls">
        <button type="button" onClick={handleRecordToggle} disabled={controls.isProcessingStartRecording}>
          {controls.isRecordingInProgress ? "Stop Recording" : "Record"}
        </button>
        <button type="button" onClick={() => controls.clearCanvas()} disabled={controls.isRecordingInProgress}>
          Clear
        </button>
      </div>

      <div className="voice-meter">
        <span>Recording level</span>
        <div className="meter-track">
          <div className="meter-fill" style={{ width: `${level * 100}%` }} />
        </div>
      </div>
    </section>
  );
}
