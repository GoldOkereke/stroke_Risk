import { useCallback, useMemo, useRef, useState } from "react";
import Webcam from "react-webcam";

import "./FaceTest.css";

type FaceTestProps = {
  baselineTarget?: number;
  onCapture?: (imageFile: File) => void;
};

export default function FaceTest({ baselineTarget = 3, onCapture }: FaceTestProps) {
  const webcamRef = useRef<Webcam | null>(null);
  const [captures, setCaptures] = useState<string[]>([]);
  const [feedback, setFeedback] = useState("Align your face inside the outline.");

  const dataUrlToFile = (dataUrl: string, filename: string) => {
    const [meta, base64] = dataUrl.split(",");
    const mimeMatch = meta.match(/data:(.*?);base64/);
    const mimeType = mimeMatch?.[1] ?? "image/jpeg";
    const binary = atob(base64 ?? "");
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }

    return new File([bytes], filename, { type: mimeType });
  };

  const baselineProgress = useMemo(() => {
    return Math.min(1, captures.length / baselineTarget);
  }, [captures.length, baselineTarget]);

  const handleCapture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) {
      setFeedback("Camera not ready. Please allow access.");
      return;
    }
    const captureIndex = captures.length + 1;
    setCaptures((prev) => [...prev, imageSrc]);
    setFeedback("Captured. Adjust angle and capture again.");
    onCapture?.(dataUrlToFile(imageSrc, `face-capture-${captureIndex}.jpg`));
  }, [captures.length, onCapture]);

  return (
    <section className="face-test">
      <div className="face-frame">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          className="face-webcam"
          onUserMedia={() => setFeedback("Face detected. Hold still.")}
          onUserMediaError={() => setFeedback("Camera access denied.")}
        />
        <div className="face-outline" />
        <div className="face-feedback">{feedback}</div>
      </div>

      <div className="face-actions">
        <button type="button" onClick={handleCapture}>
          Capture
        </button>
        <div className="face-progress">
          <span>Baseline progress</span>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${baselineProgress * 100}%` }} />
          </div>
          <small>
            {captures.length}/{baselineTarget}
          </small>
        </div>
      </div>

      {captures.length > 0 ? (
        <div className="face-preview">
          {captures.slice(-3).map((src, index) => (
            <img key={`${src}-${index}`} src={src} alt="Face capture preview" />
          ))}
        </div>
      ) : null}
    </section>
  );
}
