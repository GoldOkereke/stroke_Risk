import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";

import { uploadPPG } from "../../api/signalApi";
import "./PPGUploader.css";

type ParsedSignal = {
  samples: number[];
  sampleRate: number;
};

type PPGUploaderProps = {
  onParsed?: (signal: ParsedSignal) => void;
  onUploaded?: () => void;
};

const REQUIRED_HEADERS = ["value", "sample_rate"];

const parseCsv = async (file: File): Promise<ParsedSignal> => {
  const text = await file.text();
  const [headerLine, ...rows] = text.split(/\r?\n/).filter(Boolean);
  if (!headerLine) {
    throw new Error("CSV file is empty.");
  }

  const headers = headerLine.split(",").map((h) => h.trim().toLowerCase());
  for (const required of REQUIRED_HEADERS) {
    if (!headers.includes(required)) {
      throw new Error(`Missing required header: ${required}`);
    }
  }

  const valueIndex = headers.indexOf("value");
  const rateIndex = headers.indexOf("sample_rate");

  const samples: number[] = [];
  let sampleRate = 0;

  rows.forEach((row) => {
    const cols = row.split(",");
    const value = Number(cols[valueIndex]);
    const rate = Number(cols[rateIndex]);
    if (!Number.isFinite(value) || !Number.isFinite(rate)) return;
    samples.push(value);
    sampleRate = rate;
  });

  if (!samples.length) {
    throw new Error("No valid samples found in CSV.");
  }
  if (!sampleRate || sampleRate <= 0) {
    throw new Error("Invalid sample rate in CSV.");
  }

  return { samples, sampleRate };
};

export default function PPGUploader({ onParsed, onUploaded }: PPGUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<number[]>([]);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const previewData = useMemo(() => preview.slice(0, 200), [preview]);

  const handleFile = useCallback(async (selected: File) => {
    setError(null);
    setFile(selected);
    setProgress(0);

    try {
      const parsed = await parseCsv(selected);
      const previewCount = Math.min(parsed.samples.length, parsed.sampleRate * 5);
      setPreview(parsed.samples.slice(0, previewCount));
      onParsed?.(parsed);
    } catch (err) {
      setPreview([]);
      setError(err instanceof Error ? err.message : "Invalid CSV file.");
    }
  }, []);

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      void handleFile(dropped);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setProgress(10);

    try {
      await uploadPPG(file);
      setProgress(100);
      onUploaded?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="ppg-uploader">
      <div
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
      >
        <p>Drag & drop CSV here</p>
        <span>or</span>
        <label className="file-button">
          Select CSV
          <input
            type="file"
            accept=".csv"
            onChange={(event) => {
              const selected = event.target.files?.[0];
              if (selected) {
                void handleFile(selected);
              }
            }}
          />
        </label>
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      {previewData.length > 0 ? (
        <div className="preview">
          <h4>Preview (first few seconds)</h4>
          <div className="preview-chart">
            {previewData.map((value, index) => (
              <span
                key={`${value}-${index}`}
                style={{ height: `${Math.min(Math.abs(value) * 8 + 4, 60)}px` }}
              />
            ))}
          </div>
        </div>
      ) : null}

      <div className="upload-actions">
        <button type="button" disabled={!file || isUploading} onClick={handleUpload}>
          {isUploading ? "Uploading..." : "Upload"}
        </button>
        <div className="progress">
          <motion.div
            className="progress-bar"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.4 }}
          />
        </div>
      </div>
    </div>
  );
}
