import { useState } from "react";
import { Alert, Snackbar } from "@mui/material";
import { isAxiosError } from "axios";

import { addFaceBaseline, addVoiceBaseline } from "../../api/neurologicalApi";

import "./BaselineSetup.css";

type BaselineSetupProps = {
  onComplete?: () => void;
  faceSample?: File | null;
  voiceSample?: File | null;
};

const steps = [
  "Face capture (multiple angles)",
  "Voice recording (sustained vowel)",
  "Voice recording (reading passage)",
];

export default function BaselineSetup({ onComplete, faceSample, voiceSample }: BaselineSetupProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState("");

  const isFinalStep = currentStep === steps.length - 1;

  const handleNext = () => {
    setError(null);
    setStatus(null);
    setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1));
  };

  const handleCreateBaseline = async () => {
    if (!faceSample) {
      const message = "Capture at least one face sample before creating the baseline.";
      setError(message);
      setToastMessage(message);
      setToastOpen(true);
      return;
    }

    try {
      setIsSubmitting(true);
      setError(null);
      setStatus(voiceSample ? "Creating face and voice baseline..." : "Creating face baseline...");

      const requests = [addFaceBaseline(faceSample)];

      if (voiceSample) {
        requests.push(addVoiceBaseline(voiceSample));
      }

      await Promise.all(requests);

      setStatus("Baseline created successfully.");
      onComplete?.();
    } catch (submissionError) {
      let message = "Unable to create the baseline. Please try again.";
      if (isAxiosError(submissionError)) {
        const statusCode = submissionError.response?.status;
        const apiDetail = submissionError.response?.data?.detail;
        if (statusCode === 400) {
          message =
            typeof apiDetail === "string" && apiDetail.trim().length > 0
              ? apiDetail
              : "No face detected. Please capture a clear front-facing image.";
        } else if (typeof apiDetail === "string" && apiDetail.trim().length > 0) {
          message = apiDetail;
        } else if (submissionError.message) {
          message = submissionError.message;
        }
      } else if (submissionError instanceof Error) {
        message = submissionError.message;
      }

      setError(message);
      setStatus(null);
      setToastMessage(message);
      setToastOpen(true);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="baseline-setup">
      <header>
        <h3>Baseline Setup</h3>
        <p>
          Step {currentStep + 1} of {steps.length}
        </p>
      </header>

      <div className="baseline-step">
        <p>{steps[currentStep]}</p>
        <div className="baseline-actions">
          {!isFinalStep ? (
            <button type="button" className="secondary" onClick={handleNext}>
              Next Step
            </button>
          ) : (
            <button type="button" onClick={handleCreateBaseline} disabled={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Baseline"}
            </button>
          )}
        </div>
      </div>

      <div className="baseline-progress">
        <div>
          <span>Face sample</span>
          <strong>{faceSample ? "Captured" : "Missing"}</strong>
        </div>
        <div>
          <span>Voice sample</span>
          <strong>{voiceSample ? "Captured" : "Optional"}</strong>
        </div>
        <div>
          <span>Current step</span>
          <strong>
            {currentStep + 1}/{steps.length}
          </strong>
        </div>
      </div>

      {status ? <div className="baseline-status">{status}</div> : null}
      {error ? <div className="baseline-error">{error}</div> : null}

      <Snackbar
        open={toastOpen}
        autoHideDuration={5000}
        onClose={() => setToastOpen(false)}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
      >
        <Alert onClose={() => setToastOpen(false)} severity="error" variant="filled" sx={{ width: "100%" }}>
          {toastMessage}
        </Alert>
      </Snackbar>
    </section>
  );
}
