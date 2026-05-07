import { useEffect, useState } from "react";

import type { AlertTier } from "../../types";
import "./AlertModal.css";

type AlertModalProps = {
  alertTier: AlertTier;
  recommendation: string;
  onAcknowledge?: () => void;
};

export default function AlertModal({
  alertTier,
  recommendation,
  onAcknowledge,
}: AlertModalProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [hasAcknowledged, setHasAcknowledged] = useState(false);

  useEffect(() => {
    if (alertTier === "CRITICAL" && !hasAcknowledged) {
      setIsOpen(true);
    }
  }, [alertTier, hasAcknowledged]);

  const handleAcknowledge = () => {
    setIsOpen(false);
    setHasAcknowledged(true);
    onAcknowledge?.();
  };

  if (!isOpen) return null;

  return (
    <div className="alert-modal-backdrop" role="dialog" aria-modal="true">
      <div className="alert-modal">
        <h2>Critical Alert</h2>
        <p>{recommendation}</p>
        <div className="alert-guidance">
          <p>Emergency guidance:</p>
          <ul>
            <li>Call emergency services immediately.</li>
            <li>Do not drive yourself. Ask for help.</li>
            <li>Keep the person calm and seated.</li>
          </ul>
        </div>
        <button type="button" className="alert-ack" onClick={handleAcknowledge}>
          I Understand
        </button>
      </div>
    </div>
  );
}
