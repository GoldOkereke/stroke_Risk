import { useMemo } from "react";
import { useSelector } from "react-redux";

import type { RootState } from "../../store";
import "./Navbar.css";

type NavbarProps = {
  isAuthenticated?: boolean;
  userName?: string;
};

const resolveTierLabel = (tier: string | undefined): string => {
  if (!tier) return "NORMAL";
  return tier.toUpperCase();
};

export default function Navbar({ isAuthenticated = false, userName }: NavbarProps) {
  const alertTier = useSelector((state: RootState) => state.fusion.alertTier);
  const isProcessing = useSelector((state: RootState) => state.cardiac.isProcessing);
  const demoStage = useSelector((state: RootState) => state.demo.currentStage);

  const statusLabel = useMemo(() => {
    if (isProcessing) return "Processing";
    return "Idle";
  }, [isProcessing]);

  return (
    <header className="navbar">
      <div className="navbar-title">
        <span className="logo-dot" aria-hidden="true" />
        <span>Stroke Shield</span>
      </div>

      <div className="navbar-status">
        <span className="status-pill">Fusion: {resolveTierLabel(alertTier)}</span>
        <span className="status-pill">Cardiac: {statusLabel}</span>
        <span className="status-pill">Demo: Stage {demoStage}</span>
      </div>

      <div className="navbar-user">
        {isAuthenticated ? (
          <button type="button" className="user-button">
            {userName ?? "User"}
          </button>
        ) : (
          <button type="button" className="user-button muted">
            Guest
          </button>
        )}
      </div>
    </header>
  );
}
