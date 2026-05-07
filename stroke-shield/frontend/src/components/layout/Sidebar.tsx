import { useState } from "react";
import { NavLink } from "react-router-dom";
import { useSelector } from "react-redux";

import type { AlertTier } from "../../types";
import type { RootState } from "../../store";
import "./Sidebar.css";

type NavLink = {
  label: string;
  href: string;
  badge?: string;
};

const NAV_LINKS: NavLink[] = [
  { label: "Dashboard", href: "/" },
  { label: "Cardiac", href: "/cardiac" },
  { label: "Neurological", href: "/neurological" },
  { label: "Neuro Tests", href: "/neuro-tests" },
  { label: "Demo", href: "/demo", badge: "FR6" },
];

const resolveTierClass = (tier: AlertTier | undefined): string => {
  if (tier === "CRITICAL") return "sidebar-dot critical";
  if (tier === "ADVISORY") return "sidebar-dot advisory";
  return "sidebar-dot normal";
};

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const alertTier = useSelector((state: RootState) => state.fusion.alertTier);

  return (
    <aside className={`sidebar ${isOpen ? "open" : ""}`}>
      <div className="sidebar-header">
        <div className="sidebar-title">
          <span className={resolveTierClass(alertTier)} aria-hidden="true" />
          <span>Stroke Shield</span>
        </div>
        <button
          className="sidebar-toggle"
          type="button"
          onClick={() => setIsOpen((prev) => !prev)}
          aria-label="Toggle navigation"
        >
          {isOpen ? "Close" : "Menu"}
        </button>
      </div>

      <nav className="sidebar-nav">
        {NAV_LINKS.map((link) => (
          <NavLink
            key={link.href}
            to={link.href}
            end={link.href === "/"}
            className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}
          >
            <span>{link.label}</span>
            {link.badge ? <span className="sidebar-badge">{link.badge}</span> : null}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
