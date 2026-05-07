import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

import "./App.css";
import Sidebar from "./components/layout/Sidebar";
import Navbar from "./components/layout/Navbar";
import DashboardPage from "./pages/DashboardPage";
import CardiacPage from "./pages/CardiacPage";
import NeurologicalPage from "./pages/NeurologicalPage";
import NeuroTestsPage from "./pages/NeuroTestsPage";
import DemoPage from "./pages/DemoPage";

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.2 }}
        className="route-wrapper"
      >
        <Routes location={location}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/cardiac" element={<CardiacPage />} />
          <Route path="/neurological" element={<NeurologicalPage />} />
          <Route path="/neuro-tests" element={<NeuroTestsPage />} />
          <Route path="/demo" element={<DemoPage />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <div className="app-main">
          <Navbar />
          <AnimatedRoutes />
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App
