import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

import type { AlertTier, FusionResult } from "../../types";

export interface FusionState {
  riskScore: number;
  alertTier: AlertTier;
  confidence: number;
  contributingFactors: Record<string, number>;
  recommendation: string;
  history: number[];
  trend: number;
}

const initialState: FusionState = {
  riskScore: 0,
  alertTier: "NORMAL",
  confidence: 0,
  contributingFactors: {},
  recommendation: "",
  history: [],
  trend: 0,
};

const fusionSlice = createSlice({
  name: "fusion",
  initialState,
  reducers: {
    setFusionResult(state, action: PayloadAction<FusionResult>) {
      const { riskScore, alertTier, confidence, contributingFactors, recommendation } = action.payload;
      state.riskScore = riskScore;
      state.alertTier = alertTier;
      state.confidence = confidence;
      state.contributingFactors = contributingFactors;
      state.recommendation = recommendation;
    },
    addToHistory(state, action: PayloadAction<number>) {
      state.history.push(action.payload);
      if (state.history.length > 20) {
        state.history = state.history.slice(-20);
      }
    },
    setTrend(state, action: PayloadAction<number>) {
      state.trend = action.payload;
    },
    resetFusion() {
      return initialState;
    },
  },
});

export const { setFusionResult, addToHistory, setTrend, resetFusion } = fusionSlice.actions;
export default fusionSlice.reducer;
