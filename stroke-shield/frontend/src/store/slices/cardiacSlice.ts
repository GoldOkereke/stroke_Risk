import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

import type { CardiacResult } from "../../types";

export interface CardiacState {
  afibProbability: number;
  prediction: "AFIB" | "NORMAL";
  confidence: number;
  signalQuality: number;
  waveform: number[];
  isProcessing: boolean;
}

const initialState: CardiacState = {
  afibProbability: 0,
  prediction: "NORMAL",
  confidence: 0,
  signalQuality: 0,
  waveform: [],
  isProcessing: false,
};

const cardiacSlice = createSlice({
  name: "cardiac",
  initialState,
  reducers: {
    setCardiacResult(state, action: PayloadAction<CardiacResult>) {
      const { afibProbability, prediction, confidence, signalQuality } = action.payload;
      state.afibProbability = afibProbability;
      state.prediction = prediction;
      state.confidence = confidence;
      state.signalQuality = signalQuality;
    },
    setWaveform(state, action: PayloadAction<number[]>) {
      state.waveform = action.payload;
    },
    setProcessing(state, action: PayloadAction<boolean>) {
      state.isProcessing = action.payload;
    },
    resetCardiac() {
      return initialState;
    },
  },
});

export const { setCardiacResult, setWaveform, setProcessing, resetCardiac } =
  cardiacSlice.actions;
export default cardiacSlice.reducer;
