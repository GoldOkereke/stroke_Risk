import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

export interface BaselineProgress {
  face: number;
  voice: number;
}

export interface NeuroState {
  faceScore: number;
  voiceScore: number;
  dysarthriaScore: number;
  parkinsonsScore: number;
  baselineProgress: BaselineProgress;
  isCapturing: boolean;
}

const initialState: NeuroState = {
  faceScore: 0,
  voiceScore: 0,
  dysarthriaScore: 0,
  parkinsonsScore: 0,
  baselineProgress: {
    face: 0,
    voice: 0,
  },
  isCapturing: false,
};

const neuroSlice = createSlice({
  name: "neuro",
  initialState,
  reducers: {
    setNeuroResults(state, action: PayloadAction<Partial<NeuroState>>) {
      Object.assign(state, action.payload);
    },
    updateBaselineProgress(state, action: PayloadAction<keyof BaselineProgress>) {
      const key = action.payload;
      state.baselineProgress[key] += 1;
    },
    setCapturing(state, action: PayloadAction<boolean>) {
      state.isCapturing = action.payload;
    },
    resetNeuro() {
      return initialState;
    },
  },
});

export const { setNeuroResults, updateBaselineProgress, setCapturing, resetNeuro } =
  neuroSlice.actions;
export default neuroSlice.reducer;
