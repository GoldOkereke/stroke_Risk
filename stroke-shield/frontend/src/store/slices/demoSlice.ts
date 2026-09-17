import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

export type DemoStage = 0 | 1 | 2 | 3 | 4;

export interface DemoState {
  currentStage: DemoStage;
  isPlaying: boolean;
  scenario: string;
  stageData: Record<string, number>;
}

const initialState: DemoState = {
  currentStage: 0,
  isPlaying: false,
  scenario: "pre_tia",
  stageData: {},
};

const demoSlice = createSlice({
  name: "demo",
  initialState,
  reducers: {
    setStage(state, action: PayloadAction<DemoStage>) {
      state.currentStage = action.payload;
    },
    setPlaying(state, action: PayloadAction<boolean>) {
      state.isPlaying = action.payload;
    },
    setScenario(state, action: PayloadAction<string>) {
      state.scenario = action.payload;
    },
    nextStage(state) {
      state.currentStage = ((state.currentStage + 1) % 5) as DemoStage;
    },
    setStageData(state, action: PayloadAction<Record<string, number>>) {
      state.stageData = action.payload;
    },
    resetDemo() {
      return initialState;
    },
  },
});

export const {
  setStage,
  setPlaying,
  setScenario,
  nextStage,
  setStageData,
  resetDemo,
} = demoSlice.actions;
export default demoSlice.reducer;
