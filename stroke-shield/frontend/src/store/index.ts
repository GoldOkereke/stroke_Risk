import { configureStore } from "@reduxjs/toolkit";

import cardiacReducer from "./slices/cardiacSlice";
import demoReducer from "./slices/demoSlice";
import fusionReducer from "./slices/fusionSlice";
import neuroReducer from "./slices/neuroSlice";

export const store = configureStore({
  reducer: {
    cardiac: cardiacReducer,
    demo: demoReducer,
    fusion: fusionReducer,
    neuro: neuroReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
