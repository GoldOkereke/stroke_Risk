import axios from "axios";

import type { DemoStage } from "../types";
import { API_BASE_URL } from "../utils/constants";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const startDemo = async (): Promise<void> => {
  await api.post("/api/demo/start");
};

export const getStage = async (stageNumber: number): Promise<DemoStage> => {
  const response = await api.get<DemoStage>(`/api/demo/stage/${stageNumber}`);
  return response.data;
};

export const runFullDemo = async (): Promise<void> => {
  await api.post("/api/demo/run-full");
};

export const getScenario = async (scenarioName: string): Promise<any> => {
  const response = await api.get(`/api/demo/scenario/${scenarioName}`);
  return response.data;
};

export const injectToFusion = async (): Promise<void> => {
  await api.post("/api/demo/inject-to-fusion");
};
