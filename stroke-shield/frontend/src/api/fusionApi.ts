import axios from "axios";

import type { FusionResult } from "../types";
import { API_BASE_URL } from "../utils/constants";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export interface FusionScorePayload {
  cardiacScore?: number;
  faceScore?: number;
  voiceScore?: number;
  dysarthriaScore?: number;
  parkinsonsScore?: number;
}

export const getFusionScore = async (
  data: FusionScorePayload,
): Promise<FusionResult> => {
  const response = await api.post<FusionResult>("/api/fusion/score", data);
  return response.data;
};

export const getCardiacOnly = async (
  afibProbability: number,
  cardiacIf?: number,
): Promise<FusionResult> => {
  const response = await api.post<FusionResult>("/api/fusion/cardiac-only", {
    afibProbability,
    cardiacIf,
  });
  return response.data;
};

export const getHistory = async (): Promise<FusionResult[]> => {
  const response = await api.get<FusionResult[]>("/api/fusion/history");
  return response.data;
};

export const resetHistory = async (): Promise<void> => {
  await api.post("/api/fusion/reset");
};
