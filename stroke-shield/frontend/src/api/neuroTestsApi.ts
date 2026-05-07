import axios from "axios";

import { API_BASE_URL } from "../utils/constants";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const sendReactionTime = async (trials: number[]): Promise<any> => {
  const response = await api.post("/api/neuro-tests/reaction-time", { trials });
  return response.data;
};

export const sendFingerTracking = async (data: Record<string, number>): Promise<any> => {
  const response = await api.post("/api/neuro-tests/finger-tracking", data);
  return response.data;
};

export const sendCognitiveTest = async (results: Record<string, number>): Promise<any> => {
  const response = await api.post("/api/neuro-tests/cognitive", results);
  return response.data;
};

export const sendTapSpeed = async (taps: number[]): Promise<any> => {
  const response = await api.post("/api/neuro-tests/tap-speed", { taps });
  return response.data;
};

export const getCombinedScore = async (): Promise<any> => {
  const response = await api.post("/api/neuro-tests/combined");
  return response.data;
};
