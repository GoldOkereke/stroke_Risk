import axios from "axios";

import type { CardiacResult } from "../types";
import { API_BASE_URL } from "../utils/constants";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getRecords = async (): Promise<string[]> => {
  const response = await api.get<string[]>("/api/cardiac/records");
  return response.data;
};

export const uploadPPG = async (file: File): Promise<CardiacResult> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<CardiacResult>("/api/cardiac/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const loadMITBIH = async (recordId: string): Promise<CardiacResult> => {
  const response = await api.post<CardiacResult>(`/api/cardiac/mitbih/${recordId}`);
  return response.data;
};

export const loadMIMIC = async (recordId: string): Promise<CardiacResult> => {
  const response = await api.post<CardiacResult>(`/api/cardiac/mimic/${recordId}`);
  return response.data;
};

export const simulatePPG = async (scenario: string): Promise<CardiacResult> => {
  const response = await api.post<CardiacResult>("/api/cardiac/simulate", {
    scenario,
  });
  return response.data;
};
