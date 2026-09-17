import axios from "axios";

import { API_BASE_URL } from "../utils/constants";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const analyzeFace = async (imageFile: File): Promise<any> => {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await api.post("/api/neurological/face/analyze", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const analyzeFaceSession = async (images: File[]): Promise<any> => {
  const formData = new FormData();
  images.forEach((image) => formData.append("files", image));

  const response = await api.post("/api/neurological/face/session", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const addFaceBaseline = async (imageFile: File): Promise<any> => {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await api.post("/api/neurological/face/baseline", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const analyzeVoice = async (audioFile: File): Promise<any> => {
  const formData = new FormData();
  formData.append("file", audioFile);

  const response = await api.post("/api/neurological/voice/analyze", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const addVoiceBaseline = async (audioFile: File): Promise<any> => {
  const formData = new FormData();
  formData.append("file", audioFile);

  const response = await api.post("/api/neurological/voice/baseline", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const runSpeechTest = async (
  promptIndex: number,
  audioFile: File,
): Promise<any> => {
  const formData = new FormData();
  formData.append("prompt_index", String(promptIndex));
  formData.append("file", audioFile);

  const response = await api.post(
    "/api/neurological/voice/speech-test",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    },
  );
  return response.data;
};

export const getBaselineStatus = async (): Promise<any> => {
  const response = await api.get("/api/neurological/baseline/status");
  return response.data;
};
