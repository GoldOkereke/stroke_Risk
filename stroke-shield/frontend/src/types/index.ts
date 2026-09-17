// Shared types for the frontend app.

export type AlertTier = "NORMAL" | "ADVISORY" | "CRITICAL";

export interface FusionResult {
  riskScore: number;
  alertTier: AlertTier;
  confidence: number;
  contributingFactors: Record<string, number>;
  recommendation: string;
}

export interface CardiacResult {
  afibProbability: number;
  prediction: "AFIB" | "NORMAL";
  confidence: number;
  signalQuality: number;
}

export interface NeuroResult {
  faceScore: number;
  voiceScore: number;
  baselineProgress: number;
}

export interface DemoStage {
  stage: 0 | 1 | 2 | 3 | 4;
  values: Record<string, number>;
  fusionResult: FusionResult;
}

export interface SignalUploadRequest {
  signalType: string;
  signalSource: string;
  filename?: string;
  sampleRateHz?: number;
  durationSeconds?: number;
  metadata?: Record<string, string>;
}

export interface SegmentSchema {
  hrvRmssd?: number;
  hrvSdnn?: number;
  rrIrregularity?: number;
  samples: number[];
}

export interface FilteredSignalResponse {
  signalType: string;
  signalSource: string;
  filteredSamples: number[];
  segment?: SegmentSchema;
  metadata?: Record<string, string>;
}

export interface SignalStatusResponse {
  status: string;
  message?: string;
  timestamp: string;
}

export interface FaceFeatures {
  scores: Record<string, number>;
}

export interface FaceAnalysisResult {
  features: FaceFeatures;
  aggregatedScore: number;
}

export interface DysarthriaFeatures {
  scores: Record<string, number>;
}

export interface ParkinsonsFeatures {
  scores: Record<string, number>;
}

export interface VoiceFeatures {
  dysarthria?: DysarthriaFeatures;
  parkinsons?: ParkinsonsFeatures;
}

export interface VoiceAnalysisResult {
  features: VoiceFeatures;
  scores: Record<string, number>;
}

export interface AnomalyResult {
  score: number;
  confidence: number;
  contributingFeatures: Record<string, number>;
}
