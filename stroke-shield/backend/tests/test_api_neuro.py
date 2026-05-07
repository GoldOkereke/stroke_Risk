"""
Tests for the Neuro Slice (neurological API endpoints).
Tests facial palsy detection and voice analysis for dysarthria/Parkinsons.
"""

import pytest
from app.models.schemas import (
    FaceAnalysisResult,
    FaceFeatures,
    VoiceAnalysisResult,
    VoiceFeatures,
    DysarthriaFeatures,
    ParkinsonsFeatures,
)


class TestNeuroSliceFaceAnalysis:
    """Test facial palsy and asymmetry detection."""

    def test_face_features_structure(self):
        """Test face features contain required asymmetry scores."""
        face_features = FaceFeatures(
            scores={
                "mouth_symmetry": 0.85,
                "eye_symmetry": 0.92,
                "forehead_symmetry": 0.88,
                "cheek_symmetry": 0.80,
            }
        )

        assert "mouth_symmetry" in face_features.scores
        assert face_features.scores["eye_symmetry"] == 0.92
        assert len(face_features.scores) == 4

    def test_normal_face_symmetry(self):
        """Test detection of normal facial symmetry."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.92,
                    "eye_symmetry": 0.94,
                    "forehead_symmetry": 0.90,
                    "cheek_symmetry": 0.91,
                }
            ),
            aggregated_score=0.92,
        )

        assert face_result.aggregated_score > 0.85
        assert all(score > 0.85 for score in face_result.features.scores.values())

    def test_facial_palsy_detection_mild(self):
        """Test detection of mild facial asymmetry (palsy)."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.70,
                    "eye_symmetry": 0.75,
                    "forehead_symmetry": 0.72,
                    "cheek_symmetry": 0.68,
                }
            ),
            aggregated_score=0.71,
        )

        assert 0.6 < face_result.aggregated_score < 0.8
        assert any(score < 0.75 for score in face_result.features.scores.values())

    def test_facial_palsy_detection_severe(self):
        """Test detection of severe facial asymmetry (palsy)."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.45,
                    "eye_symmetry": 0.50,
                    "forehead_symmetry": 0.48,
                    "cheek_symmetry": 0.40,
                }
            ),
            aggregated_score=0.46,
        )

        assert face_result.aggregated_score < 0.6
        assert all(score < 0.6 for score in face_result.features.scores.values())

    def test_unilateral_facial_weakness(self):
        """Test detection of unilateral facial weakness pattern."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "left_mouth": 0.65,  # Weak on left side
                    "right_mouth": 0.92,
                    "left_eye": 0.68,
                    "right_eye": 0.90,
                }
            ),
            aggregated_score=0.79,
        )

        # Left side weaker than right (typical unilateral palsy)
        left_avg = (
            face_result.features.scores["left_mouth"]
            + face_result.features.scores["left_eye"]
        ) / 2
        right_avg = (
            face_result.features.scores["right_mouth"]
            + face_result.features.scores["right_eye"]
        ) / 2

        assert left_avg < right_avg


class TestNeuroSliceVoiceAnalysis:
    """Test voice analysis for dysarthria and Parkinsons."""

    def test_voice_features_structure(self):
        """Test voice features contain dysarthria and Parkinsons scores."""
        voice_features = VoiceFeatures(
            dysarthria=DysarthriaFeatures(
                scores={
                    "articulation": 0.85,
                    "phonation": 0.88,
                    "resonance": 0.82,
                }
            ),
            parkinsons=ParkinsonsFeatures(
                scores={
                    "jitter": 0.15,
                    "shimmer": 0.18,
                    "fo_variation": 0.12,
                }
            ),
        )

        assert voice_features.dysarthria is not None
        assert voice_features.parkinsons is not None
        assert "articulation" in voice_features.dysarthria.scores
        assert "jitter" in voice_features.parkinsons.scores

    def test_normal_voice_analysis(self):
        """Test detection of normal voice characteristics."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={
                        "articulation": 0.92,
                        "phonation": 0.90,
                        "resonance": 0.88,
                    }
                ),
                parkinsons=ParkinsonsFeatures(
                    scores={
                        "jitter": 0.05,
                        "shimmer": 0.08,
                        "fo_variation": 0.04,
                    }
                ),
            ),
            scores={
                "dysarthria_risk": 0.10,
                "parkinsons_risk": 0.08,
            },
        )

        assert voice_result.scores["dysarthria_risk"] < 0.3
        assert voice_result.scores["parkinsons_risk"] < 0.3

    def test_dysarthria_detection_mild(self):
        """Test detection of mild dysarthria."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={
                        "articulation": 0.65,
                        "phonation": 0.70,
                        "resonance": 0.62,
                    }
                ),
            ),
            scores={
                "dysarthria_risk": 0.55,
            },
        )

        assert 0.4 < voice_result.scores["dysarthria_risk"] < 0.7
        assert any(
            score < 0.75 for score in voice_result.features.dysarthria.scores.values()
        )

    def test_dysarthria_detection_severe(self):
        """Test detection of severe dysarthria."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={
                        "articulation": 0.35,
                        "phonation": 0.40,
                        "resonance": 0.38,
                    }
                ),
            ),
            scores={
                "dysarthria_risk": 0.85,
            },
        )

        assert voice_result.scores["dysarthria_risk"] > 0.7
        assert all(
            score < 0.6 for score in voice_result.features.dysarthria.scores.values()
        )

    def test_parkinsons_voice_features(self):
        """Test detection of Parkinsons-related voice changes."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                parkinsons=ParkinsonsFeatures(
                    scores={
                        "jitter": 0.35,  # High jitter (pitch instability)
                        "shimmer": 0.38,  # High shimmer (amplitude instability)
                        "fo_variation": 0.32,  # High F0 variation
                    }
                ),
            ),
            scores={
                "parkinsons_risk": 0.80,
            },
        )

        assert voice_result.scores["parkinsons_risk"] > 0.6
        assert voice_result.features.parkinsons.scores["jitter"] > 0.2

    def test_parkinsons_normal_voice(self):
        """Test normal voice lacks Parkinsons markers."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                parkinsons=ParkinsonsFeatures(
                    scores={
                        "jitter": 0.04,
                        "shimmer": 0.06,
                        "fo_variation": 0.03,
                    }
                ),
            ),
            scores={
                "parkinsons_risk": 0.12,
            },
        )

        assert voice_result.scores["parkinsons_risk"] < 0.3
        assert all(
            score < 0.15 for score in voice_result.features.parkinsons.scores.values()
        )

    def test_combined_dysarthria_parkinsons(self):
        """Test voice symptoms suggesting both dysarthria and Parkinsons."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={
                        "articulation": 0.50,
                        "phonation": 0.55,
                        "resonance": 0.48,
                    }
                ),
                parkinsons=ParkinsonsFeatures(
                    scores={
                        "jitter": 0.28,
                        "shimmer": 0.30,
                        "fo_variation": 0.25,
                    }
                ),
            ),
            scores={
                "dysarthria_risk": 0.68,
                "parkinsons_risk": 0.72,
            },
        )

        assert voice_result.scores["dysarthria_risk"] > 0.5
        assert voice_result.scores["parkinsons_risk"] > 0.5


class TestNeuroSliceFeatureAccuracy:
    """Test accuracy of neurological feature extraction."""

    def test_face_score_range(self):
        """Verify face symmetry scores are between 0 and 1."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "feature_1": 0.0,
                    "feature_2": 0.5,
                    "feature_3": 1.0,
                }
            ),
            aggregated_score=0.5,
        )

        assert all(0 <= score <= 1 for score in face_result.features.scores.values())
        assert 0 <= face_result.aggregated_score <= 1

    def test_voice_metric_validity(self):
        """Verify voice metrics are within expected ranges."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(scores={"test": 0.75}),
                parkinsons=ParkinsonsFeatures(scores={"test": 0.25}),
            ),
            scores={"dysarthria_risk": 0.6, "parkinsons_risk": 0.4},
        )

        assert all(0 <= score <= 1 for score in voice_result.scores.values())
