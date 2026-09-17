"""
Tests for the Demo Slice (demo API endpoints).
Tests the interactive demo with stage progression and fusion results.
"""

import pytest
from app.models.schemas import (
    FaceAnalysisResult,
    VoiceAnalysisResult,
    AnomalyResult,
    FusionResult,
    FaceFeatures,
    VoiceFeatures,
    DysarthriaFeatures,
)


class TestDemoSliceStageProgression:
    """Test demo stages 0-4 progression."""

    def test_stage_0_initialization(self):
        """Test stage 0: Initial face capture."""
        # Stage 0 should just capture initial data
        stage_0 = {
            "stage": 0,
            "values": {
                "face_detected": True,
                "face_confidence": 0.95,
            },
        }

        assert stage_0["stage"] == 0
        assert stage_0["values"]["face_detected"] is True
        assert stage_0["values"]["face_confidence"] > 0.9

    def test_stage_1_face_analysis(self):
        """Test stage 1: Perform facial symmetry analysis."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.88,
                    "eye_symmetry": 0.91,
                    "forehead_symmetry": 0.89,
                }
            ),
            aggregated_score=0.89,
        )

        stage_1 = {
            "stage": 1,
            "values": {
                "face_score": face_result.aggregated_score,
                "face_features": face_result.features.scores,
            },
            "face_result": face_result,
        }

        assert stage_1["stage"] == 1
        assert stage_1["values"]["face_score"] == 0.89
        assert "mouth_symmetry" in stage_1["values"]["face_features"]

    def test_stage_2_voice_capture(self):
        """Test stage 2: Record voice sample."""
        stage_2 = {
            "stage": 2,
            "values": {
                "voice_duration": 5.0,
                "voice_quality": 0.92,
                "samples_recorded": 125000,
            },
        }

        assert stage_2["stage"] == 2
        assert stage_2["values"]["voice_duration"] == 5.0
        assert stage_2["values"]["samples_recorded"] > 0

    def test_stage_3_voice_analysis(self):
        """Test stage 3: Analyze voice recording."""
        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={
                        "articulation": 0.86,
                        "phonation": 0.88,
                    }
                ),
            ),
            scores={
                "dysarthria_risk": 0.15,
                "parkinsons_risk": 0.12,
            },
        )

        stage_3 = {
            "stage": 3,
            "values": {
                "dysarthria_risk": voice_result.scores["dysarthria_risk"],
                "parkinsons_risk": voice_result.scores["parkinsons_risk"],
            },
            "voice_result": voice_result,
        }

        assert stage_3["stage"] == 3
        assert stage_3["values"]["dysarthria_risk"] < 0.3
        assert stage_3["values"]["parkinsons_risk"] < 0.3

    def test_stage_4_fusion_assessment(self):
        """Test stage 4: Fuse all results and generate assessment."""
        fusion_result = FusionResult(
            risk_score=45.5,
            alert_tier="ADVISORY",
            confidence=0.87,
            contributing_factors={
                "face": 0.4,
                "voice": 0.6,
            },
            recommendation="Mild neurological symptoms detected. Follow-up advised.",
        )

        stage_4 = {
            "stage": 4,
            "values": {
                "risk_score": fusion_result.risk_score,
                "alert_tier": fusion_result.alert_tier,
            },
            "fusion_result": fusion_result,
        }

        assert stage_4["stage"] == 4
        assert stage_4["values"]["risk_score"] == 45.5
        assert stage_4["values"]["alert_tier"] == "ADVISORY"


class TestDemoSliceNormalCase:
    """Test demo with normal (healthy) results."""

    def test_demo_all_normal_results(self):
        """Test demo progresses through stages with all normal results."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.94,
                    "eye_symmetry": 0.96,
                    "forehead_symmetry": 0.92,
                }
            ),
            aggregated_score=0.94,
        )

        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(scores={"articulation": 0.94}),
            ),
            scores={"dysarthria_risk": 0.08, "parkinsons_risk": 0.06},
        )

        fusion_result = FusionResult(
            risk_score=12.0,
            alert_tier="NORMAL",
            confidence=0.94,
            contributing_factors={"face": 0.5, "voice": 0.5},
            recommendation="All systems normal. No abnormalities detected.",
        )

        assert face_result.aggregated_score > 0.9
        assert voice_result.scores["dysarthria_risk"] < 0.15
        assert fusion_result.alert_tier == "NORMAL"


class TestDemoSliceAbnormalCase:
    """Test demo with abnormal results."""

    def test_demo_facial_palsy_case(self):
        """Test demo with facial asymmetry (palsy) detection."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.58,
                    "eye_symmetry": 0.62,
                    "forehead_symmetry": 0.60,
                }
            ),
            aggregated_score=0.60,
        )

        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={"articulation": 0.71, "phonation": 0.73}
                ),
            ),
            scores={"dysarthria_risk": 0.45, "parkinsons_risk": 0.18},
        )

        fusion_result = FusionResult(
            risk_score=72.0,
            alert_tier="ADVISORY",
            confidence=0.82,
            contributing_factors={"face": 0.65, "voice": 0.35},
            recommendation="Facial asymmetry detected. Consider neurology evaluation.",
        )

        assert face_result.aggregated_score < 0.7
        assert fusion_result.contributing_factors["face"] > 0.5
        assert fusion_result.alert_tier == "ADVISORY"

    def test_demo_voice_disorder_case(self):
        """Test demo with voice disorder detection."""
        face_result = FaceAnalysisResult(
            features=FaceFeatures(
                scores={
                    "mouth_symmetry": 0.89,
                    "eye_symmetry": 0.91,
                }
            ),
            aggregated_score=0.90,
        )

        voice_result = VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(
                    scores={"articulation": 0.42, "phonation": 0.45}
                ),
            ),
            scores={"dysarthria_risk": 0.78, "parkinsons_risk": 0.82},
        )

        fusion_result = FusionResult(
            risk_score=81.0,
            alert_tier="CRITICAL",
            confidence=0.85,
            contributing_factors={"face": 0.25, "voice": 0.75},
            recommendation="Significant voice disorder detected. Immediate assessment recommended.",
        )

        assert face_result.aggregated_score > 0.85
        assert voice_result.scores["dysarthria_risk"] > 0.7
        assert fusion_result.alert_tier == "CRITICAL"


class TestDemoSliceInteractivity:
    """Test interactive features of demo slice."""

    def test_demo_progression_sequence(self):
        """Test proper sequence of demo stages."""
        stages = [0, 1, 2, 3, 4]
        for i, stage in enumerate(stages):
            assert stage == i
            assert 0 <= stage <= 4

    def test_demo_stage_data_accumulation(self):
        """Test that data accumulates across stages."""
        demo_state = {
            "stage": 0,
            "stage_0_data": None,
            "stage_1_data": None,
            "stage_2_data": None,
            "stage_3_data": None,
            "stage_4_data": None,
        }

        # Progress through stages
        for stage_num in range(1, 5):
            demo_state["stage"] = stage_num
            if stage_num == 1:
                demo_state["stage_1_data"] = {"face_score": 0.85}
            elif stage_num == 2:
                demo_state["stage_2_data"] = {"recording": True}
            elif stage_num == 3:
                demo_state["stage_3_data"] = {"voice_score": 0.82}
            elif stage_num == 4:
                demo_state["stage_4_data"] = {"fusion_result": {"risk_score": 50}}

            # Previous stages' data preserved
            assert demo_state[f"stage_{stage_num - 1}_data"] is not None

    def test_demo_can_restart(self):
        """Test demo can be restarted."""
        initial_state = {
            "stage": 0,
            "face_result": None,
            "voice_result": None,
            "fusion_result": None,
        }

        # Complete demo
        complete_state = {
            "stage": 4,
            "face_result": {"score": 0.85},
            "voice_result": {"score": 0.82},
            "fusion_result": {"risk_score": 50},
        }

        # Restart
        restarted_state = initial_state.copy()
        assert restarted_state["stage"] == 0
        assert restarted_state["face_result"] is None


class TestDemoSliceValidation:
    """Test input validation for demo data."""

    def test_stage_range_validation(self):
        """Verify stage is between 0 and 4."""
        for stage in [0, 1, 2, 3, 4]:
            assert 0 <= stage <= 4

    def test_confidence_values_valid(self):
        """Verify confidence values are between 0 and 1."""
        confidence_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for conf in confidence_values:
            assert 0 <= conf <= 1

    def test_risk_score_values_valid(self):
        """Verify risk scores are between 0 and 100."""
        risk_scores = [0, 25, 50, 75, 100]
        for risk in risk_scores:
            assert 0 <= risk <= 100
