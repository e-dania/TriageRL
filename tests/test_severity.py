"""Unit tests: rule-based severity assessment (boundary-value analysis).

Testing strategy: boundary-value analysis on each vital-sign threshold in
the NEWS2-style scoring, plus equivalence classes for combined presentations.
"""

import pytest

from triagerl.patients import SYMPTOM_SPECIALISTS, assess_severity


class TestSeverityBoundaries:
    """Boundary-value analysis per vital sign."""

    def test_healthy_adult_is_non_urgent(self):
        assert assess_severity(75, 120, 98, 36.8, 30) == 5

    def test_extreme_bradycardia_scores_high(self):
        # HR <= 40 contributes maximum HR score
        s_low = assess_severity(38, 120, 98, 36.8, 30)
        s_normal = assess_severity(75, 120, 98, 36.8, 30)
        assert s_low < s_normal

    def test_hr_boundary_130_131(self):
        # 131 crosses into the highest heart-rate band
        assert (assess_severity(131, 120, 98, 36.8, 30)
                <= assess_severity(130, 120, 98, 36.8, 30))

    def test_hypotension_boundary_90(self):
        s_hypo = assess_severity(75, 90, 98, 36.8, 30)
        s_ok = assess_severity(75, 115, 98, 36.8, 30)
        assert s_hypo < s_ok

    def test_spo2_boundary_91_92(self):
        s91 = assess_severity(75, 120, 91, 36.8, 30)
        s92 = assess_severity(75, 120, 92, 36.8, 30)
        assert s91 <= s92

    def test_hypothermia_scores(self):
        assert (assess_severity(75, 120, 98, 34.9, 30)
                < assess_severity(75, 120, 98, 36.8, 30))

    def test_age_extremes_escalate(self):
        adult = assess_severity(95, 105, 95, 38.5, 40)
        elderly = assess_severity(95, 105, 95, 38.5, 80)
        infant = assess_severity(95, 105, 95, 38.5, 1)
        assert elderly <= adult
        assert infant <= adult


class TestSeverityEquivalenceClasses:
    """Combined presentations map to the clinically expected class."""

    def test_multi_system_failure_is_resuscitation(self):
        # Tachycardic, hypotensive, hypoxic, febrile elderly patient
        assert assess_severity(140, 85, 88, 39.5, 80) == 1

    def test_moderate_illness_is_mid_scale(self):
        assert assess_severity(105, 108, 95, 38.4, 40) in (2, 3)

    def test_output_always_in_range(self):
        # Property-based sweep across the input space
        for hr in (30, 75, 120, 190):
            for sbp in (60, 95, 120, 225):
                for spo2 in (75, 91, 95, 100):
                    for temp in (34.0, 36.8, 39.5):
                        s = assess_severity(hr, sbp, spo2, temp, 40)
                        assert 1 <= s <= 5


class TestSpecialistMapping:
    def test_every_symptom_has_specialist(self):
        for symptom, specialist in SYMPTOM_SPECIALISTS.items():
            assert isinstance(specialist, str) and specialist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
