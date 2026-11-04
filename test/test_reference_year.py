"""
Time frames and the "end year in the past" check count from the statement date's
year when a target has one, and from the current year otherwise.
"""
import datetime
import os
import sys
import unittest
from typing import Any
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ITR.interfaces import EScope, ETimeFrames, IDataProviderTarget
from ITR.target_validation import TargetProtocol


def _make_target(**kwargs) -> IDataProviderTarget:
    defaults: dict[str, Any] = dict(
        company_id="C1",
        target_type="abs",
        scope=EScope.S1S2,
        coverage_s1=0.95,
        coverage_s2=0.95,
        reduction_ambition=0.8,
        base_year=2019,
        base_year_ghg_s1=100,
        base_year_ghg_s2=100,
        end_year=2030,
    )
    defaults.update(kwargs)
    return IDataProviderTarget(**defaults)


def _pin_now(year: int):
    fake = mock.Mock(wraps=datetime)
    fake.datetime.now.return_value = datetime.datetime(year, 6, 1)
    return mock.patch("ITR.target_validation.datetime", fake)


class ReferenceYearTest(unittest.TestCase):
    def setUp(self):
        self.protocol = TargetProtocol()

    def test_time_frame_counts_from_statement_year(self):
        cases = [
            (2028, ETimeFrames.SHORT),
            (2029, ETimeFrames.MID),
            (2034, ETimeFrames.MID),
            (2035, ETimeFrames.LONG),
        ]
        with _pin_now(2030):
            for end_year, expected in cases:
                target = _make_target(end_year=end_year, statement_date=datetime.date(2024, 12, 31))
                self.assertEqual(self.protocol._assign_time_frame(target).time_frame, expected, end_year)

    def test_t_score_time_frame_counts_from_statement_year(self):
        with _pin_now(2030):
            target = _make_target(
                target_type="t_score", scope=EScope.S3, end_year=2024,
                statement_date=datetime.date(2024, 12, 31),
            )
            self.assertEqual(self.protocol._assign_time_frame(target).time_frame, ETimeFrames.MID)

    def test_end_year_before_statement_year_is_invalid(self):
        with _pin_now(2030):
            self.assertFalse(self.protocol._validate(
                _make_target(end_year=2023, statement_date=datetime.date(2024, 12, 31))
            ))
            self.assertTrue(self.protocol._validate(
                _make_target(end_year=2024, statement_date=datetime.date(2024, 12, 31))
            ))

    def test_t_score_end_year_before_statement_year_is_invalid(self):
        def t_score(end_year: int) -> IDataProviderTarget:
            return _make_target(
                target_type="t_score", scope=EScope.S3, end_year=end_year,
                statement_date=datetime.date(2024, 12, 31),
            )

        with _pin_now(2030):
            self.assertFalse(self.protocol._validate(t_score(2023)))
            self.assertTrue(self.protocol._validate(t_score(2024)))
            self.assertFalse(self.protocol._validate_t_score(t_score(2023)))
            self.assertTrue(self.protocol._validate_t_score(t_score(2024)))

    def test_timestamp_statement_date(self):
        with _pin_now(2030):
            target = _make_target(end_year=2029)
            target.statement_date = pd.Timestamp("2024-12-31")
            self.assertEqual(self.protocol._assign_time_frame(target).time_frame, ETimeFrames.MID)

    def test_without_statement_date_counts_from_now(self):
        with _pin_now(2030):
            self.assertEqual(
                self.protocol._assign_time_frame(_make_target(end_year=2034)).time_frame, ETimeFrames.SHORT
            )
            self.assertEqual(
                self.protocol._assign_time_frame(_make_target(end_year=2035)).time_frame, ETimeFrames.MID
            )
            self.assertFalse(self.protocol._validate(_make_target(end_year=2029)))
            self.assertTrue(self.protocol._validate(_make_target(end_year=2030)))

    def test_nat_statement_date_counts_from_now(self):
        with _pin_now(2030):
            target = _make_target(end_year=2034)
            target.statement_date = pd.NaT
            self.assertEqual(self.protocol._assign_time_frame(target).time_frame, ETimeFrames.SHORT)


if __name__ == "__main__":
    unittest.main()
