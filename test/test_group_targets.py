"""
group_targets builds cells without a target in blocks; the grid must match a
cell-by-cell lookup exactly.
"""
import datetime
import itertools
import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ITR.interfaces import EScope, ETimeFrames, IDataProviderCompany, IDataProviderTarget
from ITR.target_validation import TargetProtocol


def _company(company_id: str) -> IDataProviderCompany:
    return IDataProviderCompany(
        company_name=company_id, company_id=company_id, isic="A12",
        ghg_s1=100, ghg_s2=50, ghg_s3=200,
    )


def _target(company_id: str, scope: EScope, end_year: int, **kwargs) -> IDataProviderTarget:
    fields = dict(
        company_id=company_id,
        target_type="abs",
        scope=scope,
        coverage_s1=0.95,
        coverage_s2=0.95,
        coverage_s3=0.95,
        reduction_ambition=0.5,
        base_year=2019,
        base_year_ghg_s1=100,
        base_year_ghg_s2=50,
        base_year_ghg_s3=200,
        end_year=end_year,
        statement_date=datetime.date(2025, 12, 31),
    )
    fields.update(kwargs)
    return IDataProviderTarget(**fields)


def _cell_by_cell(protocol: TargetProtocol) -> pd.DataFrame:
    grid_columns = [protocol.c.COLS.COMPANY_ID, protocol.c.COLS.TIME_FRAME, protocol.c.COLS.SCOPE]
    companies = protocol.company_data[protocol.c.COLS.COMPANY_ID].unique()
    scopes = [EScope.S1, EScope.S2, EScope.S1S2, EScope.S3, EScope.S1S2S3]
    empty_columns = [c for c in protocol.target_data.columns if c not in grid_columns]
    extended_data = pd.DataFrame(
        list(itertools.product(*[companies, ETimeFrames, scopes] + [[None]] * len(empty_columns))),
        columns=grid_columns + empty_columns,
    )
    target_columns = extended_data.columns.tolist()
    results = [protocol._find_target(row, target_columns) for _, row in extended_data.iterrows()]
    data = pd.concat([df.dropna(how='all', axis=1) for df in results], ignore_index=True)
    for col in target_columns:
        if col not in data.columns:
            data[col] = None
    return data


class GroupTargetsTest(unittest.TestCase):
    def test_grid_matches_cell_by_cell_lookup(self):
        companies = [_company(f"C{i}") for i in range(6)]
        targets = [
            _target("C0", EScope.S1S2, 2030),
            _target("C0", EScope.S1S2, 2030, reduction_ambition=0.6),
            _target("C0", EScope.S3, 2040, s3_category=0),
            _target("C2", EScope.S1, 2028),
            _target("C2", EScope.S2, 2035),
            _target("C5", EScope.S1S2S3, 2050),
        ]
        protocol = TargetProtocol()
        grids = {}
        group_targets = TargetProtocol.group_targets

        def compare_both(self):
            self.target_data = self.target_data.sort_index(level=self.target_data.index.names)
            grids['expected'] = _cell_by_cell(self)
            grids['actual'] = group_targets(self).copy()
            return self.data

        with mock.patch.object(TargetProtocol, 'group_targets', compare_both):
            protocol.process(targets, companies)

        actual, expected = grids['actual'], grids['expected']
        self.assertEqual(len(actual), 6 * 3 * 5)
        self.assertEqual(list(actual.dtypes), list(expected.dtypes))
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(actual['to_calculate'].any())
        self.assertFalse(actual['to_calculate'].all())


if __name__ == '__main__':
    unittest.main()
