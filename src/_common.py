"""Input validation shared by the study analyses."""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

MISSING_ATLAS_DEFAULT = False


def require_configured_paths(namespace, inputs, outputs):
    """Require users to replace descriptive placeholders with their own files."""
    for key in inputs + outputs:
        if str(namespace[key]).startswith(('Shapefile containing', 'Excel workbook containing', 'Output ')):
            raise ValueError(f'Replace the descriptive {key} assignment with your own file location before running this analysis.')
    for key in inputs:
        if not Path(namespace[key]).is_file():
            raise FileNotFoundError(f'The file configured in {key} does not exist.')
    input_paths = {Path(namespace[key]).resolve() for key in inputs}
    output_paths = [Path(namespace[key]).resolve() for key in outputs]
    if input_paths.intersection(output_paths) or len(output_paths) != len(set(output_paths)):
        raise ValueError('Output files must be distinct and must not overwrite inputs.')


def validate_table(table, numeric_fields, label, id_field=None, nonnegative=False):
    """Reject ambiguous IDs and missing or non-finite quantitative inputs."""
    required = list(numeric_fields) + ([id_field] if id_field else [])
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f'{label}: missing fields {missing}.')
    if id_field:
        if table[id_field].isna().any() or table[id_field].duplicated().any():
            raise ValueError(f'{label}: {id_field} values must be present and unique.')
    for field in numeric_fields:
        values = pd.to_numeric(table[field], errors='raise')
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: {field} contains missing or non-finite values; do not infer zero.')
        if (nonnegative or field == 'structure') and (values < 0).any():
            raise ValueError(f'{label}: {field} contains negative damage values.')
        table[field] = values


def require_coverage(reference, other, label, id_field='fd_id'):
    """Prevent silent record loss from incomplete ID joins."""
    missing = ~reference[id_field].isin(other[id_field])
    if missing.any():
        raise ValueError(f'{label}: missing {int(missing.sum())} building IDs required by the SST depth layer.')


def align_atlas(reference, atlas, id_field='fd_id', missing_is_dry=False):
    """Align Atlas records, with an explicit policy for an inundated-only export."""
    missing = ~reference[id_field].isin(atlas[id_field])
    if missing.any() and not missing_is_dry:
        raise ValueError('Atlas-14 layer has unmatched SST building IDs. Supply the complete inventory, '
                         'or use MISSING_ATLAS_IS_DRY = True only for a documented inundated-only export.')
    aligned = reference[[id_field]].merge(atlas.drop(columns='geometry', errors='ignore'),
                                         on=id_field, how='left', validate='one_to_one', indicator=True)
    absent = aligned['_merge'].eq('left_only')
    if absent.any():
        warnings.warn(f'Treating {int(absent.sum())} absent Atlas-14 records as dry by explicit configuration.', stacklevel=2)
        aligned.loc[absent, 'depth'] = 0.0
        if 'structure' in aligned:
            aligned.loc[absent, 'structure'] = 0.0
    return aligned.drop(columns='_merge')


def realization_columns(columns, kind, expected=50):
    """Match full and 10-character DBF field names, requiring all study runs."""
    suffix = r'depth' if kind == 'depth' else r'(?:struct|structu|structure)'
    pattern = re.compile(rf'^R([1-9][0-9]*)_{suffix}$', flags=re.IGNORECASE)
    matched = {}
    for column in columns:
        result = pattern.fullmatch(str(column))
        if result:
            run = int(result.group(1))
            if run in matched:
                raise ValueError(f'Multiple {kind} fields map to realization {run}.')
            matched[run] = column
    required = set(range(1, expected + 1))
    if set(matched) != required:
        raise ValueError(f'{kind}: expected runs 1-{expected}; missing {sorted(required-set(matched))}, '
                         f'unexpected {sorted(set(matched)-required)}.')
    return dict(sorted(matched.items()))
