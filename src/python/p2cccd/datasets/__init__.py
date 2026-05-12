"""External dataset adapters used by P2CCCD benchmark suites."""

from .tight_inclusion_queries import (
    TIGHT_INCLUSION_CSV_COLUMNS,
    TIGHT_INCLUSION_QUERY_ROWS,
    TightInclusionCSVFile,
    TightInclusionFileSplit,
    TightInclusionPrimitiveQuery,
    build_file_level_split,
    discover_tight_inclusion_csv_files,
    inspect_tight_inclusion_csv,
    iter_dataset_queries,
    iter_tight_inclusion_queries,
    read_tight_inclusion_query,
)
from .tight_inclusion_stpf_features import tight_inclusion_query_to_proposal_row
from .ccd100_full_scene_queries import (
    FullSceneCCDQueryFile,
    discover_full_scene_query_files,
    iter_full_scene_queries,
)

__all__ = [
    "cad",
    "ccd",
    "objects",
    "robot",
    "TIGHT_INCLUSION_CSV_COLUMNS",
    "TIGHT_INCLUSION_QUERY_ROWS",
    "TightInclusionCSVFile",
    "TightInclusionFileSplit",
    "TightInclusionPrimitiveQuery",
    "FullSceneCCDQueryFile",
    "build_file_level_split",
    "discover_full_scene_query_files",
    "discover_tight_inclusion_csv_files",
    "inspect_tight_inclusion_csv",
    "iter_dataset_queries",
    "iter_full_scene_queries",
    "iter_tight_inclusion_queries",
    "read_tight_inclusion_query",
    "tight_inclusion_query_to_proposal_row",
]
