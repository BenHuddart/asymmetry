"""Remap a deliberately new joint result onto existing detached series owners."""

from dataclasses import replace

from asymmetry.core.representation.joint_fit import preserve_joint_source_history


def joint_source_publication(parent, launch, result, curves):
    """Retain old member/joint history when a new solve consumes different sources."""
    remap = preserve_joint_source_history(
        parent._project_model,
        {
            key: {run: fitted.input_attribution for run, fitted in results.items()}
            for key, results in result.series_results.items()
        },
        parent._next_batch_id,
    )
    if not remap:
        return launch, result, curves

    def keys(mapping):
        return {remap.get(key, key): value for key, value in mapping.items()}

    launch = replace(
        launch,
        joint_id=None,
        member_batch_ids=tuple(remap.get(key, key) for key in launch.member_batch_ids),
        shared_rows=tuple(dict(row, members=keys(row["members"])) for row in launch.shared_rows),
        models=keys(launch.models),
        global_params=keys(launch.global_params),
    )
    result = replace(
        result,
        **{
            field: keys(getattr(result, field))
            for field in (
                "series_results",
                "series_global_parameters",
                "series_reduced_chi_squared",
            )
        },
    )
    return launch, result, keys(curves)
