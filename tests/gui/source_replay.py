"""Internal sample-file doubles retain the load purpose's typed authority."""

from asymmetry.core.io.run_preparation import ReplayState, prepare_loaded_run


def sample_decoder(decode, *, expected_path=None):
    """Adapt a synthetic file decoder without changing any scientific arrays."""

    def load(path, *, replay=None):
        assert isinstance(path, str)
        assert expected_path is None or path == str(expected_path)
        assert replay is None or isinstance(replay, ReplayState)
        assert replay is None or replay.run_number is not None
        result = decode(path)
        if result is None:
            return None
        datasets = result if isinstance(result, list) else [result]
        assert all(dataset.run is not None for dataset in datasets)
        assert replay is None or any(replay.selects(dataset.run) for dataset in datasets)
        for dataset in datasets:
            dataset.run = prepare_loaded_run(dataset.run, replay)
            if replay is not None and replay.selects(dataset.run):
                assert dataset.run.instrument_choice == replay.instrument_choice
                if replay.source is not None:
                    assert dataset.run.source_assessment == replay.source
        return result

    return load


def sample_method(decode, *, expected_path=None):
    """Keep class-installed file doubles on the same keyword-only load contract."""
    return staticmethod(sample_decoder(decode, expected_path=expected_path))
