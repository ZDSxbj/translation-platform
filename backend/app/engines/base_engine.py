"""Abstract translation engine interface."""


class BaseEngine:
    """Every translation engine must implement this interface."""

    def get_display_name(self) -> str:
        raise NotImplementedError

    def get_description(self) -> str:
        raise NotImplementedError

    def get_stages(self) -> list[dict]:
        """Return list of stages: [{'id': '...', 'name': '...', 'description': '...'}]."""
        raise NotImplementedError

    def run_stage(self, stage_id: str, source_path: str, output_path: str,
                  config: dict, log_callback) -> dict:
        """
        Run a single pipeline stage.

        Args:
            stage_id: Stage identifier (e.g., 'stage1_prep')
            source_path: Path to the source C/C++ project
            output_path: Path where output artifacts should be written
            config: Engine configuration dict
            log_callback: Callable(message, level) for progress logging

        Returns:
            dict with 'summary' and 'details' keys
        """
        raise NotImplementedError
