"""Bad data is a row outcome. Software and infrastructure failures are not."""


class RejectedRecord(ValueError):
    """A known, recoverable defect in one input record."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class CompatibilityError(RuntimeError):
    """A model, schema, or index belongs to another library/configuration."""


class ContractError(RuntimeError):
    """An adapter changed row alignment or returned an invalid shape."""
