"""Public exceptions for invalid data, unsupported formats, and closed files."""


class DataError(ValueError):
    """An OSNAP invariant or scientific precondition was violated."""


class UnitError(DataError):
    """An unknown or incompatible unit was requested."""


class FormatError(DataError):
    """An external or OSNAP file does not match a supported schema."""


class ClosedDatasetError(RuntimeError):
    """A disk-backed value was accessed after its file was closed."""
