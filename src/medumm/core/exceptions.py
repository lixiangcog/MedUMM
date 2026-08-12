class MedUMMError(Exception):
    """Base exception for errors with an actionable MedUMM message."""


class ConfigurationError(MedUMMError):
    """Raised when a configuration does not satisfy the public schema."""


class ComponentNotFoundError(MedUMMError, KeyError):
    """Raised when a requested plugin has not been registered."""


class DuplicateComponentError(MedUMMError, ValueError):
    """Raised when a component name is registered more than once."""


class UnsupportedTaskError(MedUMMError, NotImplementedError):
    """Raised when a model cannot execute a requested task."""
