class ProviderError(RuntimeError):
    """Sanitized provider failure that is safe to map to an HTTP response."""

    def __init__(self, provider: str, operation: str, message: str, status_code: int):
        super().__init__(message)
        self.provider = provider
        self.operation = operation
        self.public_message = message
        self.status_code = status_code


class ProviderUnavailableError(ProviderError):
    def __init__(self, provider: str, operation: str):
        super().__init__(
            provider,
            operation,
            f"{operation.capitalize()} service is temporarily unavailable; please retry.",
            503,
        )


class ProviderOutputError(ProviderError):
    def __init__(self, provider: str, operation: str):
        super().__init__(
            provider,
            operation,
            f"{operation.capitalize()} service returned an invalid response; please retry.",
            502,
        )
