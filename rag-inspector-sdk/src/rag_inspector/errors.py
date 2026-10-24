class RAGInspectorError(Exception):
    """Base class for SDK errors."""


class InvalidQuestionError(RAGInspectorError, ValueError):
    """The pipeline question is missing or invalid."""


class MissingQuestionError(InvalidQuestionError):
    """A required question was not provided."""


class InvalidRetrievalError(RAGInspectorError, ValueError):
    """The retrieval payload is missing or invalid."""


class MissingRetrievalError(InvalidRetrievalError):
    """Retrieval data has not been recorded."""


class InvalidDocumentError(InvalidRetrievalError):
    """A retrieved document cannot be normalised."""


class InvalidAnswerError(RAGInspectorError, ValueError):
    """The generated answer is missing or invalid."""


class MissingGenerationError(InvalidAnswerError):
    """Generation data has not been recorded."""


class MissingPipelineDataError(RAGInspectorError):
    """Raised when a run is finalised before required pipeline stages are recorded."""

    def __init__(self, missing_fields):
        self.missing_fields = tuple(missing_fields)
        details = ", ".join(self.missing_fields)
        super().__init__(f"Missing required pipeline data: {details}.")


class RunStateError(RAGInspectorError, RuntimeError):
    """The operation is not valid for the run's current state."""
