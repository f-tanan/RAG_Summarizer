from langchain_ibm import WatsonxLLM
from Configuration import Config


def _validate_config():
    missing = []

    for key in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "MODEL_ID"):
        value = getattr(Config, key, None)
        if not value:
            missing.append(key)

    if missing:
        raise RuntimeError(
            "Missing required Watsonx configuration values: "
            + ", ".join(missing)
            + ". Please set the corresponding environment variables before running."
        )

    if not isinstance(Config.PARAMETERS, dict):
        raise RuntimeError("Config.PARAMETERS must be a dictionary of Watsonx generation parameters.")


def get_llm():
    """Create and return a configured WatsonxLLM instance.

    Uses configuration values from the Config class to initialize the IBM
    WatsonxLLM client with model_id, url, apikey, project_id, and params.

    Returns:
        WatsonxLLM: A ready-to-use language model client.
    """
    _validate_config()

    try:
        llm = WatsonxLLM(
            model_id=Config.MODEL_ID,
            url=Config.WATSONX_URL,
            apikey=Config.WATSONX_APIKEY,
            project_id=Config.WATSONX_PROJECT_ID,
            params=Config.PARAMETERS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize WatsonxLLM with model '{Config.MODEL_ID}': {exc}"
        ) from exc

    return llm