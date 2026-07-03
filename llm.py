from langchain_ibm import WatsonxLLM
from Configuration import  Config


def get_llm():
    """Create and return a configured WatsonxLLM instance.

    Uses configuration values from the Config class to initialize the IBM
    WatsonxLLM client with model_id, url, apikey, project_id, and params.

    Returns:
        WatsonxLLM: A ready-to-use language model client.
    """
    llm = WatsonxLLM(
        model_id=Config.model_id,
        url=Config.WATSONX_URL,
        apikey=Config.WATSONX_APIKEY,
        project_id=Config.WATSONX_PROJECT_ID,
        params=Config.parameters,
    )

    return llm