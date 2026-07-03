import os
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
load_dotenv()


class Config:
    # IBM Watsonx credentials
    WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_URL = os.getenv("WATSONX_URL")

    # Model IDs
    MODEL_ID = 'ibm/granite-3-8b-instruct'

    PARAMETERS = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
    }

    #Corpus URL for the text data
    CORPUS_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'