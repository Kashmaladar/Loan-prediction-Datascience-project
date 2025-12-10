import streamlit as st
import pandas as pd
import sys

from src.LoanPrediction.logger import logging
from src.LoanPrediction.exception import CustomException

if __name__ == "__main__":
   logging.info("The excecution has started")


try:
   a=1/0
except Exception as e:
   logging.info("Custom Exception")
   raise CustomException(e,sys)