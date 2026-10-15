"""
Feedback processing functions.
"""

import pandas as pd

def add_feedback(df: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    """Add feedback information to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe containing task data.
        exp_name (str): Name of the experiment.
        
    Returns:
        pd.DataFrame: DataFrame with added feedback information.
    """
    # For now, just return the dataframe as is
    # We can add more sophisticated feedback processing later if needed
    return df 