"""
Modul pro zpracování a načítání dat.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Načte data z CSV souboru.
    
    Args:
        filepath: Cesta k CSV souboru
        
    Returns:
        DataFrame s načtenými daty
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vyčistí data - ošetří chybějící hodnoty, outliers, atd.
    
    Args:
        df: Původní DataFrame
        
    Returns:
        Vyčištěný DataFrame
    """
    # TODO: Implementovat čištění dat
    df_clean = df.copy()
    
    # Odstranit duplicity
    df_clean = df_clean.drop_duplicates(subset=['date'])
    
    # Seřadit podle data
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    return df_clean


def split_data(df: pd.DataFrame, test_year: int = 2025) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rozdělí data na trénovací a testovací podle roku.
    
    Args:
        df: DataFrame s daty
        test_year: Rok pro testovací data
        
    Returns:
        Tuple (train_df, test_df)
    """
    train_df = df[df['date'].dt.year < test_year].copy()
    test_df = df[df['date'].dt.year >= test_year].copy()
    
    return train_df, test_df


if __name__ == "__main__":
    # Test funkcí
    df = load_data('../data/raw/techmania_cleaned_master.csv')
    print(f"Načteno {len(df)} záznamů")
    print(df.head())
