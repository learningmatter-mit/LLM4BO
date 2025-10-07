"""
Utility functions for SMILES strings.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def smiles_to_fingerprint(smiles: str, radius: int = 4, nBits: int = 4096) -> np.ndarray:
    """Convert SMILES string to Morgan fingerprint.
    
    Args:
        smiles: SMILES string
        radius: Radius of the Morgan fingerprint
        nBits: Number of bits in the fingerprint
    Returns:
        Morgan fingerprint as numpy array
    """ 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
        
    morgan_gen = GetMorganGenerator(radius=2, fpSize=nBits, countSimulation=False)
    fp = morgan_gen.GetFingerprint(mol)
    
    fp_array = np.array(fp, dtype=np.float32)
    return fp_array

def compact_df(df, index=True, sig_digits=3):
        """
        Ultra-compact DataFrame display for token efficiency.
        Maintains structure while minimizing tokens.
        
        Args:
            df: DataFrame to display
            index: Whether to include row indices (default True)
            sig_digits: Significant digits for numeric columns (default 3)
        
        Returns:
            str: Compact string representation
        """
        def format_num(x):
            if pd.isna(x):
                return 'nan'
            if isinstance(x, (int, np.integer)):
                return str(x)
            if isinstance(x, (float, np.floating)):
                # Format to sig_digits significant figures
                if x == 0:
                    return '0'
                # Use scientific notation if very large/small, otherwise decimal
                if abs(x) >= 10**(sig_digits) or (abs(x) < 10**(-sig_digits+1) and x != 0):
                    return f"{x:.{sig_digits-1}e}"
                else:
                    # Round to sig_digits significant figures
                    return f"{x:.{max(0, sig_digits - 1 - int(np.floor(np.log10(abs(x)))))}f}".rstrip('0').rstrip('.')
            return str(x)
        
        # Get column names - use shortest reasonable abbreviations
        cols = list(df.columns)
        
        # Format header
        if index:
            header = "index|" + "|".join(cols)
        else:
            header = "|".join(cols)
        
        # Format rows
        rows = []
        for idx, row in df.iterrows():
            formatted_row = []
            if index:
                formatted_row.append(str(idx))
            
            for col in cols:
                val = row[col]
                if pd.api.types.is_numeric_dtype(type(val)):
                    formatted_row.append(format_num(val))
                else:
                    # For strings, keep as-is (mainly SMILES)
                    formatted_row.append(str(val))
            
            rows.append("|".join(formatted_row))
        
        return header + "\n" + "\n".join(rows)