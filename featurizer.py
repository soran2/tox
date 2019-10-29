import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif


def atom_feature(i):
    """ Convert SMILES symbols. This function uses an approach similar to the one explained in
    this paper: [https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2523-5]
    """
    feature = [0]*27    
    if   i == '(' : feature[0]  = 1
    elif i == ')' : feature[1]  = 1
    elif i == '[' : feature[2]  = 1
    elif i == ']' : feature[3]  = 1
    elif i == '.' : feature[4]  = 1
    elif i == ':' : feature[5]  = 1
    elif i == '=' : feature[6]  = 1
    elif i == '#' : feature[7]  = 1
    elif i == '\\': feature[8]  = 1
    elif i == '/' : feature[9]  = 1 
    elif i == '@' : feature[10] = 1
    elif i == '+' : feature[11] = 1
    elif i == '-' : feature[12] = 1
    elif i == '1' : feature[13] = 1
    elif i == '2' : feature[14] = 1
    elif i == '3' : feature[15] = 1
    elif i == '4' : feature[16] = 1
    elif i == 'l' : feature[17] = 1
    elif i == 'r' : feature[18] = 1
    elif i == 'F' : feature[19] = 1
    elif i == 'S' : feature[20] = 1
    elif i == 'P' : feature[21] = 1
    elif i == 'I' : feature[22] = 1    
    elif i == 'C' : feature[23] = 1
    elif i == 'N' : feature[24] = 1
    elif i == 'O' : feature[25] = 1
    else: feature[26] = 1
    return(feature)

def mol_feature(mol):
    """ Create extended SMILES Matrix.
    """
    F = []
    for i in mol:
        f = atom_feature(i)
        F.extend(f)
    F.extend([0]*((400-len(mol))*27))
    return F

def all_grams(string, t1=2, t2=5):
    """ Create n-grams of given size.
    """
    res = []
    ns = [i for i in range(len(string)) if ((i>=t1) and (i<=t2))]    
    for n in ns:
        ngrams= zip(*[string[i:] for i in range(n)])
        res.extend([''.join(gram) for gram in ngrams])    
    return res

def featurize_smiles_ngram(smiles_list):
    """ Featurize n-grams.
    """
    vectorizer = CountVectorizer(min_df=1, analyzer=all_grams)
    X = vectorizer.fit_transform(smiles_list)    
    return({'feature_array':X, 'vectorizer':vectorizer})

def entopy_feature_importance(X, Y, vocab, feature_names):
    """ Detect important features.
    """    
    mutual_info = mutual_info_classif(X, Y, discrete_features=False, n_neighbors=3,copy=True,random_state=82)

    feature_importance = pd.DataFrame(data = {'Feature':feature_names,
                                              'MutualInfo':mutual_info})

    feature_importance['FeatureIndex'] = feature_importance.Feature.apply(lambda x: vocab[x])
    feature_importance = feature_importance.sort_values(by = 'MutualInfo', ascending=False).reset_index(drop=True)
    
    return(feature_importance)