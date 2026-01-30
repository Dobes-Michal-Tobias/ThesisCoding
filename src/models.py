"""
Modul definující wrappery pro modely (Unsupervised i Supervised).
Zajišťuje jednotné rozhraní pro trénování a predikci.
"""
import config

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import MinCovDet, EmpiricalCovariance

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# --- M1: UNSUPERVISED MODELS ---

class BaseDetector:
    def fit(self, X):
        pass
    def decision_function(self, X):
        # Vrací skóre, kde VYŠŠÍ číslo = VĚTŠÍ anomálie
        pass

class MahalanobisDetector(BaseDetector):
    def __init__(self, method='robust', random_state=42):
        self.method = method
        self.cov_model = MinCovDet(random_state=random_state) if method == 'robust' else EmpiricalCovariance()
        
    def fit(self, X):
        # Fitujeme Gaussovské rozdělení na trénovací data
        self.cov_model.fit(X)
        return self

    def decision_function(self, X):
        # Mahalanobis distance (čím vyšší, tím anomálnější)
        dist = self.cov_model.mahalanobis(X)
        return dist # Vracíme přímo vzdálenost

class IsolationForestWrapper(BaseDetector):
    def __init__(self, contamination='auto', n_estimators=100, random_state=42):
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
        
    def fit(self, X):
        self.model.fit(X)
        return self
        
    def decision_function(self, X):
        # IF vrací decision_function, kde nižší (záporné) hodnoty jsou anomálie.
        # My chceme, aby vyšší hodnota = vyšší anomálie.
        # Proto otočíme znaménko: -score
        return -self.model.decision_function(X)

class OCSVMWrapper(BaseDetector):
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale'):
        # Cache size zvětšujeme pro rychlejší běh
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, cache_size=500)
        
    def fit(self, X):
        self.model.fit(X)
        return self
        
    def decision_function(self, X):
        # Stejně jako u IF, OCSVM vrací kladná čísla pro inliers.
        # Chceme skóre anomálie -> otočíme znaménko.
        return -self.model.decision_function(X)
    
def get_unsupervised_model(model_name, random_state=config.RANDOM_SEED):
    """
    Vrátí instanci unsupervised modelu podle názvu.
    Slouží jako rozcestník pro analytické skripty.
    """
    name = model_name.lower()
    
    if 'mahalanobis' in name or 'md' in name:
        return MahalanobisDetector(method='robust', random_state=random_state)
    
    elif 'isolation' in name or 'forest' in name or name == 'if':
        return IsolationForestWrapper(random_state=random_state)
    
    elif 'svm' in name or 'ocsvm' in name:
        # OCSVM nemá random_state v initu
        return OCSVMWrapper()
    
    else:
        raise ValueError(f"Unknown unsupervised model name: {model_name}")
    
# --- M2: SUPERVISED MODELS (Placeholder) ---
# Sem později přidáme např.:

def get_supervised_model(name, random_state=config.RANDOM_SEED):
    """
    Vrátí instanci klasifikátoru podle názvu.
    Obsahuje přednastavené parametry pro tento projekt (class_weight='balanced' atd.).
    """
    models = {
        "Dummy": DummyClassifier(strategy="stratified", random_state=random_state),
        
        "LogReg": LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=random_state
        ),
        
        "SVM (Lin)": SVC(
            kernel='linear', 
            class_weight='balanced', 
            probability=True,  # Nutné pro AUPRC/ROC
            random_state=random_state
        ),
        
        "SVM (RBF)": SVC(
            kernel='rbf', 
            class_weight='balanced', 
            probability=True, 
            random_state=random_state
        ),
        
        "NaiveBayes": GaussianNB(),
        
        "RandForest": RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            n_jobs=-1, 
            random_state=random_state
        ),
    }
    
    # XGBoost přidáme jen pokud je nainstalován
    if XGBClassifier:
        # scale_pos_weight se obvykle počítá dynamicky podle poměru tříd, 
        # zde necháme default nebo nastavíme natvrdo, pokud víme poměr.
        # Pro začátek necháme default, v loopu to případně můžeme upravit.
        models["XGBoost"] = XGBClassifier(
            eval_metric='logloss', 
            n_jobs=-1, 
            random_state=random_state,
            use_label_encoder=False
        )
    
    if name not in models:
        raise ValueError(f"Model '{name}' není definován. Dostupné: {list(models.keys())}")
        
    return models[name]