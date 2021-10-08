from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted
import xgboost as xgb
import matplotlib.pyplot as plt
import lightgbm as lgb
import catboost as cb
from sklearn import metrics

class PSI(BaseEstimator, TransformerMixin):


    """
    Вычисление PSI и отбор признаков на их основе.

    Parameters
    ----------
    threshold: float
        Порог для отбора переменных по PSI.
        Если PSI для переменной выше порога - переменная макрируется
        0 (не использовать для дальнейшего анализа), если ниже
        порога - маркируется 1 (использовать для дальнейшего анализа).

    categorical_features: List[str], optional, default = None
        Список категориальных признаков для анализа.
        Опциональный параметр, по умолчанию, не используется, т.е.
        категориальные признаки отсутствуют.

    n_bins: int, optional, default = 20
        Количество бинов, на которые разбивается выборка.

    min_value: float, optional, default = 0.005
        Значение которое используется, если рассчитанный psi = 0.

    bin_type: string, optional, default = "quanitles"
        Способ разбиения на бины: "quantiles" or "bins".
        При выборе "quantiles" - выборка будет разбита на n_bins
        квантилей, при выборке "bins" - выборка будет разбита на
        n_bins бакетов с равным шагом между бакетами.
        Иные значения приводят к возникновению ValueError.

    Attributes
    ----------
    scores_: Dict[str, float]
        Словарь со значениями PSI,
        ключ словаря - название признака, значение - PSI-score.

    """
    def __init__(self,
                 threshold: float,
                 categorical_features: Optional[List[str]] = None,
                 bin_type: str = "quantiles",
                 min_value: float = 0.005,
                 n_bins: int = 20):

        self.threshold = threshold
        self.categorical_features = categorical_features
        self.min_value = min_value
        self.n_bins = n_bins
        if bin_type in ["quantiles", "bins"]:
            self.bin_type = bin_type
        else:
            raise ValueError(
                "Incorrect bin_type value. Expected 'quantiles' or 'bins', "
                f"but {bin_type} is transferred."
            )
        self.scores = {}

    def calculate_bins(self, data: pd.Series) -> np.array:
        """
        Вычисление границ бинов для разбиения выборки.

        Parameters
        ----------
        data: pandas.Series, shape = [n_samples, ]
            наблюдения из train-выборки.

        Returns
        -------
        bins: numpy.array, shape = [self.n_bins + 1]
            Список с границами бинов.

        """
        if self.bin_type == "quantiles":
            bins = np.linspace(0, 100, self.n_bins + 1)
            bins = [np.nanpercentile(data, x) for x in bins]

        else:
            bins = np.linspace(data.min(), data.max(), self.n_bins + 1)

        return np.unique(bins)

    def calculate_psi_in_bin(self, expected_score, actual_score) -> float:
        """
        Вычисление значения psi для одного бакета.

        Осуществляется проверка на равенство нулю expected_score и
        actual_score: если один из аргументов равен нулю, то его
        значение заменяется на self.min_value.

        Parameters
        ----------
        expected_score: float
            Ожидаемое значение.

        actual_score: float
            Наблюдаемое значение.

        Returns
        -------
        value: float
            Значение psi в бине.

        """
        if expected_score == 0:
            expected_score = self.min_value
        if actual_score == 0:
            actual_score = self.min_value

        value = (expected_score - actual_score)
        value = value * np.log(expected_score / actual_score)

        return value

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, bins) -> float:
        """
        Расчет PSI для одной переменной.

        Parameters
        ----------
        expected: pandas.Series, shape = [n_samples_e, ]
            Наблюдения из train-выборки.

        actual: pandas.Series, shape = [n_samples_o, ]
            Наблюдения из test-выборки.

        bins: pandas.Series, shape = [self.n_bins, ]
            Бины для расчета PSI.

        Returns
        -------
        psi_score: float
            PSI-значение для данной пары выборок.

        """
        expected_score = np.histogram(expected.fillna(-9999), bins)[0]
        expected_score = expected_score / expected.shape[0]

        actual_score = np.histogram(actual.fillna(-9999), bins)[0]
        actual_score = actual_score / actual.shape[0]

        psi_score = np.sum(
            self.calculate_psi_in_bin(exp_score, act_score)
            for exp_score, act_score in zip(expected_score, actual_score)
        )

        return psi_score

    def calculate_numeric_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """
        Вычисление PSI для числовой переменной.

        Parameters
        ----------
        expected: pandas.Series, shape = [n_samples_e, ]
            Наблюдения из train-выборки.

        actual: pandas.Series, shape = [n_samples_o, ]
            Наблюдения из test-выборки.

        Returns
        -------
        psi_score: float
            PSI-значение для данной пары выборок.

        """
        bins = self.calculate_bins(expected)
        psi_score = self.calculate_psi(expected, actual, bins)
        return psi_score

    def calculate_categorical_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """
        Вычисление PSI для категориальной переменной.
        PSI рассчитывается для каждого уникального значения категории.

        Parameters
        ----------
        expected: pandas.Series, shape = [n_samples_e, ]
            Наблюдения из train-выборки.

        actual: pandas.Series, shape = [n_samples_o, ]
            Наблюдения из test-выборки.

        Returns
        -------
        psi_score: float
            PSI-значение для данной пары выборок.

        """
        bins = np.unique(expected).tolist()
        psi_score = self.calculate_psi(expected, actual, bins)
        return psi_score

    def fit(self, X, y=None):
        """
        Вычисление PSI-значения для всех признаков.

        Parameters
        ----------
        X: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        y: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для тестирования.

        Returns
        -------
        self
        """
        missed_columns = list(set(X.columns) - set(y.columns))

        if missed_columns:
            raise MissedColumnError(
                f"Missed {list(missed_columns)} columns in data.")

        if self.categorical_features:
            numeric_features = list(
                set(X.columns) - set(self.categorical_features)
            )
            for feature in self.categorical_features:
                self.scores[feature] = self.calculate_categorical_psi(
                    X[feature], y[feature]
                )
        else:
            numeric_features = X.columns

        for feature in tqdm(numeric_features):
            self.scores[feature] = self.calculate_numeric_psi(
                X[feature], y[feature]
            )
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """
        Отбор переменных по self.threshold.
        Если PSI-score для переменной выше порога, то переменная
        помечается 0 (не использовать для дальнейшего анализа), если ниже
        порога - маркируется 1 (использовать для дальнейшего анализа).

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для тестирования.

        Returns
        -------
        scores: pandas.DataFrame, shape = [n_features, 3]
            Датафрейм с PSI-анализом переменных.

        """
        check_is_fitted(self, "scores")
        scores = pd.Series(self.scores)
        scores = pd.DataFrame({"Variable": scores.index, "PSI": scores.values})
        scores["Selected"] = np.where(scores.PSI < self.threshold, 1, 0)
        scores = scores.sort_values(by="PSI")

        mask = scores["Selected"] == 1
        self.used_features = scores.loc[mask, "Variable"].tolist()

        return scores.reset_index(drop=True)
        
        
        
        
def type_features(df):
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.to_list()# числовые значения
    count_numerical_features = len(numerical_features)
    int_features = [] # целые числа возможно дискретные или категориальные признаки
    for i in numerical_features:
        if sum(df[i].apply(lambda x : x%1))==0:
            int_features.append(i)
    count_int =len(int_features)
    no_int = list(set(numerical_features) - set(int_features))
    count_no_int = len(no_int)
    obj_features = df.select_dtypes(exclude=[np.number]).columns.to_list()
    count_obj_features = len(obj_features)
    year_feature = [feature for feature in df.columns.to_list()
                    if 'Yr' in feature or 'Year' in feature
                    ]
    count_year_feature = len(year_feature)
    print(f'датасет имеет размерность {df.shape[0]} строк и {df.shape[1]} признаков')
    print(f'числовых значений в датасете {count_numerical_features}')
    print(f'в том числе целочисленных {count_int}')
    print(f'в том числе не целочисленных {count_no_int}')
    print(f'объектных признаков {count_obj_features}')          
    print(f'признаков типа время возможно {count_year_feature}')
    
    return numerical_features, int_features, no_int, obj_features,year_feature     



def lgb_param_rnd_test(x,y,x_v,y_v,num):
    metrika = metrics.roc_auc_score
    res = pd.DataFrame()
    num_leaves = [2,5,7,8,10,12,15,25,30,40,50,55,60,45]
    max_depth = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    learning_rate = [0.05,0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
    n_estimators = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,900,1000]
    reg_alpha = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    reg_lambda = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    for i in range(num):
        params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        'num_leaves':np.random.choice(num_leaves),
        'max_depth':np.random.choice(max_depth),
        'learning_rate':np.random.choice(learning_rate),
        "metric": "auc",
        "n_jobs": 15,
        'reg_alpha':np.random.choice(reg_alpha),
        'reg_lambda':np.random.choice(reg_lambda),
         'n_estimators':np.random.choice(n_estimators) ,  
        "random_state": 27
                    }
        model = lgb.LGBMClassifier(**params)
        model.fit(x,y,early_stopping_rounds  = 20,\
            eval_set =(x_v,y_v),verbose = False)
        pred = model.predict_proba(x_v)[:,1]
        met = metrika(y_v,pred)
        params['auc'] = met
        res = res.append(params,ignore_index=True)
        
    return res.sort_values('auc',ascending=False)     
    
def xgb_param_rnd_test(x,y,x_v,y_v,num):
    metrika = metrics.roc_auc_score
    res = pd.DataFrame()
    min_child_weight = [2,5,7,8,10,12,15,25,30,40,50,55,60,45]
    max_depth = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    learning_rate = [0.05,0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
    n_estimators = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,900,1000]
    reg_alpha = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    gamma = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    reg_lambda = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    for i in range(num):
        params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        'min_child_weight':np.random.choice(min_child_weight),
        'max_depth':np.random.choice(max_depth),
        'learning_rate':np.random.choice(learning_rate),
        "eval_metric": "auc",
        "n_jobs": 15,
        'reg_alpha':np.random.choice(reg_alpha),
        'reg_lambda':np.random.choice(reg_lambda),
        'n_estimators':np.random.choice(n_estimators) ,  
        "random_state": 27,
         'gamma': np.random.choice(gamma) ,
          'use_label_encoder':False  
                    }
        model = xgb.XGBClassifier(**params)
        model.fit(x,y,early_stopping_rounds  = 20,\
            eval_set =[(x_v,y_v)],verbose = False)
        pred = model.predict_proba(x_v)[:,1]
        met = metrika(y_v,pred)
        params['auc'] = met
        res = res.append(params,ignore_index=True)
        
    return res.sort_values('auc',ascending=False)     
    
    
def cb_param_rnd_test(x,y,x_v,y_v,num):
    metrika = metrics.roc_auc_score
    res = pd.DataFrame()
    n_estimators = [50,100,200,300,400,500,600,700,1000]
    min_child_samples = [5,10,15,20,25,30,35,40,45,50]
    max_depth = [2,3,4,5,6,7,8,9,10,11,12]
    learning_rate = [0.05,0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]  
    l2_leaf_reg = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    max_bin=[5,10,25,30,35,40,45,50,55,60]
    reg_lambda = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    for i in range(num):
        params = {
        "loss_function": "Logloss",
        "task_type": "GPU",
        'devices':'0',     
        'verbose':False , 
        "eval_metric": "AUC",
        "thread_count": 15, 
        "early_stopping_rounds": 20,
        "random_seed": 27,    
        'max_depth':np.random.choice(max_depth),      
        'n_estimators':np.random.choice(n_estimators) ,
        'learning_rate':np.random.choice(learning_rate), 
        'l2_leaf_reg':np.random.choice(l2_leaf_reg),   
        'reg_lambda':np.random.choice(reg_lambda),   
        'min_child_samples':np.random.choice(min_child_samples),    
        'max_bin': np.random.choice(max_bin) 
                        }
        model = cb.CatBoostClassifier(**params)
        model.fit(x,y,early_stopping_rounds  = 20,\
            eval_set =[(x_v,y_v)],verbose = False)
        pred = model.predict_proba(x_v)[:,1]
        met = metrika(y_v,pred)
        params['auc'] = met
        res = res.append(params,ignore_index=True)
    return res.sort_values('auc',ascending=False)     
    
