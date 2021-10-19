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
from typing import List, Tuple
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.model_selection import TimeSeriesSplit


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
    num_leaves = [2,5,7,8,10,12,15,25,30,40,50,55,60,45,70]
    max_depth = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    learning_rate = [0.05,0.01,0.1,0.15,0.2,0.25,0.3,0.35]
    n_estimators = [1000]
    reg_alpha = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60,100,200,300]
    reg_lambda = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60,100,200,300]
    for i in tqdm(range(num)):
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
        model.fit(x,y,early_stopping_rounds  = 50,\
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
    max_depth = [2,3,4,5,6,7,8,9,10,11,12,13,14]
    learning_rate = [0.05,0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
    n_estimators = [1000]
    reg_alpha = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    gamma = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    reg_lambda = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60]
    for i in tqdm(range(num)):
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
        model.fit(x,y,early_stopping_rounds  = 50,\
            eval_set =[(x_v,y_v)],verbose = False)
        pred = model.predict_proba(x_v)[:,1]
        met = metrika(y_v,pred)
        params['auc'] = met
        res = res.append(params,ignore_index=True)
        
    return res.sort_values('auc',ascending=False)     
    
    
def cb_param_rnd_test(x,y,x_v,y_v,num,cat_columns=[]):
    metrika = metrics.roc_auc_score
    res = pd.DataFrame()
    n_estimators = [1000]
    min_child_samples = [5,10,15,20,25,30,35,40,45,50,60]
    max_depth = [2,3,4,5,6,8,10,12,14]
    learning_rate = [0.05,0.01,0.1,0.15,0.2,0.25,0.3,0.35]  
    l2_leaf_reg = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60,100,150,200]
    max_bin=[5,10,25,30,35,40,45,50,55,60]
    reg_lambda = [0,0.5,1,2,3,4,5,6,8,10,12,15,20,25,30,40,45,50,60,100,150,200]
    for i in tqdm(range(num)):
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
        'min_child_samples':np.random.choice(min_child_samples),    
        'max_bin': np.random.choice(max_bin) 
                }
        if  len(cat_columns)!=0:
            params['cat_features'] = cat_columns
            
        model = cb.CatBoostClassifier(**params)    
        model.fit(x,y,early_stopping_rounds  = 50,\
            eval_set =[(x_v,y_v)],verbose = False,)
        pred = model.predict_proba(x_v)[:,1]
        met = metrika(y_v,pred)
        params['auc'] = met
        res = res.append(params,ignore_index=True)
    return res.sort_values('auc',ascending=False)    


def bootstrap_calculate_confidence_interval(y_true,y_pred,metric, n_samples = 1000,conf_interval = 0.95):
        """
        Вычисление доверительного интервала.
        Из  бутстреп-выборок образумых из предсказний
        валидационной выьорки

        y_true: np.array
            Вектор целевой переменной.

        y_pred: np.array
            Вектор прогнозов.

        metric: callable
            Функция для вычисления метрики.
            Функция должна принимать 2 аргумента: y_true, y_pred.

        n_samples: int, optional, default = 1000
            Количество создаваемых бутстреп выборок.
            Опциональный параметр, по умолчанию, равен 1000.

         Returns
        -------
        conf_interval: Tuple[float]
            Кортеж с границами доверительного интервала.

        """
   
        def create_bootstrap_samples(data, n_samples = n_samples):  
            bootstrap_idx = np.random.randint(
            low=0, high=len(data), size=(n_samples, len(data))
                    )
            return bootstrap_idx
        
        def create_bootstrap_metrics(y_true,y_pred,metric=metrics,n_samlpes = n_samples):
      
            scores = []
            if isinstance(y_true, pd.Series):
                y_true = y_true.values

            bootstrap_idx = create_bootstrap_samples(y_true)
            for idx in bootstrap_idx:
                y_true_bootstrap = y_true[idx]
                y_pred_bootstrap = y_pred[idx]

                score = metric(y_true_bootstrap, y_pred_bootstrap)
                scores.append(score)

            return scores

        def calculate_confidence_interval(scores, conf_interval= conf_interval):
   
            left_bound = np.percentile(
                scores, ((1 - conf_interval) / 2) * 100
            )
            right_bound = np.percentile(
                scores, (conf_interval + ((1 - conf_interval) / 2)) * 100
            )

            return left_bound, right_bound
        
        scores = create_bootstrap_metrics(y_true,y_pred,metric, n_samples)
        result = calculate_confidence_interval(scores, conf_interval= conf_interval)
        return result


def create_bootstrap_samples(data: np.array, n_samples: int = 1000) -> np.array:
    """
    Создание бутстреп-выборок.

    Parameters
    ----------
    data: np.array
        Исходная выборка, которая будет использоваться для
        создания бутстреп выборок.

    n_samples: int, optional, default = 1000
        Количество создаваемых бутстреп выборок.
        Опциональный параметр, по умолчанию, равен 1000.

    Returns
    -------
    bootstrap_idx: np.array
        Матрица индексов, для создания бутстреп выборок.

    """
    bootstrap_idx = np.random.randint(
        low=0, high=len(data), size=(n_samples, len(data))
    )
    return bootstrap_idx


def create_bootstrap_metrics(y_true: np.array,
                             y_pred: np.array,
                             metric: callable,
                             n_samlpes: int = 1000) -> List[float]:
    """
    Вычисление бутстреп оценок.

    Parameters
    ----------
    y_true: np.array
        Вектор целевой переменной.

    y_pred: np.array
        Вектор прогнозов.

    metric: callable
        Функция для вычисления метрики.
        Функция должна принимать 2 аргумента: y_true, y_pred.

    n_samples: int, optional, default = 1000
        Количество создаваемых бутстреп выборок.
        Опциональный параметр, по умолчанию, равен 1000.

    Returns
    -------
    bootstrap_metrics: List[float]
        Список со значениями метрики качества на каждой бустреп выборке.

    """
    scores = []

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    bootstrap_idx = create_bootstrap_samples(y_true)
    for idx in bootstrap_idx:
        y_true_bootstrap = y_true[idx]
        y_pred_bootstrap = y_pred[idx]

        score = metric(y_true_bootstrap, y_pred_bootstrap)
        scores.append(score)

    return scores


def calculate_confidence_interval(scores: list, conf_interval: float = 0.95) -> Tuple[float]:
    """
    Вычисление доверительного интервала.

    Parameters
    ----------
    scores: List[float / int]
        Список с оценками изучаемой величины.

    conf_interval: float, optional, default = 0.95
        Уровень доверия для построения интервала.
        Опциональный параметр, по умолчанию, равен 0.95.

    Returns
    -------
    conf_interval: Tuple[float]
        Кортеж с границами доверительного интервала.

    """
    left_bound = np.percentile(
        scores, ((1 - conf_interval) / 2) * 100
    )
    right_bound = np.percentile(
        scores, (conf_interval + ((1 - conf_interval) / 2)) * 100
    )

    return left_bound, right_bound
        
def make_cross_validation(X,y,estimator, metric, cv_strategy):
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        estimator.fit(x_train, y_train, \
          eval_set=[(x_train, y_train),(x_valid,y_valid)],eval_metric = 'auc',early_stopping_rounds  = 50,verbose=None) 

        y_valid_pred = estimator.predict_proba(x_valid)[:,1]
        
        fold_train_scores.append(estimator.best_score_['training']['auc'])
        fold_valid_scores.append(estimator.best_score_['valid_1']['auc'])
        oof_predictions[valid_idx] = y_valid_pred

        msg = (
            f"Fold: {fold_number+1}, train-observations = {len(train_idx)}, "
            f"valid-observations = {len(valid_idx)}\n"
            f"train-score = {round(fold_train_scores[fold_number], 4)}, "
            f"valid-score = {round(fold_valid_scores[fold_number], 4)}" 
        )
        print(msg)
        print("="*69)
        estimators.append(estimator)

    oof_score = metric(y, oof_predictions)
    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions   
    
    
def make_cross_validation_gr(X,y,estimator, metric, cv_strategy,gr):
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y,gr)):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        estimator.fit(x_train, y_train, \
          eval_set=[(x_train, y_train),(x_valid,y_valid)],eval_metric = 'auc',early_stopping_rounds  = 50,verbose=None) 

        y_valid_pred = estimator.predict_proba(x_valid)[:,1]
        
        fold_train_scores.append(estimator.best_score_['training']['auc'])
        fold_valid_scores.append(estimator.best_score_['valid_1']['auc'])
        oof_predictions[valid_idx] = y_valid_pred

        msg = (
            f"Fold: {fold_number+1}, train-observations = {len(train_idx)}, "
            f"valid-observations = {len(valid_idx)}\n"
            f"train-score = {round(fold_train_scores[fold_number], 4)}, "
            f"valid-score = {round(fold_valid_scores[fold_number], 4)}" 
        )
        print(msg)
        print("="*69)
        estimators.append(estimator)

    oof_score = metric(y, oof_predictions)
    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions     



def make_cross_validation_cb(X,y,estimator, metric, cv_strategy):
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        estimator.fit(x_train, y_train, \
          eval_set=[(x_train, y_train),(x_valid,y_valid)],early_stopping_rounds  = 50,verbose=None) 

        y_valid_pred = estimator.predict_proba(x_valid)[:,1]
        
        fold_train_scores.append(estimator.best_score_['validation_0']['AUC'])
        fold_valid_scores.append(estimator.best_score_['validation_1']['AUC'])
        oof_predictions[valid_idx] = y_valid_pred

        msg = (
            f"Fold: {fold_number+1}, train-observations = {len(train_idx)}, "
            f"valid-observations = {len(valid_idx)}\n"
            f"train-score = {round(fold_train_scores[fold_number], 4)}, "
            f"valid-score = {round(fold_valid_scores[fold_number], 4)}" 
        )
        print(msg)
        print("="*69)
        estimators.append(estimator)

    oof_score = metric(y, oof_predictions)
    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions 
