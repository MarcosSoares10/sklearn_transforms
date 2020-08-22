from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
 
# Retorna um novo dataframe com valor nulo igual a média dos demais values nas colunas escolhidas
class RecalculateNullNumericValues(BaseEstimator, TransformerMixin):
    def __init__(self, InitialNumericColumn,FinalNumericColumn):
        self.InitialNumericColumn = InitialNumericColumn
        self.FinalNumericColumn = FinalNumericColumn
    
    def fit(self,X,y=None):
        return self

    def transform(self, X):
        data = X.copy()
        values = data.loc[:,self.InitialNumericColumn:self.FinalNumericColumn]
        values = values.T.fillna(round(values.mean(axis=1),1)).T
        data.loc[:,self.InitialNumericColumn:self.FinalNumericColumn] = values
        return data

# Retorna um novo dataframe com dados balanceados, para isso é verificado se a melhor opção é o Oversampling ou Downsampling
class BalanceClasses(BaseEstimator, TransformerMixin):
    def oversample(self, classe, n_samples):
        return resample(classe, replace=True, n_samples=n_samples, random_state=42)
        
    def undersample(self, classe, n_samples):
        return resample(classe, replace=False, n_samples=n_samples, random_state=42 )
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        target = []
        freq = data.PERFIL.value_counts()
        mean_avg = int( freq.mean() )
        resampled = []
        
        for value in data.PERFIL.unique():
            target.append( data[data.PERFIL == value] )
            if target[-1].shape[0] > mean_avg:
                resampled.append( self.undersample( target[-1], mean_avg ))
            else:
                resampled.append( self.oversample( target[-1], mean_avg ))
        
        return pd.concat(resampled)

    
# Transformação Clamp para garantir valores de notas entre 0-10
class ClampClasses(BaseEstimator, TransformerMixin):
    def __init__(self, InitialNumericColumn,FinalNumericColumn):
        self.InitialNumericColumn = InitialNumericColumn
        self.FinalNumericColumn = FinalNumericColumn
        
    def fit(self, X, y=None):
          return self

    def transform(self, X):
          data = X.copy()
          values = data.loc[:,self.InitialNumericColumn:self.FinalNumericColumn]
          values = values.clip(0.0,10.0)
          data.loc[:,self.InitialNumericColumn:self.FinalNumericColumn] = values
          return data
