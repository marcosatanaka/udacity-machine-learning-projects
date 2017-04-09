# -*- coding: utf-8 -*-

# Este é um problema de classificação, pois estamos classificando os alunos em duas categorias: aqueles que precisam de intervenção antecipada antes de serem reprovados, e aqueles que não precisam. Um problema de regressão possui apenas uma saída contínua, um número, e nestes tipos de problema, tentamos encontrar a linha que melhor descreve o padrão dos dados que estamos vendo.

# Importar bibliotecas
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Ler os dados dos estudantes
student_data = pd.read_csv("student-data.csv")
print "Os dados dos estudantes foram lidos com exito!"



# TODO: Calcule o numero de estudante
n_students = len(student_data)

# TODO: Calcule o numero de atributos
n_features = len(student_data.columns) - 1

# TODO: Calcule o numero de alunos aprovados
n_passed = len(student_data.loc[student_data['passed'] == 'yes'])

# TODO: Calcule o numero de alunos reprovados
n_failed = len(student_data.loc[student_data['passed'] == 'no'])

# TODO: Calcule a taxa de graduacao
grad_rate = float(n_passed) / n_students

# Imprima os resultados
# print "Numero total de estudantes: {}".format(n_students)
# print "Numero de atributos: {}".format(n_features)
# print "Numero de estudantes aprovados: {}".format(n_passed)
# print "Numero de estudantes reprovados: {}".format(n_failed)
# print "Taxa de graduacao: {:.2f}%".format(grad_rate)



# Extraia as colunas dos atributo
feature_cols = list(student_data.columns[:-1])

# Extraia a coluna-alvo, 'passed'
target_col = student_data.columns[-1] 

# Mostre a lista de colunas
# print "Colunas de atributos:\n{}".format(feature_cols)
# print "\nColuna-alvo: {}".format(target_col)

# Separe os dados em atributos e variaveis-alvo (X_all e y_all, respectivamente)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Mostre os atributos imprimindo as cinco primeiras linhas
# print "\nFeature values:"
# print X_all.head()



def preprocess_features(X):
    # Pre-processa os dados dos estudantes e converte as variaveis binarias nao numericas em
    # variaveis binarias (0/1). Converte variaveis categoricas em variaveis posticas.
    
    # Inicialize nova saida DataFrame
    output = pd.DataFrame(index = X.index)

    # Observe os dados em cada coluna de atributos 
    for col, col_data in X.iteritems():
        
        # Se o tipo de dado for nao numerico, substitua todos os valores yes/no por 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # Se o tipo de dado for categorico, converta-o para uma variavel dummy
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
		
		# Reuna as colunas revisadas
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
# print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))



# TODO: Importe qualquer funcionalidade adicional de que você possa precisar aqui
from sklearn.cross_validation import train_test_split

# TODO: Embaralhe e distribua o conjunto de dados de acordo com o número de pontos de treinamento e teste abaixo
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.24)

# Mostre o resultado da distribuição
print "O conjunto de treinamento tem {} amostras.".format(X_train.shape[0])
print "O conjunto de teste tem {} amostras.".format(X_test.shape[0])


def train_classifier(clf, X_train, y_train):
	''' Ajusta um classificador para os dados de treinamento. '''

	# Inicia o relógio, treina o classificador e, então, para o relógio
	start = time()
	clf.fit(X_train, y_train)
	end = time()

	# Imprime os resultados
	print "[{:.4f}s]".format(end - start)

def predict_labels(clf, features, target):
	''' Faz uma estimativa utilizando um classificador ajustado baseado na pontuação F1. '''

	# Inicia o relógio, faz estimativas e, então, o relógio para
	start = time()
	y_pred = clf.predict(features)
	end = time()

	# Imprime os resultados de retorno
	print "[{:.4f}s]".format(end - start)
	return f1_score(target.values, y_pred, pos_label='yes')

def train_predict(clf, X_train, y_train, X_test, y_test):
	''' Treina e faz estimativas utilizando um classificador baseado na pontuação do F1. '''

	# Indica o tamanho do classificador e do conjunto de treinamento
	print "Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train))

	# Treina o classificador
	train_classifier(clf, X_train, y_train)

	# Imprime os resultados das estimativas de ambos treinamento e teste
	print "Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, y_train, False))
	print "Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, y_test, True))


# TODO: Importe os três modelos de aprendizagem supervisionada do sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

# TODO: Inicialize os três modelos
clf_A = GaussianNB()
clf_B = SVC(random_state=1)
clf_C = AdaBoostClassifier(random_state=1)
clf_D = DecisionTreeClassifier(random_state=1)
clf_E = KNeighborsClassifier()
clf_F = SGDClassifier(random_state=1)

# TODO: Configure os tamanho dos conjuntos de treinamento
X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]

X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]

X_train_300 = X_train[0:300]
y_train_300 = y_train[0:300]

# TODO: Executar a função 'train_predict' para cada classificador e cada tamanho de conjunto de treinamento
'''
print ""
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
print ""
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
print ""
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print ""

print ""

print ""
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
print ""
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
print ""
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print ""

print ""

print ""
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
print ""
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
print ""
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)
print ""

print ""

print ""
train_predict(clf_D, X_train_100, y_train_100, X_test, y_test)
print ""
train_predict(clf_D, X_train_200, y_train_200, X_test, y_test)
print ""
train_predict(clf_D, X_train_300, y_train_300, X_test, y_test)
print ""

print ""

print ""
train_predict(clf_E, X_train_100, y_train_100, X_test, y_test)
print ""
train_predict(clf_E, X_train_200, y_train_200, X_test, y_test)
print ""
train_predict(clf_E, X_train_300, y_train_300, X_test, y_test)
print ""

print ""

print ""
train_predict(clf_F, X_train_100, y_train_100, X_test, y_test)
print ""
train_predict(clf_F, X_train_200, y_train_200, X_test, y_test)
print ""
train_predict(clf_F, X_train_300, y_train_300, X_test, y_test)
print ""

print ""
'''


# TODO: Importe 'GridSearchCV' e 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Crie a lista de parâmetros que você gostaria de calibrar
parameters = {'kernel' : ('linear', 'rbf'), 'C' : np.arange(0.1, 1, 0.1), 'gamma' : ['auto', 0, 0.5, 1.0, 1.5, 2.0]}

# TODO: Inicialize o classificador
clf = SVC(random_state=1)

# TODO: Faça uma função de pontuação f1 utilizando 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO: Execute uma busca em matriz no classificador utilizando o f1_scorer como método de pontuação
grid_obj = GridSearchCV(clf, parameters, scoring = f1_scorer)

# TODO: Ajuste o objeto de busca em matriz para o treinamento de dados e encontre os parâmetros ótimos
grid_obj = grid_obj.fit(X_train, y_train)

print("Melhores parâmetros:")
print(grid_obj.best_params_)

# Get the estimator
clf = grid_obj.best_estimator_

# Reporte a pontuação final F1 para treinamento e teste depois de calibrar os parâmetrosprint "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "O modelo calibrado tem F1 de {:.4f} no conjunto de treinamento.".format(predict_labels(clf, X_train, y_train))
print "O modelo calibrado tem F1 de {:.4f} no conjunto de teste.".format(predict_labels(clf, X_test, y_test))

