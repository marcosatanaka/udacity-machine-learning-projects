# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"





# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [85, 181, 183]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)





**Answer:**
- 0: Acredito que o primeiro estabelecimento escolhido seja uma distribuidora ou um mercado com foco em produtos de mercearia e limpeza, dado que seus gastos com produtos destas categorias é maior do que 1.000% acima da média (categoria Detergents_Paper).
- 1: O segundo provavelmente é um restaurante, dada a quantidade de produtos frescos (carnes, vegetais, frutas e verduras) adquiridos (~800% a mais na categoria Fresh).
- 2: E a terceira deve ser algo como uma cafeteria, dada a quantidade de leite, produtos congelados e delicatessen (~1.000% a mais em Frozen, ~3000% a mais em Delicatessen).





from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy(deep=True).drop(['Frozen'])

# TODO: Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data, data['Frozen'], test_size=0.25, random_state=1)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)