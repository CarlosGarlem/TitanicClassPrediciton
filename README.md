## Predicción sobrevivientes titanic
El siguiente proyecto presenta un método de ensemble learning que busca aplicar 4 modelos de clasificación (DecisionTree, SVM, NaiveBayes, Logistic Regression) para determinar si un pasajero sobrevive o no al hundimiento del Titanic. 

### Estructura
Los archivos del proyecto se encuentran distribuidos de la siguiente manera:
**Modelo y funciones**
- **TrainingModel.ipynb**. Contiene la lógica principal del proyecto y la primera parte descrita en el enunciado para el entrenamiento y configuración de los modelos
- **PredictionModel.ipynb**. Contiene una simulación de 10 predicciones para el ensemble de los modelos, corresponde a la segunda parte del enunciado
- **auxiliaryFuntions.py**. Dado que algunas funciones se usan de forma recurrente entre distintos notebooks, se creo este script el cual contiene diversas funciones utilizadas para reutilizar código a lo largo del proyecto y tener un flujo en los notebooks principales más ordenado. 
- **ResearchEssayNotebook.ipynb**. Contiene en ensayo e investigación correspondiente a la parte de K-Folds Cross Validation, así como el ensayo de SVM

**Carpetas con Archivos adicionales**
- **results**. Esta carpeta contiene la bitácora utilizada para llevar registro de los experimentos realizados para cada uno de los modelos
- **models**. Contiene los archivos generados al almacenar los vectores de pesos, o información utilizada para recrear los modelos
- **graphs**. Contiene la información generada al utilizar tensorboard
- **imgs**. Imágenes utilizadas para ejemplificar escenarios en el markdown de los notebooks