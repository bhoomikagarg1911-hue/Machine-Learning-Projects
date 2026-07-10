import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data={
    'Hours':[1,2,3,4,5],
    'Pass':[0,0,0,1,1]
    }
df=pd.DataFrame(data)
X=df[['Hours']]
Y=df['Pass']
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
    )
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
prediction=model.predict([[3.5]])
print("Predicted Result:",prediction)
print("hello")