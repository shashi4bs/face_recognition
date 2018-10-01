import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from DB_handler import ModelDatabase

def train_model_and_store() :
    x = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
    y = np.array([2, 4, 6, 8, 10]).reshape(-1,1)
    z = np.array([2.5, 5.0, 7.5, 10.0, 12.5]) 

    linear_model1 = LinearRegression()
    linear_model1.fit(x, y)
    
    linear_model2 = LinearRegression()
    linear_model2.fit(x,z)

    print("For the actual model trained : ")
    print("linear_model1(7) : ")
    print(linear_model1.predict(7))
    print("linear_model2(8) : ")
    print(linear_model2.predict(8))

    # store into the database
    model_db = ModelDatabase('Database1')
    model_db.clear_database()
    model_db.store_user_model('mdl_1', linear_model1)
    model_db.store_user_model('mdl_2', linear_model2)
    print()

def display_stored_model() :

    model_dbx = ModelDatabase('Database1')
    users = model_dbx.get_list_of_users()
    print("Users are : ")
    for user in users :
        print(user)
    print()

def retrieve_model_and_predict() :
    another_model_db = ModelDatabase('Database1')
    trained_model1 = another_model_db.get_user_model('mdl_1')
    trained_model2 = another_model_db.get_user_model('mdl_2')
    print("For linear_model1 retreived from database ")
    print(trained_model1.predict(7))
    print("For linear_model2(8) retrieved from database ")
    print(trained_model2.predict(8))
    print()


def main() :
    train_model_and_store()
    display_stored_model()
    retrieve_model_and_predict()

if __name__ == '__main__' :
    main()
