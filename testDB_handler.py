import unittest
from DB_handler import *

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



class testModelDatabase(unittest.TestCase) :
    def setUp(self):
        # initialize a Image Database Object
        self.model_db = ModelDatabase('test')

        # for test purposes
        self.x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
        self.y = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)
        self.z = np.array([4,8,12,16,20,24,28,32,36,40]).reshape(-1,1)
        self.k = np.array([5,5,5,5,5,5,5,5,5,5]).reshape(-1,1)
        self.l = np.array([-2,-4,-6,-8,-10,-12,-14,-16,-18,-20]).reshape(-1,1)

        # y = 2x ; Assume id : '121A'
        self.regression_model1 = LinearRegression()
        self.regression_model1.fit(self.x, self.y)

        # z = 4x ; Assume id : '243'
        self.regression_model2 = LinearRegression()
        self.regression_model2.fit(self.x,self.z)

        # k = 5 ; Assume id : '392'
        self.regression_model3 = LinearRegression()
        self.regression_model3.fit(self.x,self.k)

        # l = -2x ; Assume id : '41A3'
        self.regression_model4 = LinearRegression()
        self.regression_model4.fit(self.x,self.l)

        

    def test_store_user_model_for_single_model(self) :
        self.model_db = ModelDatabase('test1')
        self.model_db.clear_database()

        # assert ImgaeDatabase Object does not contain our regression model
        self.assertEqual(len(self.model_db.get_list_of_users()), 0)
        self.assertTrue('243' not in self.model_db.get_list_of_users())


        # store regression model into our Image Database
        self.assertTrue(self.model_db.store_user_model('243', self.regression_model2))

        # assert That the regression Model is inserted
        self.assertEqual(len(self.model_db.get_list_of_users()), 1)
        self.assertTrue('243' in self.model_db.get_list_of_users())

        # asert False when we try to store the same regression model into our database
        self.assertFalse(self.model_db.store_user_model('243', self.regression_model2))
        self.assertEqual(len(self.model_db.get_list_of_users()), 1)
        self.assertTrue('243' in self.model_db.get_list_of_users())



    def test_store_user_model_for_multiple_models(self) :
        self.model_db = ModelDatabase('test2')
        self.model_db.clear_database()

        # assert that our ModelDatabase is empty
        self.assertEqual(len(self.model_db.get_list_of_users()), 0)

        # store 2 regression model into our database
        self.assertTrue(self.model_db.store_user_model('392', self.regression_model3))
        self.assertTrue(self.model_db.store_user_model('41A3', self.regression_model4))

        # assert that the stored models exist in the database
        self.assertEqual(len(self.model_db.get_list_of_users()), 2)
        self.assertTrue('392' in  self.model_db.get_list_of_users())
        self.assertTrue('41A3' in self.model_db.get_list_of_users())


        # assert that the regression_model1 with Id '121A' does not exist in the ImageDatabse
        self.assertTrue('121A' not in self.model_db.get_list_of_users())

        # Insert the regressio_model1 with id '121A' into the ImageDatabse Object
        self.assertTrue(self.model_db.store_user_model('121A', self.regression_model1))
        self.assertFalse(self.model_db.store_user_model('121A', self.regression_model1))
        
        # assert that the ModelDatabase now contain the three models that we inserted
        self.assertEqual(len(self.model_db.get_list_of_users()), 3)
        self.assertTrue('121A' in self.model_db.get_list_of_users())
        self.assertTrue('392' in  self.model_db.get_list_of_users())
        self.assertTrue('41A3' in self.model_db.get_list_of_users())


    def test_get_user_model(self) :
        self.model_db = ModelDatabase('test3')
        self.model_db.clear_database()

        # show regression_model2 works
        self.assertAlmostEqual(self.regression_model2.predict(4.5)[0][0], 18.0)
        self.assertAlmostEqual(self.regression_model2.predict(20.0)[0][0], 80.0)

        # show regression_model4 works
        self.assertAlmostEqual(self.regression_model4.predict(11.0)[0][0] ,-22.0)
        self.assertAlmostEqual(self.regression_model4.predict(15.0)[0][0] ,-30.0)
        
        # insert the two model into the ImageDatabse
        self.assertTrue(self.model_db.store_user_model('243', self.regression_model2))
        self.assertTrue(self.model_db.store_user_model('41A3', self.regression_model4))  

        # retrieve the two models as rg_model2 and rg_model4
        rg_model2 = self.model_db.get_user_model('243')
        rg_model4 = self.model_db.get_user_model('41A3')

        # assert that the retreived regression models are not none
        self.assertTrue(rg_model2 is not None)
        self.assertTrue(rg_model4 is not None)

        # assert that the retreived regression_model2 works
        self.assertAlmostEqual(rg_model2.predict(4.5)[0][0], 18.0)
        self.assertAlmostEqual(rg_model2.predict(20.0)[0][0], 80.0)
        self.assertAlmostEqual(rg_model2.predict(16.0)[0][0], 64.0)

        # assert that the retrieved regression_model4 works
        self.assertTrue(rg_model4.predict(11.0)[0][0], -22.0)
        self.assertTrue(rg_model4.predict(15.0)[0][0], -30.0)
        self.assertTrue(rg_model4.predict(2.5)[0][0], -5.0)

        # assert that a model that is not stored retreives a None Object
        self.assertTrue(self.model_db.get_user_model(523) is None)
        self.assertTrue(self.model_db.get_user_model('121A') is None)
        

    def test_get_list_of_users(self) :
        self.model_db = ModelDatabase('test4')
        self.model_db.clear_database()

        # assert that get_list_of_users initially returns an empty list
        self.assertEqual(len(self.model_db.get_list_of_users()), 0)
        self.assertTrue(type(self.model_db.get_list_of_users() is list))

        # store regression_models into our database
        self.model_db.store_user_model('392', self.regression_model3)
        self.model_db.store_user_model('41A3', self.regression_model4)
        self.model_db.store_user_model('121A', self.regression_model1)


        # assert that the stored models exist in the database
        self.assertEqual(len(self.model_db.get_list_of_users()), 3)
        self.assertTrue('392' in  self.model_db.get_list_of_users())
        self.assertTrue('41A3' in self.model_db.get_list_of_users())
        self.assertTrue('121A' in self.model_db.get_list_of_users())



    def test_remove_user_model(self) :
        self.model_db = ModelDatabase('test5')
        self.model_db.clear_database()

        # assert that our ModelDatabase is empty
        self.assertEqual(len(self.model_db.get_list_of_users()), 0)

        # store regression_models into our database
        self.model_db.store_user_model('392', self.regression_model3)
        self.model_db.store_user_model('41A3', self.regression_model4)
        self.model_db.store_user_model('121A', self.regression_model1)


        # assert that the stored models exist in the database
        self.assertEqual(len(self.model_db.get_list_of_users()), 3)
        self.assertTrue('392' in  self.model_db.get_list_of_users())
        self.assertTrue('41A3' in self.model_db.get_list_of_users())
        self.assertTrue('121A' in self.model_db.get_list_of_users())

        # remove regression_model4 from the ModelDatabase
        self.assertTrue(self.model_db.remove_user_model('41A3'))
        self.assertFalse(self.model_db.remove_user_model('41A3'))

        # assert that regression_model4 is removed from the ModelDatabase
        self.assertEqual(len(self.model_db.get_list_of_users()), 2)
        self.assertTrue('392' in      self.model_db.get_list_of_users())
        self.assertTrue('41A3' not in self.model_db.get_list_of_users())
        self.assertTrue('121A' in     self.model_db.get_list_of_users())

        # remove the remaining regression model from the ModelDatabase
        self.assertTrue(self.model_db.remove_user_model('392'))
        self.assertTrue(self.model_db.remove_user_model('121A'))

        # assert False when we try to delete the model again
        self.assertFalse(self.model_db.remove_user_model('392'))
        self.assertFalse(self.model_db.remove_user_model('121A'))

        # assert that the ModelDatabase is now empty
        self.assertEqual(len(self.model_db.get_list_of_users()), 0)
        self.assertTrue('392' not in  self.model_db.get_list_of_users())
        self.assertTrue('41A3' not in self.model_db.get_list_of_users())
        self.assertTrue('121A' not in self.model_db.get_list_of_users())

    
    def test_Persistence_of_model(self) :
        self.model_db = ModelDatabase('testP1')
        self.model_db.clear_database()

        # assert that our ModelDatabase is empty
        self.assertEqual(len(self.model_db.get_list_of_users()), 0)

        # show regression_model2 works
        self.assertAlmostEqual(self.regression_model2.predict(4.5)[0][0], 18.0)
        self.assertAlmostEqual(self.regression_model2.predict(20.0)[0][0], 80.0)

        # show regression_model4 works
        self.assertAlmostEqual(self.regression_model4.predict(11.0)[0][0] ,-22.0)
        self.assertAlmostEqual(self.regression_model4.predict(15.0)[0][0] ,-30.0)
        
        # insert the two model into the ModelDatabase
        self.assertTrue(self.model_db.store_user_model('243', self.regression_model2))
        self.assertTrue(self.model_db.store_user_model('41A3', self.regression_model4))  

        # reinitialze the model_db to some other ModelDatabase Object
        # our purpose is to deliberately loose reference to the object
        # thus calling the constructor with the same unique name, we can retreive previous data

        self.model_db = ModelDatabase('test3')
        self.model_db.clear_database()


        # reference back to our previous object through the unique name assigned
        self.model_db = ModelDatabase('testP1')

        # retrieve the two models as rg_model2 and rg_model4
        rg_model2 = self.model_db.get_user_model('243')
        rg_model4 = self.model_db.get_user_model('41A3')

        # assert that the retreived regression models are not none
        self.assertTrue(rg_model2 is not None)
        self.assertTrue(rg_model4 is not None)


        # assert that the retreived regression_model2 works
        self.assertAlmostEqual(rg_model2.predict(4.5)[0][0], 18.0)
        self.assertAlmostEqual(rg_model2.predict(20.0)[0][0], 80.0)
        self.assertAlmostEqual(rg_model2.predict(16.0)[0][0], 64.0)

        # assert that the retrieved regression_model4 works
        self.assertTrue(rg_model4.predict(11.0)[0][0], -22.0)
        self.assertTrue(rg_model4.predict(15.0)[0][0], -30.0)
        self.assertTrue(rg_model4.predict(2.5)[0][0], -5.0)


if __name__ == '__main__':
    unittest.main()
