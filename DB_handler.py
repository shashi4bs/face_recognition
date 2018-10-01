import pickle


'''
The ImageDatabase provides an interface to store a trained model
The constructor takes in a unique name parameter which identifies the Database.
To retreive the data, we can simply call the ModelDatabase
'''

class ModelDatabase:
    def __init__(self, name='unNamed'):
        self.__name = name
        self.__dataFileLocation = './dataFiles'
        self.__userFileLocation = './dataFiles/{}_users.dat'.format(self.__name)
        self.__fileExtension = '.dat'

        # creating file if it does not exist
        open(self.__userFileLocation, 'a+').close()



    # maps the model to the given userID and stores into the database
    # returns true if the model is successfully stored, else false
    # user Id has to be unique
    def store_user_model(self, userID, model) :
        data_file_destination = self.__dataFileLocation + '/' + self.__name + '_{}' + self.__fileExtension
        
        if userID not in self.get_list_of_users() :
            with open(data_file_destination.format(userID), 'wb') as f:
                pickle.dump(model, f)

            self.__add_userID_to_userFile(userID)
            return True
        return False


    # returns the stored model for the provided userID
    # returns the model upon success, else returns None
    def get_user_model(self, userID) :
        loaded_model  = None
        data_file_destination = self.__dataFileLocation + '/' + self.__name + '_{}' + self.__fileExtension
        if userID in self.get_list_of_users() :
            loaded_model = pickle.load(open(data_file_destination.format(userID), 'rb'))
        return loaded_model


    # returns a list of users whose model exists in the database, if empty, returns empty list
    def get_list_of_users(self) :
        userList = []
        with open(self.__userFileLocation, 'r') as f:
            for user in f:
                userList.append(user.strip('\n'))

        return userList

    # removes the model of the given userID
    # returns true if the user model successfully removed, else false
    def remove_user_model(self, userID) :
        userList = self.get_list_of_users()

        if userID in userList :
            # rewrite the userList File without the userID
            userList.remove(userID)
            with open(self.__userFileLocation, 'w+') as f:
                for user in userList :
                    f.write(user + '\n')
            return True
        return False

    
    def __add_userID_to_userFile(self, userID) :
        with open(self.__userFileLocation, 'a') as f:
            f.write(userID + '\n')

    def clear_database(self) :
        open(self.__userFileLocation, 'w+').close()
