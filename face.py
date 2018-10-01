import config
import cv2
import detect
import numpy as np
from DB_handler import ModelDatabase

class Face(object):
    """
Class to obtain and store a trained model

Attributes:
Name = person's or user's name
id = unique id for each user
image_List = set of images to train a classifier 
"""
    def __init__(self,Name,Id,image_List):
        self.Name = Name
        self.Id = Id
        self.recognizer = self.trainRecognizer(image_List)
        self.store_model(self.recognizer)
    def extractFaces(self,image_List,levelFace=False):
        faceCascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
        eyeCascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)
        
        result = []
        for img in image_List:
            image, faces = detect.detectFaces(img, faceCascade, eyeCascade, True)
        for ((x, y, w, h), eyedim) in faces:
            if not levelFace:
                result.append(image[y:y+h, x:x+w])
            else:
                result.append(detect.levelFace(image, ((x, y, w, h), eyedim)))
        return result
      
    def trainRecognizer(self,image_List,trainSize=config.DEFAULT_FACE_SIZE, showFaces=False):
        #recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer = cv2.face.FisherFaceRecognizer_create()
        #recognizer = cv2.face.createEigenFaceRecognizer()
        images = []
        labels = []
        for image in image_List:
            faces = self.extractFaces(image, True)

            # resize all faces to same size for some recognizers
            images.extend([cv2.resize(face, trainSize) for face in faces])

            if showFaces:
                cv2.namedWindow("faces", 1)
                cv2.imshow("faces", image)
                cv2.waitKey(0)

            if showFaces:
                cv2.destroyWindow("faces")

        for i in range(len(image_List)):
            labels.append(1)
        recognizer.train(image_List, np.array(labels))

        #saveRecognizer(recognizer)
        return recognizer
    def store_model(self,model) :
        '''
        # store into the database
        model_db = ModelDatabase('Database1')
        model_db.clear_database()
        model_db.store_user_model(self.Id, model)
        print()
        '''
        model.save(self.Id+".dat")
