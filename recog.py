import config
import cv2
import detect
class Test(object):

    def __init__(self,image,Id):
        self.result = self.RecognizeFace(image,Id)
        
        
    def extractFaces(self,image,levelFace=False):
        faceCascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
        eyeCascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)
        
        result = []
        img, faces = detect.detectFaces(image, faceCascade, eyeCascade, True)
        for ((x, y, w, h), eyedim) in faces:
            if not levelFace:
                result.append(img[y:y+h, x:x+w])
            else:
                result.append(detect.levelFace(img, ((x, y, w, h), eyedim)))
        return result
        
    def RecognizeFace(self,image,Id, faceSize=config.DEFAULT_FACE_SIZE, threshold=500):
        faceCascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
        eyeCascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)
        recognizer = cv2.face.FisherFaceRecognizer_create()
        recognizer.read(Id+".dat")
        found_faces = []
        '''
        gray, faces = detect.detectFaces(image, faceCascade, eyeCascade, returnGray=1)

        # If faces are found, try to recognize them
        for ((x, y, w, h), eyedim)  in faces:
            label, confidence = recognizer.predict(cv2.resize(detect.levelFace(gray, ((x, y, w, h), eyedim)), faceSize))
            # note that for some distributions of python-opencv, the predict function
            # returns the label only.
            #label = recognizer.predict(cv2.resize(detect.levelFace(gray, ((x, y, w, h), eyedim)), faceSize))
            #confidence = -1
            #if confidence > threshold:
            found_faces.append((label, confidence, (x, y, w, h)))
            print(confidence,label)'''
        print(recognizer.predict(image)) 
        print(found_faces)        

        return found_faces
