def main():
    import os
    import cv2
    from face import Face
    from recog import Test
    images = []
    for filename in os.listdir('./face_data/s1'):
        try:
            img = cv2.imread('./face_data/s1/'+filename)
            if img is not None:
                img = img[:,:,0]
                images.append(img)
        except ValueError:
            print(filename +" is not recognised as image file")
        except IsADirectoryError:
            print(filename +" is a directory")
    Id = input('Enter Id: ')        
    f1 = Face(input('Enter Name: '),Id,images)
    img = cv2.imread('./face_data/s3/2.pgm')
    img = img[:,:,0]
    t1 = Test(img,Id)
    print(t1)
    
if __name__ == "__main__":
    main()                
