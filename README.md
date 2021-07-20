# FaceRecognition

Face recognition on videos and images using cv2 and face_recognition in python.
This project was based on tutorials on the channel "sentdex".

The basic idea of this sistem is to first train our model with images in the folder named as known_faces.
Then, we test it with images in the other folder named as unknown_faces. 
As a result, the model was capable of identify even images with masks, and also images of ourselves as a child.

For the video file, it was possible to identify the person on the camera with that same model.

## How to use

You can use this sistem with images of yourself!
The only thing that you will need to do is to create a folder with your name in known_faces and paste some images of yourself.
By doing that, and running facesImage.py or facesVideo.py our model will be trained with your images too.

You can also change the tolerance of our model ("TOLERANCE") if you want more security to the identification process.

Have fun!
