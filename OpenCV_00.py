import cv2
import numpy as np

def image() :
    img = cv2.imread('Assets/girl_reading.jpg')                # to read the image
    width = 600                                                
    height = 400                                               
    img_resized = cv2.resize(img, (width, height))             # resize image dimensions
    cv2.imshow('Output', img_resized)                          # to display the image
    cv2.waitKey(0)                                             # to keep the window open otherwise the window containing the image will automatically close as soon as it opens;                             0 --> infinite                    1000 --> 1 sec


def video() :
    vid = cv2.VideoCapture('Assets/cat.mp4')                   # to work with videos
    if (vid.isOpened() == False) :                             # checki if OpenCV is able to read the video stream 
        print('Error Opening Video stream or File !')
    
    while(vid.isOpened()) :                                    # a video is a series of frames and to read the frames one by one while loop is used
        ret, frame = vid.read()                                # provides Boolean output; True if able to read the frame otherwise False
        
        if ret == True :                                       # able to read the frame
            cv2.imshow('Frame', frame)                         # displaying the frame
            
            if cv2.waitKey(25) & 0xFF == ord('q') :            # press 'q' to exit or it will close after the last frame is displayed that is the video ends
                break

        else :
            break

    vid.release()                                              # release the video object

    cv2.destroyAllWindows()                                    # closes all the frames


def video_loop() :
    while True:                                                # Outer loop to make the video play continuously
        vid = cv2.VideoCapture('Assets/cat.mp4')               # Open the video file
        
        if not vid.isOpened():  
            print('Error Opening Video stream or File!')
            break

        while vid.isOpened():  
            ret, frame = vid.read()  

            if ret:
                cv2.imshow('Frame', frame)  

                if cv2.waitKey(25) & 0xFF == ord('q'):  
                    vid.release()
                    cv2.destroyAllWindows()
                    exit()                                      # Exits both loops and closes the window
            
            else:
                break                                           # End of video, exit inner loop to restart the video

    vid.release()                                               # Release the video and start over
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    image()
    #video()
    video_loop()                                                # will exit only upon pressing 'q' 