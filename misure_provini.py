import cv2


def click_event(event, x, y, flags, params): 
    global side ,counter
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
        side.append(x*scale_F)
        if counter<5:
            if len(side)==2:
                pxmm = abs(side[1]-side[0])/30
                print(pxmm)
                side=[]
                counter+=1
        else:
            print("final pxmm = ", pxmm)
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 

        cv2.imshow('image', img) 
        
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 


if __name__=="__main__": 
  
    # reading the image 
    img = cv2.imread('C:\\Users\\alberto.scolari\\Pictures\\digiCamControl\\Session1\\DSC_0014.jpg') 
    scale_F = 4
    global side, counter
    counter =0
    side = []
    h = int(img.shape[1]/scale_F)
    w = int(img.shape[0]/scale_F)
    img = cv2.resize(img,( h,w) )
    # displaying the image 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 
  
    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 