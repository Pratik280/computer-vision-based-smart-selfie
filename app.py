from flask import Flask,render_template,Response,request
import cv2
import numpy as np

app=Flask(__name__)

def generate_frames():
    camera=cv2.VideoCapture(0)
    while(True):
        success,frame=camera.read()
        if not success:
            break
        else:
            faceCascade = cv2.CascadeClassifier(r"C:\Users\ASUS\Desktop\Datasets\ml\dataset\haarcascade_frontalface_default.xml")
            smileCascade = cv2.CascadeClassifier(r"C:\Users\ASUS\Desktop\Datasets\ml\dataset\haarcascade_smile.xml")
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray,1.3,5)
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray=gray[y:y+h,x:x+w]
                    roi_color=frame[y:y+h,x:x+w]
                    smiles=smileCascade.detectMultiScale(roi_gray,1.5,30,minSize=(50,50))
                    for i in smiles:
                        if len(i)>1:
                            cv2.putText(frame,"SMILING",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv2.LINE_AA)
                            path=r'C:\Users\ASUS\Desktop\Datasets\ml\dataset\img.jpg'
                            cv2.imwrite(path,frame)
                            camera.release()
                            cv2.destroyAllWindows()
                            break

            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_eye():
    first_read = True
    b=0
    camera=cv2.VideoCapture(0)
    while(True):
        success,frame=camera.read()
        if not success:
            break
        else:
            faceCascade = cv2.CascadeClassifier(r"C:\Users\ASUS\Desktop\Datasets\ml\dataset\haarcascade_frontalface_default.xml")
            eye_cascade=cv2.CascadeClassifier(r"C:\Users\ASUS\Desktop\Datasets\ml\dataset\haarcascade_eye_tree_eyeglasses.xml")
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray,1.3,5)
            a=0
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray=gray[y:y+h,x:x+w]
                    roi_color=frame[y:y+h,x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5,minSize=(50,50)) 
                    if(len(eyes)>=1):
                        if(first_read):
                            a=1
                            cv2.putText(frame,"Eye detected ",(70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
                        else:
                            if(len(eyes)>=1):
                                if b==1:
                                    path=r'C:\Users\ASUS\Desktop\Datasets\ml\dataset\eye.jpg'
                                    cv2.imwrite(path,frame)
                                    camera.release()
                                    cv2.destroyAllWindows()
                                b=0
                                cv2.putText(frame,"blink ", (70,70),cv2.FONT_HERSHEY_PLAIN, 2,(123,123,255),2)
                    else:
                        if(first_read):
                            #To ensure if the eyes are present before starting
                            cv2.putText(frame,"ok", (70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                        else:
                            b=1
                            cv2.putText(frame,"okkkk", (70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                            #This will print on console and restart the algorithm
                            first_read=True
                            break
            else:
                cv2.putText(frame,"No face detected",(100,100),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
                
            if(a==1 and first_read):
                #This will start the detection
                first_read = False
            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

            
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/smile.html')
def smile():
    return render_template('smile.html')

@app.route('/eye.html',methods=["POST","GET"])
def eye():
    return render_template('eye.html')

@app.route('/Video')
def Video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Videos')
def Videos():
    return Response(generate_frames_eye(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
