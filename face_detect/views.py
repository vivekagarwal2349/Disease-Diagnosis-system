from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from .models import *
import cv2
import numpy as np
from django.http import JsonResponse
from os import listdir
from os.path import join, isfile
from django.forms.models import model_to_dict

import joblib as jb
model = jb.load('trained_model')

def Login(request):
    error=""
    if request.method=="POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u,password=p)

        # CHECK IF USERNAME AND PASSWORD IS CORRECT

        try:
            user.is_staff
        except:
            Message = {'type': 'danger', 'message': 'Incorrect Username or Password', 'heading': 'Error'}
            return render(request,'login1.html',{'Message': Message})

        if user.is_staff:

            # TRAINING MODEL

            data_path = './face_detect/images/'

            Training_Data, Labels = [], []

            for i in range(1,101):
                image_path = data_path + str(u) + str(i) + ".jpg"
                images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if images is None:
                    Message = {'type': 'danger', 'message': 'Image Dataset not Found', 'heading': 'Error'}
                    return render(request,'login1.html',{'Message': Message})
                Training_Data.append(np.asarray(images, dtype=np.uint8))
                Labels.append(i)

            Labels = np.asarray(Labels, dtype=np.int32)

            model = cv2.face.LBPHFaceRecognizer_create()

            model.train(np.asarray(Training_Data), np.asarray(Labels))

            print("Model training Complete !!!!!")

            face_classifier = cv2.CascadeClassifier(
                './face_detect/haarcascade_frontalface_default.xml')

            def face_detector(img, size=0.5):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)

                if faces is ():
                    return img, []

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    roi = img[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (200, 200))

                return img, roi

            # DETECTING CAMERA

            all_camera_idx_available = -1

            for camera_idx in range(10):
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(f'Camera index available: {camera_idx}')
                    all_camera_idx_available=camera_idx
                    cap.release()
                    break

            if all_camera_idx_available == -1:
                Message = {'type': 'danger', 'message': 'Camera Not Found', 'heading': 'Error'}
                return render(request,'login1.html',{'Message': Message})


            cap = cv2.VideoCapture(all_camera_idx_available)
            count = 0

            # PREDICTING THE FRAME

            while True:
                ret, frame = cap.read()
                image, face = face_detector(frame)

                if len(face) != 0:
                    count += 1
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face)
                    cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', frame)

                    if result[1] < 500:
                        confidence = int(100 * (1 - (result[1]) / 300))

                    if confidence > 85:
                        login(request,user)
                        error="yes"
                        break

                    elif count<=5 :
                        pass

                    else:
                        error="no"
                        break

                else:
                    print("Face Not Found")
                    cap_not = frame
                    cv2.putText(cap_not, str(count)+" Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', cap_not)

                    if cv2.waitKey(1) == 5 or count == 2:
                        error ="not_found"
                        break

            cap.release()
            cv2.destroyAllWindows()

            if error=="yes":
                Message = {'type': 'success', 'message': 'Logged IN', 'heading': 'Success!'}
                error=""
                return render(request,'home.html',{'Message': Message})
            elif error=="no":
                Message = {'type': 'danger', 'message': 'Face Not Matching', 'heading': 'Error!'}
                error=""
                return render(request,'login1.html',{'Message': Message})
            else:
                Message = {'type': 'info', 'message': 'Face Not in Frame', 'heading': 'Info!'}
                error=""
                return render(request,'login1.html',{'Message': Message})

    return render(request,'login1.html')

def checkdisease(request):

    diseaselist=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes ',
  'Gastroenteritis','Bronchial Asthma','Hypertension ','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)',
  'Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
  'Hepatitis E', 'Alcoholic hepatitis','Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
  'Heart attack', 'Varicose veins','Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
  'Arthritis', '(vertigo) Paroymsal  Positional Vertigo','Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']


    symptomslist=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
  'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination',
  'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
  'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating',
  'dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes',
  'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
  'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
  'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
  'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
  'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
  'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
  'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
  'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
  'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
  'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
  'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
  'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
  'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
  'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
  'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
  'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
  'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
  'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
  'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
  'yellow_crust_ooze']

    alphabaticsymptomslist = sorted(symptomslist)
    
    if request.method == 'GET':
        print("GET")
        return render(request,'checkdisease.html', {"list2":alphabaticsymptomslist})

    elif request.method == 'POST':
      print("POST")

      inputno = int(request.POST["noofsym"])
      if (inputno == 0 ) :
          Message = {'type': 'info', 'message': ' Please add some symptoms', 'heading': 'Info'}
          return render(request,'checkdisease.html',{'Message': Message})

      else :
        psymptoms = []
        psymptoms = request.POST.getlist("symptoms[]")  
       
        print(psymptoms)

    # CREATING INPUT ARRAY FOR THE MODEL 

    testingsymptoms = []

    for x in range(0, len(symptomslist)):
        testingsymptoms.append(0)


    for k in range(0, len(symptomslist)):

        for z in psymptoms:
            if (z == symptomslist[k]):
                testingsymptoms[k] = 1


    inputtest = [testingsymptoms]

    print(inputtest)

    predicted = model.predict(inputtest)
    print("predicted disease is : ")
    print(predicted)

    y_pred_2 = model.predict_proba(inputtest)
    confidencescore=y_pred_2.max() * 100
    print(" confidence score of : = {0} ".format(confidencescore))

    confidencescore = format(confidencescore, '.0f')
    predicted_disease = str(predicted[0])
    print(type(predicted_disease))

    Rheumatologist = [  'Osteoarthristis','Arthritis']
       
    Cardiologist = [ 'Heart attack','Bronchial Asthma','Hypertension ']
    
    ENT_specialist = ['(vertigo) Paroymsal  Positional Vertigo','Hypothyroidism' ]

    Orthopedist = []

    Neurologist = ['Varicose veins','Paralysis (brain hemorrhage)','Migraine','Cervical spondylosis']

    Allergist_Immunologist = ['Allergy','Pneumonia',
    'AIDS','Common Cold','Tuberculosis','Malaria','Dengue','Typhoid']

    Urologist = [ 'Urinary tract infection',
        'Dimorphic hemmorhoids(piles)']

    Dermatologist = [  'Acne','Chicken pox','Fungal infection','Psoriasis','Impetigo']

    Gastroenterologist = ['Peptic ulcer diseae', 'GERD','Chronic cholestasis','Drug Reaction','Gastroenteritis','Hepatitis E',
    'Alcoholic hepatitis','Jaundice','hepatitis A',
        'Hepatitis B', 'Hepatitis C', 'Hepatitis D','Diabetes ','Hypoglycemia']
        
    if predicted_disease in Rheumatologist :
        consultdoctor = "Rheumatologist"
        
    if predicted_disease in Cardiologist :
        consultdoctor = "Cardiologist"
        

    elif predicted_disease in ENT_specialist :
        consultdoctor = "ENT specialist"
    
    elif predicted_disease in Orthopedist :
        consultdoctor = "Orthopedist"
    
    elif predicted_disease in Neurologist :
        consultdoctor = "Neurologist"
    
    elif predicted_disease in Allergist_Immunologist :
        consultdoctor = "Allergist/Immunologist"
    
    elif predicted_disease in Urologist :
        consultdoctor = "Urologist"
    
    elif predicted_disease in Dermatologist :
        consultdoctor = "Dermatologist"
    
    elif predicted_disease in Gastroenterologist :
        consultdoctor = "Gastroenterologist"
    
    else :
        consultdoctor = "Physician"

    return JsonResponse({'predicteddisease': predicted_disease, "confidencescore" : confidencescore, "consultdoctor" : consultdoctor})

def home(request):
    return render(request,'home.html')

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')


def signup(request):
    error = False
    success_flag = False
    incomplete_details = ""

    # GETTING USER DETAILS FROM THE FORM

    if request.method=="POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        a = request.POST['add']
        m = request.POST['mobile']
        try:
            i = request.FILES['image']
        except:
            i = ""
        
        if u=="" or p=="" or f=="" or l=="" or e=="" or a=="" or m=="" or i=="":
            Message = {'type': 'danger', 'message': 'Please fill all the details', 'heading': 'Error'}
            return render(request,'signup.html',{'Message': Message})

        try:
            user = User.objects.create_superuser(username=u,password=p,email=e,first_name=f,last_name=l)
                
            Profile.objects.create(user=user,mobile=m,add=a,image=i)
            face_classifier = cv2.CascadeClassifier(
                './face_detect/haarcascade_frontalface_default.xml')

            def face_extractor(img):

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)

                if faces is ():
                    return None

                for (x, y, w, h) in faces:
                    cropped_faces = img[y:y + h, x:x + w]

                return cropped_faces

            # DETECTING CAMERA
            
            all_camera_idx_available = -1

            for camera_idx in range(10):
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(f'Camera index available: {camera_idx}')
                    all_camera_idx_available=camera_idx
                    cap.release()
                    break

            if all_camera_idx_available == -1:
                Message = {'type': 'danger', 'message': 'Camera Not Found', 'heading': 'Error'}
                return render(request,'signup.html',{'Message': Message})

            # DETECTING FACE AND CAPTURE 100 SUCH IMAGES

            cap = cv2.VideoCapture(all_camera_idx_available)
            count = 0

            while True:
                ret, frame = cap.read()
                if frame is None:
                    Message = {'type': 'danger', 'message': 'Camera Not Found', 'heading': 'Error'}
                    return render(request,'signup.html',{'Message': Message})

                if face_extractor(frame) is not None:
                    count += 1
                    face = cv2.resize(face_extractor(frame), (400, 400))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    file_name_path = './face_detect/images/'+ str(u) + str(count) + '.jpg'
                    cv2.imwrite(file_name_path, face)

                    cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', frame)

                else:
                    cap_not = frame
                    cv2.putText(cap_not, str(count)+" Face Not in Frame", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', cap_not)
                    pass

                if cv2.waitKey(1) == 13 or count == 100:
                    success_flag = True
                    break

            cap.release()
            cv2.destroyAllWindows()
            error = True

        except:
            Message = {'type': 'danger', 'message': 'Username already exists', 'heading': 'Error'}
            return render(request,'signup.html',{'Message': Message})


    if success_flag:
        success_flag = False
        return redirect('/login')

    d = {'error':error,
         'incomplete_details':incomplete_details}

    return render(request,'signup.html',d)

def Logout(request):
    logout(request)
    return redirect('home')

