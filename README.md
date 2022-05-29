# Disease Prediction based on symptoms provided by patient with face Authentication Login System

In this era of IT, technology has revolutionized the health domain to a great extent. This project aims to design a diagnostic model for various diseases relying on their symptoms. This System has used data mining techniques such as classification in order to achieve such a model. Datasets consisting of voluminous data about patient diseases are gathered, refined and classified and were used for training the intelligent agent. Here, the Naive Bayes Algorithm is used for classification purpose. Naïve Bayes Classifier calculates the probability of the disease. Based on the result, the patient can contact the doctor accordingly for further treatment. It's an exemplar where technology and health knowledge are sewn into a thread perfectly with a desire to achieve "prediction is better than cure".

## **Installation:**

Clone the repo:

`$ git clone https://github.com/vivekagarwal2349/Disease-Diagnosis-system.git`

Open the project's folder:

`$ cd Disease-Diagnosis-system`

install the dependencies:

`$ pip install -r requirement.txt`

run the web application:

`$ python manage.py migrate`

`$ python manage.py runserver`

Navigate to the URL: <http://localhost:8000/>

#### In case of any issues, please try on Incognito Tab in your browser 

## Technologies Used:

**Frontend :** HTML, CSS, Bootstrap, JavaScript, JQuery

**Backend :** Django (Python based web framework)

**Database :** SQLite3

**Library :** OpenCV, Numpy, Pandas, SKLearn

## Architecture of the system:

<p align="center">
  <img width="800" height="400" src="https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/media/flowchart.png">
  <em>Architecture flowchart</em>
</p>


## Face Authentication Login System

### Face Detection

Here we used OpenCV ([haarcascade](https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/face_detect/haarcascade_frontalface_default.xml)) to detect and crop the face from the image frame. 
Firstly, while signing up, we took the 100 image from the user's webcam and cropped it. Then, we saved it in the local database.

### Face Recognition
#### Local Binary Patterns Histogram
Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number.
Using the LBP combined with histograms we can represent the face images with a simple data vector.

**Benefits:**

- LBPH is one of the easiest face recognition algorithms.
- Computational power and time complexity are both low.
- It provide great accuracy.
- It is provided by the OpenCV library

## Disease Prediction based on symptoms provided by patient

***Dataset Collection***

Data collection has been done from the internet to identify the disease here real symptoms of the disease are collected i.e. no dummy values are entered.

The symptoms of the disease are collected from Kaggle. The CSV file contains around 5000 rows of record of the patients with their symptoms(132 types of different symptoms) and their corresponding disease(40 class of general diseases).

[Dataset link](https://www.kaggle.com/datasets/neelima98/disease-prediction-using-machine-learning?select=Training.csv)

***Training Algorithm***

#### Naive Bayes

Due to big data progress in biomedical and healthcare communities, accurate study of medical data benefits early disease recognition, patient care and community services. When the quality of medical data is incomplete the exactness of study is reduced. Moreover, different regions exhibit unique appearances of certain regional diseases, which may results in weakening the prediction of disease outbreaks.

In this project, it bid a Machine learning Decision tree map, Navie Bayes, Random forest algorithm by using structured and unstructured data from hospital. It also uses Machine learning algorithm for partitioning the data. To the highest of gen, none of the current work attentive on together data types in the zone of remedial big data analytics. Compared to several typical calculating algorithms, the scheming accuracy of the proposed algorithm reaches 94.8% with an regular speed which is quicker than that of the unimodal disease risk prediction algorithm and produces report.

[code refer](https://www.kaggle.com/code/prashfio/clinical-decision-support-system)

## Some screenshots of this Webapp:

<span align="left">
  <img width="400" height="300" src="https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/media/home.png">
</span>
<span align="right">
  <img width="400" height="300" src="https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/media/signup.png">
</span>
<span align="left">
  <img width="400" height="300" src="https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/media/login.png">
</span>
<span align="right">
  <img width="400" height="300" src="https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/media/symptoms.png">
</span>
<span align="left">
  <img width="400" height="300" src="https://github.com/vivekagarwal2349/Disease-Diagnosis-system/blob/main/media/prediction.png">
</span>

## Future Scope:

A separate Doctor profile by which patient can consult corresponding doctor at very same platform. Further, the system can be extended to have more number of symptoms and disease. Currently, it does not recommend medications of the disease and Past history of the disease has not been considered.



<i>if you like this project, do give it a "Star" Thank you..
