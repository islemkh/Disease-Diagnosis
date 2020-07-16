#  import of librarys
import numpy as np
from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import  svm
from sklearn.feature_extraction.text import CountVectorizer 
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

''' from model import restore_model
from forms import Form
from model import restore_model '''

app = Flask(__name__)
app.secret_key = 'development key'
@app.route('/')

def index():
    return render_template('home.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/describe.html')
def describe():
    return render_template('describe.html')
@app.route('/describe', methods=['POST'])

def predict():
    df = pd.read_csv('Data1.csv', encoding="latin-1")
    df['class'] = df['Disease'].map({'Emotional pain':0, 'Hair falling out':1, 'Heart hurts':2,
        'Infected wound':3, 'Foot ache':4, 'Shoulder pain':5,
        'Injury from sports':6, 'Skin issue':7, 'Stomach ache':8, 'Knee pain':9,
        'Joint pain':10, 'Hard to breath':11, 'Head ache':12, 'Body feels weak':13,
        'Feeling dizzy':14, 'Back pain':15, 'Open wound':16, 'Internal pain':17,
        'Blurry vision':18, 'Acne':19, 'Muscle pain':20, 'Neck pain':21, 'Cough':22,
        'Ear ache':23, 'Feeling cold':24})
    names = df.Disease.unique()

    # data preprocessing

    df['Symptoms'].dropna(inplace=True)
    #Remove punctuation
    df['Symptoms'] = [w for w in df['Symptoms'] if w not in string.punctuation]
    #Change all the text to lower case.
    df['lowerSym'] = [entry.lower() for entry in df['Symptoms']]
    #Tokenization 
    df['tokenSym']= [word_tokenize(entry) for entry in df['lowerSym']]
    #Remove Stop words, Non-Numeric and apply Lemmenting.

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(df['tokenSym']):
    
        Final_words = []
        
        word_Lemmatized = WordNetLemmatizer()
        #provide the 'tag', if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            #check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        
        df.loc[index,'processed_Symptoms'] = str(Final_words)

    X = df['processed_Symptoms']
    y = df['class']

# model creation 
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM = SVM.fit(X_train,y_train)
    # predict the labels on validation dfset
    # predictions_SVM = SVM.predict(X_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = SVM.predict(vect)
    return render_template('result.html',prediction = my_prediction)

@app.route('/choose.html')
def choose():
    return render_template('choose.html')
# choose method prediction 
@app.route('/choose', methods=['POST'])
def predict2():
    df=pd.read_csv("train3.csv")

    l1=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
    'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition',
    'spotting_urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss',
    'restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
    'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea',
    'loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhea','mild_fever',
    'yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes',
    'sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain',
    'dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts',
    'drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck',
    'swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of_urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic_patches','watering_from_eyes','increased_appetite','polyuria',
    'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
    'distention_of_abdomen','history_of_alcohol_consumption','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
    'scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze']

    #List of Diseases is listed in list disease.

    disease=['Fungal infection', 'Allergy', 'GERD',
        'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae',
        'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma',
        'Hypertension ', 'Migraine', 'Cervical spondylosis',
        'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria',
        'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B',
        'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
        'Tuberculosis', 'Common Cold', 'Pneumonia',
        'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
        'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
        'Osteoarthristis', 'Arthritis',
        '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
        'Urinary tract infection', 'Psoriasis', 'Impetigo']

    l2=[]

    for i in range(0,len(l1)):
        l2.append(0)


    #Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

    df.replace({'Disease':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)

    #check the df 
    #print(df.head())

    X= df[l1]

    #print(X)

    y = df[["Disease"]]
    np.ravel(y)

    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    from sklearn.ensemble import RandomForestClassifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM = SVM.fit(X_train,y_train)

    from sklearn.metrics import accuracy_score
    y_pred=SVM.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    if request.method == 'POST':
        Symptom1 = request.form['symptom1']
        Symptom2 = request.form['symptom2']
        Symptom3 = request.form['symptom3']
        Symptom4 = request.form['symptom4']
        Symptom5 = request.form['symptom5']
        
        psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
        symps =[]
        for z in range(0,len(psymptoms)):
            if(psymptoms[z]) != 'none':
                symps.append(psymptoms[z])

        if symps != [] :   

            for k in range(0,len(l1)):
                for z in symps:
                    if(z==l1[k]):
                        l2[k]=1

            inputtest = [l2]
            predict = SVM.predict(inputtest)
            predicted=predict[0]

            h='no'
            r=''
            for a in range(0,len(disease)):
                if(predicted == a):
                    h='yes'
                    break


            if (h=='yes'):
                    #r.delete("1.0", END)
                r=( disease[a])
        else:
                #r.delete("1.0", END)
            r=("Not Found")
    return render_template('resultChoose.html',my_prediction = r)

if __name__ == '__main__':
   app.run(debug = True)
