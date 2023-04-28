import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import os
import numpy
from keras.models import load_model
from gtts import gTTS
from playsound import playsound
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI\VEN_10DE&DEV_1F99&SUBSYS_14471025&REV_A1\4&C337064&0&0009'
model = load_model('traffic_classifier.h5')

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    audio_file = os.path.join(os.getcwd(), "temp_audio.mp3")
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)


# Initialise GUI
top = tk.Tk()
top.geometry('900x700')
top.title('Traffic Sign Classification')
top.configure(background='#34495E')

label = Label(top, background='#34495E', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict(image)[0]
    pred_class = numpy.argmax(pred)
    sign = classes[pred_class+1]
    print(sign)
    label.configure(foreground='#ECF0F1', text=sign)
    speak_text(sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=20, pady=10)
    classify_b.configure(background='#F39C12', foreground='white', font=('arial', 14, 'bold'), borderwidth=0)
    classify_b.place(relx=0.5, rely=0.6, anchor=CENTER)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Upload an image", command=upload_image, padx=20, pady=10)
upload.configure(background='#3498DB', foreground='white', font=('arial', 14, 'bold'), borderwidth=0)

upload.place(relx=0.5, rely=0.15, anchor=CENTER)
sign_image.place(relx=0.5, rely=0.4, anchor=CENTER)
label.place(relx=0.5, rely=0.75, anchor=CENTER)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 28, 'bold'))
heading.configure(background='#34495E', foreground='#ECF0F1')
heading.place(relx=0.5, rely=0.05, anchor=CENTER)
top.mainloop()
