import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


new_prop = 100

def on_button_click(text_field, root):
        try: 
            i = 0
            for widget in root.winfo_children():
                if widget.winfo_class() == "Label" and i!=0:
                    widget.destroy()
                i+=1

            shape_model = tf.keras.models.load_model(r"C:\Users\IlmoK\OneDrive\coding\shape_predictor.h5", compile=True)
            dir = text_field.get()
            image = tf.keras.preprocessing.image.load_img(dir, target_size=(new_prop, new_prop))
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            resize_image = tf.reshape(image_array, [-1, 3, new_prop, new_prop])
            prediction = shape_model.predict(resize_image)

            labels = ['circle','square','star','triangle']
            label = labels[np.argmax(prediction)]
            message = tk.Label(root, text = 'It is a '+label)
            message.pack()
        

            match_list = prediction[0]
            prob = match_list[np.argmax(match_list)]*100
            prob_message = tk.Label(root, text ='propability: '+ str(prob)+'%')
            prob_message.pack()


            tk_image = ImageTk.PhotoImage(file = dir)
            image_label = tk.Label(root, image = tk_image)
            image_label.image = tk_image
            image_label.pack()
        except:
            label = tk.Label(root, text = 'Invalid address')
            label.pack()
        



def main():
   
    root = tk.Tk()
    root.title("Shape recognizer")
    Title = tk.Label(root, text ='Insert the address')
    Title.pack()
    text_field = tk.Entry(root, width=80)
    text_field.pack()
    button = tk.Button(root, text="predict!", command=lambda: on_button_click(text_field, root))
    button.pack()
    root.mainloop()
    
if __name__=='__main__':
    main()
