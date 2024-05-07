import matplotlib.pyplot as plt
from tkinter import *
import threading
import os
from google_play_scraper import app, Sort,  reviews
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

appNameList=[]
url=[]

class MyWindow:
    def __init__(self, win):
        self.appNameList=[]
        self.url=[]
        self.lbl1=Label(win, text='Application Name')
        self.lbl1['fg']="blue"
        self.lbl2=Label(win, text='Application Url')
        self.lbl2['fg']="blue"
        self.lbl3=Label(win, text='RESULT')
        self.lbl4=Label(win, text='')  
        self.t1=Entry(bd=3,width=80)
        self.t2=Entry(width=80)  
        self.t3 = Text(win, height=20, width=60)
        self.t3.pack()
        self.btn1 = Button(win, text='Analysis')        
        self.lbl1.place(x=100, y=50)
        self.t1.place(x=200, y=50)
        self.lbl2.place(x=100, y=100)
        self.t2.place(x=200, y=100)
        self.b1=Button(win, text='Analysis', command=self.processinBack)
        self.b1.place(x=700, y=100)      
        self.lbl3.place(x=450, y=150)
        self.lbl4.place(x=550, y=150)
        self.t3.place(x=180, y=200)

    def processinBack(self):
        if (str(self.t1.get())!="" or str(self.t1.get())!=""):           
            self.lbl4['text']=""
            self.lbl4['fg']="white"
            download_thread = threading.Thread(target=self.Process)
            download_thread.start()      
        else:
            print("else")
            self.lbl4['text']="*Please Enter All Fields"
            self.lbl4['fg']="red"
 
    def Process(self):
        self.b1['state'] ="disabled"
        self.t3.delete(0.0, 'end')
        self.appNameList.append(str(self.t1.get()))
        self.url.append(str(self.t2.get()))        
        startingPoint = 0        
        self.t3.insert(END,("Started Analysing the Training Model\n"))     
        
        ##############################################################################################

        #print(str(self.t2.get()))
        appurl=str(self.t2.get())
        start = appurl.index('details?id=')
        end = len(appurl)
        #print(len(appurl))        
        #print(start)
        appid = appurl[start+len('details?id='):end]
        #print(appid)
    
        us_reviews, token =  reviews(
            appid, # app's ID, found in app's url
            lang='en',            # defaults to 'en'
            country='us',         # defaults to 'us'
            sort=Sort.NEWEST,     # defaults to Sort.MOST_RELEVANT
            filter_score_with=None,  # defaults to None (get all scores)
            count=40              # defaults to 100
            # , continuation_token=token
        )

        app_reviews_df = pd.DataFrame(us_reviews)
        #print(app_reviews_df["content"])
        #print(app_reviews_df["score"]) 
        finReviews=app_reviews_df["content"]         
        finRatings = app_reviews_df["score"].to_string(index=False)       
        
            
        ######################################################################################################## 

        
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Completed getting the Reviews and Ratings\n"))
      

        df = pd.read_csv('training.csv')
        df.head()

        def preprocess_data(df):

            # Remove package name as it's not relevant

            df = df.drop('package_name', axis=1)
            
            # Convert text to lowercase

            df['review'] = df['review'].str.strip().str.lower()
            return df

        df = preprocess_data(df)

        x = df['review']
        y = df['polarity']
        x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

        vec = CountVectorizer(stop_words='english')
        x = vec.fit_transform(x).toarray()
        x_test = vec.transform(x_test).toarray()

        #Using naive_bayes to train model and classification        
       
        model = MultinomialNB()
        model.fit(x, y)
        model.score(x_test, y_test)      
        fop=model.predict(vec.transform(finReviews))
        
        self.t3.insert(END,("\n"))
        self.t3.insert(END, ("Completed Analysis"))
        self.t3.insert(END,("\n"))
        self.t3.insert(END,(fop))
        PosrevCount=0
        NegrevCount=0
        for i in fop: 
            if(i==1):
                PosrevCount+=1
            else:
                NegrevCount+=1
          
        negratCount = finRatings.count('1')
        negratCount += finRatings.count('2')
        posratCount = finRatings.count('3')
        posratCount += finRatings.count('4')
        posratCount += finRatings.count('5')
               
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Negitive Reviews count : "+str(NegrevCount)))
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Positive Reviews count : "+str(PosrevCount)))
        self.t3.insert(END,("\n"))

        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Negitive Ratings count : "+str(negratCount)))
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Positive Ratings count : "+str(posratCount)))
        self.t3.insert(END,("\n"))
        PosrevCount +=posratCount
        NegrevCount +=negratCount

        PosrevCount=PosrevCount/2
        NegrevCount=NegrevCount/2
        self.t3.insert(END,("Average Negitive Reviews and Ratings percent : "+str((NegrevCount/(NegrevCount+PosrevCount)*100))))
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Average Positive Reviews and Ratings percent : "+str((PosrevCount/(NegrevCount+PosrevCount)*100))))
        self.t3.insert(END,("\n"))
        fig = plt.figure()

        if PosrevCount>=NegrevCount:
            fig.suptitle(self.appNameList[startingPoint]+" Verdict: This is a good APP")
        else:
            fig.suptitle(self.appNameList[startingPoint]+" Verdict: This is a Fraud/Faulty APP")
       
        ax = fig.add_axes([0,0,1,1])
        ax.axis('equal')
        langs = ['Positive reviews', 'Negetive reviews']
        students = [PosrevCount,NegrevCount]
        ax.pie(students, labels = langs,autopct='%1.2f%%')
        plt.show()
        self.b1.config(state="normal")


window=Tk()
mywin=MyWindow(window)
window.title('Fraud App Detector')
window.geometry("800x600+10+10")
window.mainloop()

