import tkinter as tk
#Preprocessor Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PreProcessFunctions import preProcess
#from FFNNFunctions import neuralNetwork
#from FFNN import *
from tkinter import *
from optparse import Option
from tkinter import messagebox, filedialog, ttk, scrolledtext
from tkinter.filedialog import asksaveasfile
from tkinter.ttk import *
import queue
import sys
import cProfile
import pstats
from threading import *

class InvalidTrainingDataException(Exception):
    "Raised When Training Data is Not Selected"
    pass

class InvalidTestingDataException(Exception):
    "Raised When Testing Data is Not Selected"
    pass




#Resize Window Function
#def resize(e):
#    size = e.width/10
#    
#    if e.height <=400 and e.height > 300:
#        B1.config(font=("Helvetica", 40))
#    
#    elif e.height < 300 and e.height > 300:
#        B1.config(font=("Helvetica", 30))
#    
#    elif e.height < 200:
#        B1.config(font=("Helvetica", 40))




#Print Logger
class printLogger():
    def __init__(self, textbox) -> None:
        self.textbox = textbox
        
    def write(self, text):
        self.textbox.configure(state='normal')
        self.textbox.insert(tk.END, text)
        self.textbox.configure(state='disabled')
        
    def flush(self):
        
        pass
    
#PreProcessing    
#class preProcess:    
#    #Data Selection Functions
#    def readTrainCSV(self):
#        filename = filedialog.askopenfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Select Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", ("*.csv*")),("Text Files (*.txt*)", "*.txt*"), ("All Files (*.*)", "*.*")))
#        self.train = pd.read_csv(filename)
#        self.train.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
#        #Change Label Contents
#        label_file_explorer.configure(text="File Opened: " +filename)
#        #return train
#
#    def readTestCSV(self):
#        filename1 = filedialog.askopenfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Select Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", ("*.csv*")),("Text Files (*.txt*)", "*.txt*"), ("All Files (*.*)", "*.*")))
#        self.test = pd.read_csv(filename1)
#        self.test.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
#        #Change Label Contents
#        label_file_explorer1.configure(text="File Opened: " +filename1)
#        #return test
#
#    #Pre-Processor
#    def preProcess(self):
#        scaler = StandardScaler()
#
#        import time
#        progress['value'] = 0
#        top.update_idletasks()
#        time.sleep(1)
#
#    # extract numerical attributes and scale it to have zero mean and unit variance  
#        cols = self.train.select_dtypes(include=['float64','int64']).columns
#        sc_train = scaler.fit_transform(self.train.select_dtypes(include=['float64','int64']))
#        sc_test = scaler.fit_transform(self.test.select_dtypes(include=['float64','int64']))
#
#    # turn the result back to a dataframe
#        sc_traindf = pd.DataFrame(sc_train, columns = cols)
#        sc_testdf = pd.DataFrame(sc_test, columns = cols)
#
#        encoder = LabelEncoder()
#
#        progress['value'] = 20
#        top.update_idletasks()
#        time.sleep(1)
#        
#        # extract categorical attributes from both training and test sets 
#        cattrain = self.train.select_dtypes(include=['object']).copy()
#        cattest = self.test.select_dtypes(include=['object']).copy()
#
#        # encode the categorical attributes
#        traincat = cattrain.apply(encoder.fit_transform)
#        testcat = cattest.apply(encoder.fit_transform)
#
#        progress['value'] = 40
#        top.update_idletasks()
#        time.sleep(1)
#
#        # separate target column from encoded data 
#        enctrain = traincat.drop(['class'], axis=1)
#        cat_Ytrain = traincat[['class']].copy()
#        
#        train_x = pd.concat([sc_traindf,enctrain],axis=1)
#        train_y = self.train['class']
#        #print(train_x.shape)
#        
#        test_df = pd.concat([sc_testdf,testcat],axis=1)
#        #print(test_df.shape)
#        
#        progress['value'] = 60
#        top.update_idletasks()
#        time.sleep(1)
#        
#        rfc = RandomForestClassifier();
#
#        # fit random forest classifier on the training set
#        rfc.fit(train_x, train_y);
#        # extract important features
#        score = np.round(rfc.feature_importances_,3)
#        importances = pd.DataFrame({'feature':train_x.columns,'importance':score})
#        importances = importances.sort_values('importance',ascending=False).set_index('feature')
#        # plot importances
#        plt.rcParams['figure.figsize'] = (11, 4)
#        importances.plot.bar();
#
#        progress['value'] = 80
#        top.update_idletasks()
#        time.sleep(1)
#
#        from sklearn.feature_selection import RFE
#        import itertools
#        rfc = RandomForestClassifier()
#
#        # create the RFE model and select 10 attributes
#        rfe = RFE(rfc, n_features_to_select=15)
#        rfe = rfe.fit(train_x, train_y)
#
#        # summarize the selection of the attributes
#        feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
#        selected_features = [v for i, v in feature_map if i==True]
#        
#        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)
#        
#        progress['value'] = 100
#        top.update_idletasks()
#        time.sleep(1)
#        
#        file = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save X Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
#        file1 = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save X Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
#        file2 = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save Y Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
#        file3 = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save Y Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
#
#        
#        
#        self.X_train.to_csv(file, index = False)
#        self.X_test.to_csv(file1, index = False)
#        self.Y_train.to_csv(file2, index = False)
#        self.Y_test.to_csv(file3, index = False)
#        #messagebox.showinfo("Data Shapes","Shape of Test Data is: "+ str(train_x.shape)+ " Shape of Train Data is: "+ str(train_y.shape))
        
    
p = preProcess()

def windowThread():
    t1 = Thread(target=NetworkIntrusionDetectionApp)
    t1.start()
    #t1.join(30)
    #if t1.is_alive():
    #    print("Test Thread is still active")
    #else:
    #    print("Test Thread has stopped")
    #t1.join
    
def thread1():
    t2 = Thread(target=p.readTrainCSV)
    t2.start()
    #t2.join(30)
    if t2.is_alive():
        print("Test Thread is still active")
    else:
        print("Test Thread has stopped")
    t2.join

def thread2():
    t3 = Thread(target=p.readTestCSV)
    t3.start()

the_queue = queue

class NetworkIntrusionDetectionApp(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        #top = Tk()
        #top.geometry("800x800")
        #top.title("Neural Network")
        #top.protocol("WM_DELETE_WINDOW", self.callback)
        self.title("Neural Network")
        self.geometry("800x800")
        self.protocol("WM_DELETE_WINDOW", self.callback)
        #self.withdraw()
        #Menubar
        menubar = Menu()
        filemenu = tk.Menu(menubar, tearoff=0)
        helpmenu = tk.Menu(menubar, tearoff=0)
        editmenu = tk.Menu(menubar, tearoff=0)

        #File
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Open", command=self.donothing)
        filemenu.add_command(label="Save", command=self.donothing)
        filemenu.add_command(label="Save as...", command=self.donothing)
        filemenu.add_command(label="Close", command=self.donothing)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.callback)

        #Edit
        editmenu.add_command(label="Undo", command=self.donothing)
        editmenu.add_command(label="Redo", command=self.donothing)

        #Help
        helpmenu.add_command(label="Welcome", command=self.Welcome)
        helpmenu.add_command(label="Check For Updates...", command=self.donothing)
        helpmenu.add_command(label="About", command=self.About)

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Edit", menu=editmenu)
        menubar.add_cascade(label="Help", menu=helpmenu)
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Undo", command=self.donothing)
        self.config(menu=menubar)
    #label_file_explorer = Label(top,text="No Training Data Selected", width= 25)

    #label_file_explorer1 = Label(top,text="No Testing Data Selected", width= 25)

        #Grid
        Grid.rowconfigure(self,0,weight=1)
        Grid.rowconfigure(self,1,weight=1)
        Grid.rowconfigure(self,4,weight=1)
        
        Grid.columnconfigure(self,0,weight=1)
        Grid.columnconfigure(self,1,weight=1)
        
        
        #Buttons
        B1 = tk.Button(self, text="Select Training Data",command=p.readTrainCSV)
        B2 = tk.Button(self, text="Select Testing Data",command=p.readTestCSV)
        B3 = tk.Button(self, text="Pre-Process Data", command=p.Process)
        B4 = tk.Button(self, text="Initiate Neural Network", command=p.classifyModel)
        B5 = tk.Button(self, text="Start GridSearch", command=p.gridSearch)
        
        #Int Value Widgets
        epochs = tk.Entry(self)
        
        #Progress Bar Widget
        #progress = Progressbar(top, orient= HORIZONTAL, length= 100, mode= "determinate")

                #import time
                #progress['value'] = 0
                #top.update_idletasks()
                #time.sleep(1)

        #Place Widgets Within Window
        #label_file_explorer.grid(column=0, row=1, sticky=N)
        B1.grid(column=0, row=0, sticky=N)
        #label_file_explorer1.grid(column=0, row=3, sticky=N)
        B2.grid(column=0, row=1, sticky=N)
        B3.grid(column=0, row=2, sticky=N)
        #progress.grid(column=1, row=6, sticky=N, pady=10)
        B4.grid(column=0, row=3, sticky=N, pady=20)
        B5.grid(column=0, row=4, sticky=N)
        
        #Entry
        epochs.grid(column=1, row=3, sticky=W)
        epochs.insert(0,"10")
        
        #Output
        Label(self, text = "Output:").grid(column=0, row=5)
        t = tk.Text(self)
        pl = printLogger(t)
        #sys.stdout = pl
        t.configure(state="disabled")
        t.grid(column=0, row=6, sticky=N)
        
    def callback(self):
        if messagebox.askokcancel("Quit", "Do you really wish to quit?"):
            self.quit()
    
    #Functions for menubar
    def donothing(self):
        filewin = Toplevel(self)
        button = Button(filewin, text="Do nothing button")
        button.pack()

    def About(self):
        messagebox.showinfo("Version Info", "Neural Network Version: 0.01")

    def Welcome(self):
        messagebox.showinfo("Welcome", "Welcome to the Network Traffic Machine Learning Program")


    #File Browser
    #def browseFiles():
    #    global v
    #    filepath = filedialog.askopenfilename(initialdir= "/", title="Select a File", filetypes=(("Text Files (*.txt*)", "*.txt*"),("Comma-Seperated Value (*.csv*)", ("*.csv*")), ("All Files (*.*)", "*.*")))
    #    print(filepath)
#
    ##Save File Browser
    #def save():
    #    files = [("Comma-Seperated Value (*.csv*)", ("*.csv*")), ("All Files (*.*)", "*.*")]
#
    #    file = asksaveasfile(filetypes= files, defaultextension= files)
    
    

    #text_area = scrolledtext.ScrolledText(top, width=40, height=10)
    #text_area.configure(state='disabled')
    #text_area.grid(column=0, row=7, pady=10, padx=10)
    #text_area.insert(tk.INSERT, pl)





    #top.bind('<Configure>', resize)
#cProfile.run('NetworkIntrusionDetectionApp()','NetworkIntrusionDetectionApp.profile')
#stats = pstats.Stats('NetworkIntrusionDetectionApp.profile')
#stats.strip_dirs().sort_stats('time').print_stats()
netapp = NetworkIntrusionDetectionApp()
netapp.mainloop()
#netapp.update
#netapp.update_idletasks



#Copy of class version of program

#class NetworkIntrusionDetectionApp(tk.Tk):
#    
#    def __init__(self):
#        tk.Tk.__init__(self)
#        top = tk.Tk()
#        top.geometry("400x400")
#        top.title("Neural Network")
#        self.withdraw()
#        #Menubar
#        menubar = Menu(top)
#        filemenu = tk.Menu(menubar, tearoff=0)
#        helpmenu = tk.Menu(menubar, tearoff=0)
#        editmenu = tk.Menu(menubar, tearoff=0)
#
#        #File
#        filemenu.add_command(label="New", command=self.donothing)
#        filemenu.add_command(label="Open", command=self.donothing)
#        filemenu.add_command(label="Save", command=self.donothing)
#        filemenu.add_command(label="Save as...", command=self.donothing)
#        filemenu.add_command(label="Close", command=self.donothing)
#        filemenu.add_separator()
#        filemenu.add_command(label="Exit", command=top.quit)
#
#        #Edit
#        editmenu.add_command(label="Undo", command=self.donothing)
#        editmenu.add_command(label="Redo", command=self.donothing)
#
#        #Help
#        helpmenu.add_command(label="Welcome", command=self.Welcome)
#        helpmenu.add_command(label="Check For Updates...", command=self.donothing)
#        helpmenu.add_command(label="About", command=self.About)
#
#        menubar.add_cascade(label="File", menu=filemenu)
#        menubar.add_cascade(label="Edit", menu=editmenu)
#        menubar.add_cascade(label="Help", menu=helpmenu)
#        editmenu = Menu(menubar, tearoff=0)
#        editmenu.add_command(label="Undo", command=self.donothing)
#        top.config(menu=menubar)
#    #label_file_explorer = Label(top,text="No Training Data Selected", width= 25)
#
#    #label_file_explorer1 = Label(top,text="No Testing Data Selected", width= 25)
#
#        #Grid
#        Grid.rowconfigure(top,0,weight=1)
#        Grid.columnconfigure(top,0,weight=1)
#        Grid.rowconfigure(top,1,weight=1)
#
#        #Buttons
#        B1 = tk.Button(top, text="Select Training Data",command=p.readTrainCSV)
#        B2 = tk.Button(top, text="Select Testing Data",command=p.readTestCSV)
#        B3 = tk.Button(top, text="Pre-Process Data", command=p.preProcess)
#
#        #Progress Bar Widget
#        #progress = Progressbar(top, orient= HORIZONTAL, length= 100, mode= "determinate")
#
#
#                #import time
#                #progress['value'] = 0
#                #top.update_idletasks()
#                #time.sleep(1)
#
#
#
#        #Place Widgets Within Window
#        #label_file_explorer.grid(column=0, row=1, sticky=N)
#        B1.grid(column=0, row=2, sticky=N)
#        #label_file_explorer1.grid(column=0, row=3, sticky=N)
#        B2.grid(column=0, row=4, sticky=N)
#        B3.grid(column=0, row=5, sticky=N, pady=20)
#        #progress.grid(column=1, row=6, sticky=N, pady=10)
#
#
#        #Output
#        Label(top, text = "Output:").grid(column=0, row=6)
#        t = tk.Text(top)
#        pl = printLogger(t)
#        sys.stdout = pl
#        #t.configure(state="normal")
#        t.grid(column=0, row=7)
#
#    
#    
#    #Functions for menubar
#    def donothing(self):
#        filewin = Toplevel(self.top)
#        button = Button(filewin, text="Do nothing button")
#        button.pack()
#
#    def About(self):
#        messagebox.showinfo("Version Info", "Neural Network Version: 0.01")
#
#    def Welcome(self):
#        messagebox.showinfo("Welcome", "Welcome to the Network Traffic Machine Learning Program")
#
#
#    #File Browser
#    #def browseFiles():
#    #    global v
#    #    filepath = filedialog.askopenfilename(initialdir= "/", title="Select a File", filetypes=(("Text Files (*.txt*)", "*.txt*"),("Comma-Seperated Value (*.csv*)", ("*.csv*")), ("All Files (*.*)", "*.*")))
#    #    print(filepath)
##
#    ##Save File Browser
#    #def save():
#    #    files = [("Comma-Seperated Value (*.csv*)", ("*.csv*")), ("All Files (*.*)", "*.*")]
##
#    #    file = asksaveasfile(filetypes= files, defaultextension= files)
#    
#    
#
#    #text_area = scrolledtext.ScrolledText(top, width=40, height=10)
#    #text_area.configure(state='disabled')
#    #text_area.grid(column=0, row=7, pady=10, padx=10)
#    #text_area.insert(tk.INSERT, pl)
#
#
#
#
#
#    #top.bind('<Configure>', resize)
#netapp = NetworkIntrusionDetectionApp()
#netapp.mainloop()



