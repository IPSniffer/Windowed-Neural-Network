import tkinter as tk
from PreProcessFunctions import preProcess
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
from idlelib.tooltip import Hovertip

#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
        self.geometry("400x400")
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


        #Grid
        #Grid.rowconfigure(self,0,weight=1)
        #Grid.rowconfigure(self,1,weight=1)
        #Grid.rowconfigure(self,4,weight=1)
        
        Grid.columnconfigure(self,0,weight=1)
        Grid.columnconfigure(self,1,weight=1)
        
        #Buttons
        B1 = tk.Button(self, text="Select Training Data", command=p.readTrainCSV)
        B2 = tk.Button(self, text="Select Testing Data", command=p.readTestCSV)
        B3 = tk.Button(self, text="Pre-Process Data", command=self.saveSwitch)
        B4 = tk.Button(self, text="Initiate Neural Network", command=self.Number)
        B5 = tk.Button(self, text="Start GridSearch", command=p.gridSearch)
        
        #Int Value Widgets
        self.epochs = tk.Entry(self)
        self.batchsize = tk.Entry(self)
        self.Cbvalue = tk.IntVar(self)
        Cb= Checkbutton(self, text="Save Data Locally?", variable=self.Cbvalue, onvalue=1, offvalue=0)

        #Place Widgets Within Window
        B1.grid(column=0, row=0, sticky=N)
        B2.grid(column=0, row=1, sticky=N, pady=20)
        B3.grid(column=0, row=2, sticky=N)
        B4.grid(column=0, row=3, sticky=N, pady=20)
        B5.grid(column=0, row=4, sticky=N)
        Cb.grid(column=1, row=2, sticky=W)
        Label(self, text = "CPU Usage Intesive").grid(column=1, row=4, sticky=W)
        
        #Entry
        self.epochs.grid(column=1, row=3, sticky=W)
        self.epochs.insert(0,"Number of Epochs")
        self.batchsize.grid(column=2, row=3, sticky=W)
        self.batchsize.insert(0,"Batch Size")
        
        #ToolTips
        Hovertip(B1,'Click to open a dialog box to select Training Data')
        Hovertip(B2,'Click to open a dialog box to select Testing Data')
        Hovertip(B3,'Click to Pre-Process Data Selected')
        
        
        #Redirected Output
        #Label(self, text = "Output:").grid(column=0, row=5)
        t = tk.Text(self)
        pl = printLogger(t)
        #sys.stdout = pl
        t.configure(state="disabled")
        #t.grid(column=0, row=6)
        
    def callback(self):
        if messagebox.askokcancel("Quit", "Do you really wish to quit?"):
            self.quit()
    
    #Functions for menubar
    def donothing(self):
        filewin = Toplevel(self)
        button = Button(filewin, text="Do nothing button")
        button.pack()

    def About(self):
        messagebox.showinfo("Version Info", "Neural Network Version: 0.1")

    def Welcome(self):
        messagebox.showinfo("Welcome", "Welcome to the Network Traffic Machine Learning Program")

    #Gets the value placed in the Entry box and converts it into a Int value then runs the classification
    def Number(self):
        #number = IntVar()
        number = self.epochs.get()
        batch = self.batchsize.get()
        p.classifyModel(epochs=(int(number)), batch_size=(int(batch)))
        #p.report()
    
    def saveSwitch(self):
        if self.Cbvalue.get() == 1:
            p.processDialog()
        else:
            p.Process()
            
    #Drop Columns from the Training Data Commented out for now
    #def trainColumn(self):
    #    ctop = Toplevel()
    #    ctop.title("Enter Column names to be Dropped")
    #    ctop.geometry("200x200")
    #    Label(ctop, text="Enter Training Columns to be Dropped (Remeber to use Commas)").grid(column=0, row=0)
    #    self.traincolumn = tk.Entry(ctop)
    #    self.traincolumn.grid(column=0, row=1, sticky=W)
    #    self.traincolumn.insert(0,"Enter Columns Here")
    #    B1 = tk.Button(ctop, text="Open Training Data CSV File", command=self.dropColumn)
    #    B1.grid(column=0, row=2, sticky=W)
    #    
    #def dropColumn(self):
    #    columns = self.traincolumn.get()
    #    p.readTrainCSV(columns=columns)

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
    
    #top.bind('<Configure>', resize)
#cProfile.run('NetworkIntrusionDetectionApp()','NetworkIntrusionDetectionApp.profile')
#stats = pstats.Stats('NetworkIntrusionDetectionApp.profile')
#stats.strip_dirs().sort_stats('time').print_stats()
netapp = NetworkIntrusionDetectionApp()
netapp.mainloop()
#netapp.update
#netapp.update_idletasks
