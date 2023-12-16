
import pickle
import os
import json



def save_dict(x_dict, path_save):
    f = open(path_save, "w")
    for key, v in  x_dict.items():
     str_ket_val = str(key) + " " + str(v) + "\n"
     f.write(str_ket_val)
    f.close()

def save_pickle(file_save,mydict):
    f  = open(file_save, "wb") 
    pickle.dump(mydict, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()      
         
def load_pickle(file_pickle):
    f  = open(file_pickle , "rb") 
    mydict = pickle.load(f)
    f.close()
    return mydict            

def create_clean_DIR(path):
 try:  
    os.mkdir(path)
 except OSError:
    path_ids = os.listdir(path)
    print ("Dir %s exist" % path)
    count_rm = 0
    for fileA in path_ids:
     if(len(fileA) > 2):
      print(fileA)
      if os.path.isdir(path + "/" + fileA):
       continue
      os.remove(path + "/" + fileA)
      count_rm +=1   
    print ("Dir" ,  path, "cleaned, files removed = " , count_rm)
 else:  
    print ("Successfully created the directory %s " % path) 

         

def new_dir(path ):
    if os.path.exists(path):
     print("exist:  ", path )
     
     return 1
    else:
     os.mkdir(path)
     return -1        
        

