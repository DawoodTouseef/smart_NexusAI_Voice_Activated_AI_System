from colorama import Style,Fore
from datetime import datetime
from env import CACHE_DIR
import os
import json

def printf(text:str,type:str="info"):
    types=['error','warn','info']
    date_time=datetime.now()
    texts=f"{date_time} -{type.upper()} - {text} "
    if type in types:
        if type.lower() =="error":
            print(Fore.RED + Style.BRIGHT + texts)
        elif type.lower() =="warn":
            print(Fore.YELLOW + Style.BRIGHT + texts)
        elif type.lower()=="info":
            print(Fore.GREEN+Style.BRIGHT+texts)
    else:
        print(Fore.WHITE + Style.BRIGHT + texts)
    log_dir=os.path.join(CACHE_DIR,"logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    log_path=os.path.join(log_dir,"logs.json")
    if not os.path.exists(log_path):
        with open(log_path,"w") as f:
            json.dump([{"type":type,"message":text,"time":f"{date_time}"}],f)
    else:
        with open(log_path,"r") as f:
            data=json.load(f)
        data.append({"type":type,"message":text,"time":f"{date_time}"})
        with open(log_path,"w") as f:
            json.dump(data,f)
    print(Fore.RESET+Style.RESET_ALL)