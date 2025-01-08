import argparse
from printing import printf

def arguments(args):
    """

    :param args:
    :return:
    """
    if args.gui:
        from Main import gui
        gui()
    elif args.interpreter:
        from Main import  system
        printf("Entering to Open Interpreter")
        while True:
            try:
                user_input=input("> ")
                text=system(user_input)
                printf(text)
            except Exception as e:
                printf("Goodbye!")
    elif args.autogen:
        import threading
        import webbrowser
        from Main import autogen

        auto = threading.Thread(target=autogen, args=(args.port, args.host))
        auto.start()
        printf("Opening Autogen Studio ")
        webbrowser.open_new_tab(f"https://{args.host}:{args}")
    elif args.start:
        from Main import main
        try:
            main()
        except Exception as e:
            printf("Good ByeG!G")
def main():
    """

    :return:
    """
    printf(r"""
         ____.       _____      __________  ____   ____ .___       _________                                                                                                                                           
    |    |      /  _  \     \______   \ \   \ /   / |   |     /   _____/                                                                                                                                           
    |    |     /  /_\  \     |       _/  \   Y   /  |   |     \_____  \                                                                                                                                            
/\__|    |    /    |    \    |    |   \   \     /   |   |     /        \                                                                                                                                           
\________| /\ \____|__  / /\ |____|_  / /\ \___/ /\ |___| /\ /_______  / /\                                                                                                                                        
           \/         \/  \/        \/  \/       \/       \/         \/  \/                                                                                                                                        
    ( Just A Rather Very Intelligent System )
    Made by:
        -----------------------------------------------
        | Name               |       USN              |
        |--------------------|------------------------|
        | Dawood Touseef     |  4CB21CG014            |
        |  Roshan D Talekar  |  4CB21CG042            |
        | Abhiram Hegde      |  4CB21CG001            |
        | Nishanth           |  4CB21CG032            |
        -----------------------------------------------
    College Name:Canara Engineering College
    Team Leader:Dawood Touseef
    Email:tdawood140@gmail.com
    """)
    parser = argparse.ArgumentParser(description="JARVIS Arguments")

    # Define the arguments individually - no group conflicts here
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode.")
    parser.add_argument("--interpreter", action="store_true", help="Enable Open Interpreter mode.")
    parser.add_argument("--start", action="store_true", help="Enable Jarvis mode.")
    parser.add_argument("--autogen", action="store_true", help="Enable Autogen Studio.")
    parser.add_argument("--port", type=int, default=8081, help="Port to run Autogen Studio.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run Autogen Studio.")
    args = parser.parse_args()
    arguments(args)

if __name__=="__main__":
    main()