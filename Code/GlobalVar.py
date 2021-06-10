# Define the global variables used in multi-files
# Usually, it is only used for Writer variable (in TensorBoard)

def _init():
    global _global_dict
    _global_dict = {}

def set_var(key,value):
    # define a global variable
    _global_dict[key] = value
 
def del_var(key):
    # delete a global variable
    try:
        del _global_dict[key]
    except KeyError:
        print("Key:"+key+"unexists\r\n")

def get_var(key,defValue=None):
    # get the value of a global variable
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
        print("Reading"+key+"fails\r\n")
