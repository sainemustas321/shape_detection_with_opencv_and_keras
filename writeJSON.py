import json
import glob
import os

save_path = './ans_auto/'

def write(SignPath,SignName,shapename,shape_color,alpha_color,TEXT,Heading,new_lat,new_long):
    NamePath = os.listdir('D:/work/avengers_assemble/ans_auto/')
    print(NamePath)
        
    SignPath = os.path.splitext(SignPath)[0]
    print(SignName, SignName.split('.jpg')[0] + '.json')
    if not SignName.split('.jpg')[0] + '.json' in NamePath:
        
        print(SignPath)
        

        Data = {
        "alphanumeric": TEXT,
        "alphanumeric_color": alpha_color,
        "autonomous": True,
        "latitude": new_lat,
        "longitude": new_long,
        "mission": 3,
        "orientation": Heading,
        "shape": shapename,
        "shape_color": shape_color,
        "type": "STANDARD"
    }
        SignName = SignName.split('.jpg')
        SignName = SignName[0]
        with open(save_path + SignName + ".json",'w') as outfile:
            json.dump(Data, outfile,indent=4)
            print(SignName + ".json")
            print("---------------JSON SAVED---------------")

