import json
from typing import OrderedDict


def convertToJson(aResult,bResult):
    
    json_data = OrderedDict()
    #sender
    sender_list = []

    for s_name,s_count in aResult[0][0].items():
        temp=OrderedDict()
        temp["name"]= s_name
        temp["count"]=int(s_count)
        sender_list.append(temp)

    #ratio
    ratio_list=[]
    for r_name,r_count in aResult[0][1].items():
        temp=OrderedDict()
        temp["name"]= r_name
        temp["count"]=int(r_count)
        ratio_list.append(temp)

    temp_value=500
    #topic - korean
    topic_list=[]
    for t_name in aResult[1][0]:
        temp=OrderedDict()
        temp["text"]= t_name
        temp["value"]=int(temp_value)
        temp_value=temp_value-5
        topic_list.append(temp)

    #topic - english
    for t_name in bResult[0][0]:
        temp=OrderedDict()
        temp["text"]= t_name
        temp["value"]=int(temp_value)
        temp_value=temp_value-5
        topic_list.append(temp)

    delete_list=[]
    #delete
    for d_mails in bResult[1]:
        strings = d_mails.split('<')
        temp=OrderedDict()
        temp["name"]= strings[0]
        temp["address"]=strings[1]
        delete_list.append(temp)

    json_data["sender"]=sender_list
    json_data["ratio"]=ratio_list
    json_data["topic"]=topic_list
    json_data["delete"]=delete_list

    json_data=json.dumps(json_data , ensure_ascii=False , indent='\t')
 
    # Convert json to dict
    mail_dict = json.loads(json_data)
    return mail_dict