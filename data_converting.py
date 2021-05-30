import numpy as np
import pandas as pd

filename = "truncated_data.csv"
print("reading...")
csv_data = pd.read_csv(filename, low_memory = False)
credit = pd.DataFrame(csv_data)
df2 = credit.copy()
df2["NAICS"] = [str(fe)[0:2] for fe in df2["NAICS"]]
credit["NAICS"] = df2['NAICS']
print(credit['NAICS'])
def count(feature):
    dict_ = credit[feature].value_counts()
    fes = list(dict_.keys())
    print(fes)
    new_dic = {}
    for fe in fes:
        num = 0
        df2 = credit.copy()
        df2 = df2[(df2["MIS_Status"] == "CHGOFF")&(df2[feature] == fe)]
        num = len(df2)     
        new_dic[fe] = round(num/dict_[fe],4)
    print([float(i*100) for i in new_dic.values()])
    lis = [(fe, new_dic[fe]) for fe in fes]
    lis.sort(key=lambda x:x[1], reverse=True)
    print([item[0] for item in lis])
    df2 = credit.copy()
    df2[feature] = [new_dic[fe] for fe in df2[feature]]
    credit[feature] = df2[feature]

def count_double(feature1,feature2):
    data = []
    dict_ = credit[feature1].value_counts()
    fes = list(dict_.keys())
    dict_2 = credit[feature2].value_counts()
    # fes2 = list(dict_2.keys())
    fes2 = ['VA', 'FL', 'OR', 'SD', 'IL',  'DE', 'SC', 'NC', 'AL', 'RI', 'OH', 'NY', 'UT', 'AK', 'TX', 'MO', 'ID', 'WI', 'IA', 'GA', 'TN', 'DC', 'AR', 'MI', 'MN', 'AZ', 'CT', 'WA', 'WV', 'KS', 'NJ', 'OK', 'MD', 'NE', 'NV', 'NM', 'KY', 'MA', 'HI', 'MS', 'NH', 'PA', 'IN', 'CO', 'ND', 'ME', 'VT', 'LA', 'MT', 'WY']
    print(fes2)
    new_dic = {}
    for i,fe in enumerate(fes):
        for j, fe2 in enumerate(fes2):
            num = 0
            df2 = credit.copy()
            df4 = df2[(df2[feature1] == fe)&(df2[feature2] == fe2)]
            df3 = df2[(df2["MIS_Status"] == "CHGOFF")&(df2[feature1] == fe)&(df2[feature2] == fe2)]
            num = len(df3)     
            # print(num)
            if num == 0:
                data.append([j,i,0])
            else:
                data.append([j,i,int(round(num/len(df4),2) * 100)])
            # print(data)

    print(data)

# this part is for "converted_data"
# count("State")
# count("Bank")
# count("BankState")
# count("NAICS")
# count("NewExist")
# count("UrbanRural")
# count("LowDoc")
# count_double("NewExist", "NAICS")
# count_double("LowDoc", "BankState")
# credit.to_csv("converted_data.csv")

# this part is for "converted_for_other"
# filename = "converted_data.csv"
# print("reading...")
# csv_data = pd.read_csv(filename, low_memory = False)#防止弹出警告
# credit = pd.DataFrame(csv_data)
# df2 = credit.copy()
# df2['FranchiseCode'] = [1 if int(item) > 1 else 0 for item in df2["FranchiseCode"]]
# df2['RevLineCr'] = [1 if item == "Y" else 0 for item in df2['RevLineCr']]
# df2['LowDoc'] = [1 if item == "Y" else 0 for item in df2['LowDoc']]
# df2['MIS_Status'] = [1 if item == "CHGOFF" else 0 for item in df2['MIS_Status']]
# df2["DisbursementGross"] = [np.log(float(item[1:].replace(",","").strip())) for item in df2["DisbursementGross"]]
# df2["GrAppv"] = [np.log(float(item[1:].replace(",","").strip())) for item in df2["GrAppv"]]
# df2["SBA_Appv"] = [np.log(float(item[1:].replace(",","").strip())) for item in df2["SBA_Appv"]]
# df2 = df2.drop(axis =1, columns = ['Zip','Loan_id','ApprovalDate', 'ApprovalFY',"id","City","Name","ChgOffDate","DisbursementDate",\
#     "BalanceGross","ChgOffPrinGr"])
# credit = df2
# credit.to_csv("converted_for_other.csv",index=False)

# dataset for catboost
count("NAICS")
df2 = credit.copy()
df2['FranchiseCode'] = [1 if int(item) > 1 else 0 for item in df2["FranchiseCode"]]
df2['MIS_Status'] = [1 if item == "CHGOFF" else 0 for item in df2['MIS_Status']]
df2["DisbursementGross"] = [np.log(float(item[1:].replace(",","").strip())) for item in df2["DisbursementGross"]]
df2["GrAppv"] = [np.log(float(item[1:].replace(",","").strip())) for item in df2["GrAppv"]]
df2["SBA_Appv"] = [np.log(float(item[1:].replace(",","").strip())) for item in df2["SBA_Appv"]]
df2 = df2.drop(axis =1, columns = ['Zip','Loan_id','ApprovalDate', 'ApprovalFY',"id","City","Name","ChgOffDate","DisbursementDate",\
    "BalanceGross","ChgOffPrinGr"])
credit = df2
credit.to_csv("converted_for_catboost.csv",index=False)
