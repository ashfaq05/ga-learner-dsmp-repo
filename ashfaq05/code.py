# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
path
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
#Code starts here
data=np.genfromtxt(path,delimiter=",", skip_header=1)

census=np.concatenate((data,new_record))
print(census)


# --------------
import numpy as np
import statistics as std
#Code starts here

age=np.array(census[:,0])
max_age=np.max(age)
min_age=np.min(age)
age_mean=np.mean(age)
age_std=np.std(age)
print(max_age,min_age,round(age_mean,2),round(age_std,2))


# --------------
#Code starts here
import numpy as np 
import pandas as pd

race_0=census[census[:,2]==0]
len_0=len(race_0)
race_1=census[census[:,2]==1]
len_1=len(race_1)
race_2=census[census[:,2]==2]
len_2=len(race_2)
race_3=census[census[:,2]==3]
len_3=len(race_3)
race_4=census[census[:,2]==4]
len_4=len(race_4)
lis=[len_0,len_1,len_2,len_3,len_4]
minority_race=lis.index(min(lis))
print(minority_race)


# --------------
#Code starts here
import numpy as np
#a=np.array[[1,2,3],[1,3,4],[2,2,5]]
#print(a[a[:,0]==1][:,[0,1]])
#census[census[:,0]>=60][:,[6]]


senior_citizens=census[census[:,0]>60]

#print(working_hours_sum)
working_hours_sum=census[census[:,0]>60][:,[6]].sum()


#a[a[:,0]==1][:,[0,1]]
print(working_hours_sum)

senior_citizens_len=len(senior_citizens)
print(senior_citizens_len)

avg_working_hours=working_hours_sum / senior_citizens_len
print(avg_working_hours)




# --------------
#Code starts here
high=census[census[:,1]>10]
low=census[census[:,1]<10]

avg_pay_high=census[census[:,1]>10][:,[7]].mean()
#working_hours_sum=census[census[:,0]>60][:,[6]].sum()
print(round(avg_pay_high,2))
avg_pay_low=census[census[:,1]<=10][:,[7]].mean()
print(round(avg_pay_low,2))


