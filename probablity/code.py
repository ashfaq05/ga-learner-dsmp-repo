# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
fico=df['fico'].loc[df['fico']>700]
total=len(df)
p_a=len(fico)/total
print(p_a)
purpose=df['purpose'].loc[df['purpose']=='debt_consolidation']
p_b=len(purpose)/total
print(p_b)

df1=df['purpose'].loc[(df['purpose']=='debt_consolidation') & (df['fico']>700)]
p_a_b=len(df1)/len(df)
print(p_a_b)
result=p_a_b==p_a*p_b
print(result)
# codresulte ends here


# --------------

df=pd.read_csv(path)
#p(A)
prob_lp=len(df.loc[df['paid.back.loan']=='Yes'])/len(df)

#p(B)
prob_cs=len(df.loc[df['credit.policy']=='Yes'])/len(df)

new_df=df.loc[df['paid.back.loan']=='Yes']

prob_pd_cs =len(new_df[new_df['credit.policy'] == 'Yes'])/len(new_df)

bayes = (prob_pd_cs*prob_lp)/prob_cs
print(round(bayes,2))


# --------------


df1=df.loc[df['paid.back.loan'] == 'No']
x=df1['purpose'].value_counts()
x.plot(kind='bar')


# --------------
# code starts here

inst_median=len(df['installment'])+1/2


inst_mean=sum(df['installment'])/len(df['installment'])

# code ends here
df['installment'].plot(kind='hist')
df['log.annual.inc'].plot(kind='hist')



