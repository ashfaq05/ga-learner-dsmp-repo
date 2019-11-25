# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 


path
# code starts here
bank= pd.read_csv(path)

categorical_var=bank.select_dtypes(include = 'object')
print(categorical_var.head(5))

numerical_var=bank.select_dtypes(include = 'number')
print(numerical_var)
# code ends here


# --------------
# code starts here

bank=pd.read_csv(path)
bank.drop(['Loan_ID'],inplace=True,axis=1)
banks=bank
print(banks.isnull().sum())
bank_mode=banks.mode()
banks.fillna("bank_mode", inplace = True)
print(banks)



# --------------



avg_loan_amount = pd.pivot_table(banks, values='LoanAmount', index=[ 'Gender','Married','Self_Employed'],
                     aggfunc=np.mean)
print(avg_loan_amount)




# --------------

loan_approved_se=banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')]['Self_Employed'].count()
print(loan_approved_se)
loan_approved_nse=banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')]['Self_Employed'].count()
print(loan_approved_nse)
Loan_Status = banks['Loan_Status'].count()
print(Loan_Status)
percentage_se = (loan_approved_se/Loan_Status) *100
print(percentage_se)
percentage_nse =   (loan_approved_nse/Loan_Status) *100
print(percentage_nse)


# --------------
# code starts here


loan_term=banks['Loan_Amount_Term'].apply(lambda x:x/12)
print(loan_term)


big_loan_term=loan_term.apply(lambda x: x>=25).value_counts().loc[True]
print(big_loan_term)

# code ends here


# --------------
# code starts here

loan_groupby=banks.groupby('Loan_Status')

loan_groupby=loan_groupby['ApplicantIncome', 'Credit_History']

mean_values=loan_groupby.mean()
# code ends here


