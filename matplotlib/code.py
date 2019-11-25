# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path 
data=pd.read_csv(path)

#Code starts here


loan_status=data['Loan_Status'].value_counts()
loan_status.plot(kind='bar')


# --------------
#Code starts here
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(path)

#property_and_loan=data.groupby(['Property_Area','Loan_Status'])
property_and_loan=data.groupby(['Property_Area','Loan_Status']).size().unstack()
property_and_loan.plot(kind='bar',stacked=False)
#property_and_loan=property_and_loan.size().unstack()
plt.xlabel('Property Area') 
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here



education_and_loan=data.groupby(['Education','Loan_Status']).size().unstack()
education_and_loan.plot(kind='bar')

plt.xlabel('Education Status') 
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()



# --------------
#Code starts here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

graduate=data[data['Education'] == 'Graduate']

not_graduate=data[data['Education'] == 'Not Graduate']

graduate.plot(kind='density',label='Graduate')

not_graduate.plot(kind='density',label='not_graduate')

plt.show()



#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(path)
fig ,(ax_1,ax_2,ax_3)=plt.subplots(nrows = 3 , ncols = 1)
plt.subplot(2,2,1)
plt.scatter(data['ApplicantIncome'],data['LoanAmount'],label="ax_1")
plt.title('Applicant Income')

plt.subplot(2,2,2)
plt.scatter(data['CoapplicantIncome'],data['LoanAmount'],label="ax_2")
plt.title('Coapplicant Income')


data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome'] 
plt.subplot(2,2,3)
plt.scatter(data['TotalIncome'],data['LoanAmount'], label="ax_3")
plt.title('Total Income')







