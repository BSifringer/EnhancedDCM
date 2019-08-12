import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
#import biogeme.models as models
import biogeme.loglikelihood as ll
import biogeme.distributions as dist



pandas = pd.read_table("optima_train.dat")
database = db.Database("optima",pandas)

from headers import *


variables = [TimePT, TimeCar, CostPT, distance_km, CostCarCHF,
    TripPurpose, age, NbBicy, NbCar, NbChild, HouseType, Gender, GenAbST,
     Income, Education, SocioProfCat, OwnHouse, FamilSitu, CalculatedIncome]

exclude = (Choice == -1.0)
for variable in variables:
    exclude = exclude +  (variable == -1.0)
# exclude = (Choice == -1.0)
database.remove(exclude)



### Variables

French = DefineVariable('French',LangCode==1,database)
Student = DefineVariable('Student',OccupStat == 8,database)
Urban = DefineVariable('Urban',UrbRur ==2,database)
individualHouse = DefineVariable('individualHouse',\
                                 HouseType == 1,database)
male = DefineVariable('male',Gender == 1,database)
haveChildren = DefineVariable('haveChildren',\
                              ((FamilSitu == 3)+(FamilSitu == 4)) > 0,database)
haveGA = DefineVariable('haveGA',GenAbST == 1,database)
highEducation = DefineVariable('highEducation', Education >= 6,database)

### Coefficients
coef_intercept = Beta('coef_intercept',4.0,None,None,0 )

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to
# write it across several lines, each line must terminate with
# the \ symbol

CARLOVERS = \
coef_intercept #+\
#sigma_s * omega


### Measurement equations

INTER_DifficultPT = Beta('INTER_DifficultPT',0,None,None,1)
INTER_EasyCAR = Beta('INTER_EasyCAR',1.0,None,None,0 )

B_DifficultPT = Beta('B_DifficultPT',1,None,None,1)
B_EasyCAR = Beta('B_EasyCAR',1.0,None,None,0 )


MODEL_DifficultPT = INTER_DifficultPT + B_DifficultPT * CARLOVERS * TimeCar
MODEL_EasyCAR = INTER_EasyCAR + B_EasyCAR * CARLOVERS * TimeCar

SIGMA_STAR_DifficultPT = Beta('SIGMA_STAR_DifficultPT',1,None,None,1)
SIGMA_STAR_EasyCAR = Beta('SIGMA_STAR_EasyCAR',1.0,0.0001,None,0 )


#Here is the Regression t*I = alpha + lambda* t_car *carlover + sigma
# Correct? 
F = {}
F['Envir01'] = Elem({0:0, \
 1:ll.loglikelihoodregression( Mobil10*TimeCar ,MODEL_DifficultPT,SIGMA_STAR_DifficultPT)},\
          (Mobil10 > 0)*(Mobil10 < 6))
F['Envir02'] = Elem({0:0, \
 1:ll.loglikelihoodregression(Mobil13*TimeCar,MODEL_EasyCAR,SIGMA_STAR_EasyCAR)},\
  (Mobil13 > 0)*(Mobil13 < 6))


loglike = bioMultSum(F)


biogeme  = bio.BIOGEME(database,loglike)
biogeme.modelName = "02Optima"
results = biogeme.estimate()
print(f"Estimated betas: {len(results.data.betaValues)}")
print(f"final log likelihood: {results.data.logLike:.3f}")
print(f"Output file: {results.data.htmlFileName}")
results.writeLaTeX()
print(f"LaTeX file: {results.data.latexFileName}")
