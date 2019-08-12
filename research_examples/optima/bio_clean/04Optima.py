import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.distributions as dist
import biogeme.results as res

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
Work = DefineVariable('Work',TripPurpose ==1,database)

individualHouse = DefineVariable('individualHouse',\
                                 HouseType == 1,database)

ScaledIncome = DefineVariable('ScaledIncome',\
                              CalculatedIncome / 1000,database)
### Coefficients
# Read the estimates from the structural equation estimation
structResults = res.bioResults(pickleFile='02Optima.pickle')
structBetas = structResults.getBetaValues()

coef_intercept = structBetas['coef_intercept']

### Latent variable: structural equation


omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s',1,-1000,1000,0)


# Here is an added random term omega, which will need to be integrated in final utility specification.
# Correct?

CARLOVERS = \
coef_intercept +\
sigma_s * omega


# Choice model


ASC_CAR	 = Beta('ASC_CAR',0,-10000,10000,0)
ASC_PT	 = Beta('ASC_PT',0,-10000,10000,1)
ASC_SM	 = Beta('ASC_SM',0,-10000,10000,0)
BETA_COST_HWH = Beta('BETA_COST_HWH',0.0,-10000,10000,0 )
BETA_DIST	 = Beta('BETA_DIST',0.0,-10000,10000,0)
BETA_TIME_CAR = Beta('BETA_TIME_CAR',0.0,-10000,0,0)
BETA_TIME_PT = Beta('BETA_TIME_PT',0.0,-10000,0,0 )
BETA_LATENT = Beta('BETA_LATENT',0.0,-10000,0,0 )

BETA_STUDENT = Beta('BETA_STUDENT',0.0,-10000,10000,0 )
BETA_URBAN = Beta('BETA_URBAN',0.0,-10000,10000,0 )
BETA_NbChild = Beta('BETA_NbChild', 0.0, -10000, 10000, 0)
BETA_NbCar = Beta('BETA_NbCar', 0.0, -10000, 10000, 0)
BETA_Work = Beta('BETA_Work', 0.0, -10000, 10000, 0)
BETA_French = Beta('BETA_French', 0.0, -10000, 10000, 0)
BETA_NbBicy = Beta('BETA_NbBicy', 0.0, -10000, 10000, 0)


# The scale by /200 is not my crashing problem right?
TimePT_scaled  = DefineVariable('TimePT_scaled', TimePT   /  200 ,database)
TimeCar_scaled  = DefineVariable('TimeCar_scaled', TimeCar   /  200 ,database)
MarginalCostPT_scaled  = \
 DefineVariable('MarginalCostPT_scaled', MarginalCostPT   /  10 ,database)
CostCarCHF_scaled  = \
 DefineVariable('CostCarCHF_scaled', CostCarCHF   /  10 ,database)
distance_km_scaled  = \
 DefineVariable('distance_km_scaled', distance_km   /  5 ,database)
PurpHWH = DefineVariable('PurpHWH', TripPurpose == 1,database)
PurpOther = DefineVariable('PurpOther', TripPurpose != 1,database)

### DEFINITION OF UTILITY FUNCTIONS:


V0 = ASC_PT + \
     BETA_TIME_PT * TimePT_scaled + \
     BETA_COST_HWH * MarginalCostPT_scaled/ScaledIncome  +\
     BETA_STUDENT * Student +\
     BETA_URBAN * Urban


V1 = ASC_CAR + \
      BETA_TIME_CAR * TimeCar_scaled + \
      BETA_COST_HWH * CostCarCHF_scaled/ScaledIncome  +\
      BETA_NbChild * NbChild +\
      BETA_NbCar * NbCar +\
      BETA_Work * Work +\
      BETA_French * French +\
      BETA_LATENT * CARLOVERS * TimeCar_scaled



V2 = ASC_SM + BETA_DIST * distance_km_scaled + BETA_NbBicy * NbBicy


# Associate utility functions with the numbering of alternatives
V = {0: V0,
     1: V1,
     2: V2}

# Associate the availability conditions with the alternatives.
# In this example all alternatives are available for each individual.
av = {0: 1,
      1: 1,
      2: 1}

# The choice model is a logit, conditional to the value of the latent variable

# Do the logit:
condprob = models.logit(V,av,Choice)

#Integrate logit by the random term of carlovers
#Correct?
prob = Integrate(condprob * density,'omega')
loglike = log(prob)
biogeme  = bio.BIOGEME(database,loglike)
biogeme.modelName = "04Optima"
results = biogeme.estimate()
print(f"Estimated betas: {len(results.data.betaValues)}")
print(f"Final log likelihood: {results.data.logLike:.3f}")
print(f"Output file: {results.data.htmlFileName}")
results.writeLaTeX()
print(f"LaTeX file: {results.data.latexFileName}")
