import numpy as np 

#solve the lambda that satisfies the constraints on a distribution given a large deviation
#Constraint: 3.0 < X < 3.2
#ACTIVE CONSTRAINT: X < 3.2

dice = [1,2,3,4,5,6]

#Gradient descent to calculate lambda
def gradientDescent():
    
    alpha = 0.1 #Step Size
    theta = 3.2 #Theta
    epsilon = 0.00001 #Very small number to check convergence
    lmda = np.random.randn() #Randomly initialized Lambda_0
    numSteps = 0
    lmdaNew = None
    converged = False

    #loop until lambda converges
    while converged is not True:
        
        #Update Step
        numSteps += 1
        lmdaNew = lmda - alpha*(expectation(lmda) - theta)
        
        #Convergence Check
        if np.abs(lmdaNew - lmda) < epsilon:
            converged = True
        
        lmda = lmdaNew
    
    #Check Constraint Satisfaction
    print("Steps to converge: ", numSteps)
    print("E[X]: ", expectation(lmda))

    return lmda

#expectation of X under Gibbs representationof p*
def expectation(lmda):
    
    numerator = np.sum([x*np.exp(x*lmda) for x in dice]) #This is x*e^{lambdax}
    denominator = np.sum([np.exp(x*lmda) for x in dice]) #This is Z_lmda
    expectation = numerator/denominator

    return expectation

def pStarDistribution(lmda):

    numerator = [np.exp(x*lmda) for x in dice] #This is e^{lambda*x}
    denominator = np.sum([np.exp(x*lmda) for x in dice]) #This is Z_lmda
    pstar = numerator/denominator

    #Check the distribution sums to one
    print('Pstar sum: ', np.sum(pstar))
    return pstar


if __name__ == "__main__":
   
    #Find and print the minimizing Lambda
    lmda = gradientDescent()
    print("Lambda: ", lmda)

    #Using Lambda, find and print the empirical distribution
    pstar = pStarDistribution(lmda)
    print("Pstar Distribution: ",pstar)

    



