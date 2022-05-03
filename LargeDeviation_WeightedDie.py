import numpy as np 

#solve the lambda that satisfies the constraints on a distribution given a large deviation
#Constraint: 3.6 < X < 3.8
#Constraint: U  < 17

#ACTIVE CONSTRAINT: X^2 < 17

dice = [1,2,3,4,5,6]
weights = [.1,.1,.2,.1,.2,.3]

#Gradient descent to calculate lambda
def gradientDescent():
    
    alpha = 0.01 #Step Size
    theta = 17 #Theta   
    epsilon = 0.00001 #Very small number to check convergence
    lmda = np.random.randn() #Randomly initialized Lambda_0
    numSteps = 0
    lmdaNew = None
    converged = False

    #loop until lambda converges
    while converged is not True:
        
        #Update Step
        numSteps += 1
        lmdaNew = lmda - alpha*(expectation2(lmda) - theta)
        
        #Convergence Check
        if np.abs(lmdaNew - lmda) < epsilon:
            converged = True
            print('converged')
        
        lmda = lmdaNew

        if numSteps % 10000 == 0:
            print("step: ", numSteps)
            print(lmda)
    
    #Check Constraint Satisfaction
    print("Steps to converge: ", numSteps)
    # print("E[X]: ", expectation1(lmda))
    print("E[X^2]: ", expectation2(lmda))

    return lmda

#expectation of X under Gibbs representationof p*
def expectation1(lmda):
    
    numerator = np.sum([weights[x]*dice[x]*np.exp(dice[x]*lmda) for x in range(6)]) #This is h(x)*x*e^{lambda*x}
    denominator = np.sum([weights[x]*np.exp(dice[x]*lmda) for x in range(6)]) #This is Z_lmda
    expectation = numerator/denominator

    return expectation

#expectation of X^2 under Gibbs representationof p*
def expectation2(lmda):
    
    numerator = np.sum([weights[x]*(dice[x]**2)*np.exp((dice[x]**2)*lmda) for x in range(6)]) #This is h(x)*(x^2)*e^{lambda*(x^2)}
    denominator = np.sum([weights[x]*np.exp((dice[x]**2)*lmda) for x in range(6)]) #This is Z_lmda
    expectation = numerator/denominator

    return expectation

def pStarDistribution(lmda):

    numerator = [weights[x]*np.exp((dice[x]**2)*lmda) for x in range(6)] #This is h(x)*e^{lambda*x}
    denominator = np.sum([weights[x]*np.exp((dice[x]**2)*lmda) for x in range(6)]) #This is Z_lmda
    pstar = numerator/denominator

    #Check the distribution sums to one
    print('Pstar sum: ', np.sum(pstar))
    return pstar


def main():
    #Find and print the minimizing Lambda
    lmda = gradientDescent()
    print("Lambda: ", lmda)

    #Using Lambda, find and print the empirical distribution
    pstar = pStarDistribution(lmda)
    print("Pstar Distribution: ",pstar)

    #Chack that Pstar satisfies the other constraint X < 3.8
    X = np.sum([dice[x]*pstar[x] for x in range(6)])
    print("E[X]: ", X)


if __name__ == "__main__":
    main()
    



