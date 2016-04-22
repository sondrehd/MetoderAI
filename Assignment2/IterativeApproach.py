import numpy as np


class forward_backward():
    def __init__(self, evidence):
        #Initialising transition model and observation model
        self.transition_matrix = np.matrix([[0.7, 0.3], [0.3, 0.7]])
        self.observation_matrix = np.matrix([[0.9, 0.2], [0.1, 0.8]])
        self.evidence = evidence
        self.evidence_matrices = []
        #initialising lists to keep track of messages
        self.forward_probabilities = []
        self.backward_probabilities = []
        self.forward_backward_probabilities=[]
        self.generate_evidence_matrices()

    def generate_evidence_matrices(self):
        #Converting the evidence to matrixes
        for element in self.evidence:
            if element ==1:
                self.evidence_matrices.append(np.matrix([[0.9,0.0],[0.0,0.2]]))
            if element ==0:
                self.evidence_matrices.append(np.matrix([[0.1,0.0],[0.0,0.8]]))

    def calculate_forward_probabilities(self,k):
        #fixing shape
        priori = np.array([0.5, 0.5])[:, None]
        #appending to message list
        self.forward_probabilities.append(priori)
        #Going through each day
        for i in range(k):
            #multiplying evidence matrix with transition matrix
            distribution = np.dot(self.evidence_matrices[i],self.transition_matrix)
            #multiplying result from previous calculation with forward probability message from day before
            distribution_w_forward_p =  distribution* self.forward_probabilities[i]
            #normalizing
            distribution_w_forward_p *= 1/(distribution_w_forward_p[0]+distribution_w_forward_p[1])
            #Appending to message list
            self.forward_probabilities.append(distribution_w_forward_p)

    def calculate_backward_probabilities(self,k):
        #fixing shape and initialising array with distribution for day after last
        posterior = np.array([1.0, 1.0])[:, None]
        #adding to messages
        self.backward_probabilities.append(posterior)
        #looping through all days
        for i in range (k-1,-1, -1):
            #multiplying transition matrix with evidence matrices and probability from day after
            distribution = np.dot(self.transition_matrix,self.evidence_matrices[i])*self.backward_probabilities[-1]
            #normalizing
            distribution *= 1/(distribution[0]+distribution[1])
            self.backward_probabilities.append(distribution)

        self.backward_probabilities = list(reversed(self.backward_probabilities))

    def calculate_forward_backward_probabilities(self,k):
        self.calculate_backward_probabilities(k)
        self.calculate_forward_probabilities(k)

        for i in range(len(self.forward_probabilities)):
            #multiply backward and forward probabilities
            unnormalized = np.multiply(self.backward_probabilities[i],self.forward_probabilities[i])
            #normalize
            normalized = unnormalized* 1/(unnormalized[0]+unnormalized[1])
            self.forward_backward_probabilities.append(normalized)


ob = forward_backward([1, 1])
ob.calculate_forward_probabilities(2)
print("Part B")
print ("Distribution of X2 given umbrella on day 1 and two: ")
print(ob.forward_probabilities[-1])
print("Probability of rain day five given sequence of evidence")
ob = forward_backward([1, 1, 0, 1, 1])
ob.calculate_forward_probabilities(5)
print (ob.forward_probabilities[-1][0])
print("Forward probabilities: ")
print (ob.forward_probabilities)
print("Part C:")
ob = forward_backward([1, 1])
ob.calculate_forward_backward_probabilities(2)
print("Distribution of X1 given e1,2:")
print(ob.forward_backward_probabilities[1])
ob = forward_backward([1, 1, 0, 1, 1])
ob.calculate_forward_backward_probabilities(5)
print("Probability of rain on day one given e1-5:")
print (ob.forward_backward_probabilities[1])
print("Backwardmessages:")
print(ob.forward_backward_probabilities)
